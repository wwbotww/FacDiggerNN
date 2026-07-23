from __future__ import annotations

import json
import math
from datetime import date, timedelta

import polars as pl
import pytest

from facdigger.data.config import DatasetBuildConfig, SplitConfig
from facdigger.data.contracts import (
    DataContractError,
    validate_bars,
    validate_delistings,
    validate_universe,
)
from facdigger.data.snapshots import build_dataset_snapshot
from facdigger.datasets.splits import assign_chronological_splits
from facdigger.features.price_volume import build_price_volume_features
from facdigger.features.scaling import fit_train_robust_scaler
from facdigger.labels.forward_return import build_forward_excess_return_labels


def sessions(count: int) -> list[date]:
    result: list[date] = []
    current = date(2020, 1, 1)
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current)
        current += timedelta(days=1)
    return result


def synthetic_frames(count: int = 90) -> tuple[pl.DataFrame, pl.DataFrame]:
    trading_days = sessions(count)
    bars: list[dict] = []
    universe: list[dict] = []
    for security_id, base, slope in [("sec-a", 100.0, 0.10), ("sec-b", 80.0, 0.05)]:
        for index, trade_date in enumerate(trading_days):
            close = base + slope * index
            symbol = "AAA2" if security_id == "sec-a" and index >= 45 else security_id[-1].upper()
            bars.append(
                {
                    "security_id": security_id,
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "open": close - 0.05,
                    "high": close + 0.20,
                    "low": close - 0.20,
                    "close": close,
                    "volume": 1_000_000.0 + index,
                    "dollar_volume": close * (1_000_000.0 + index),
                    "adj_factor": 1.0,
                    "source_revision": "synthetic-v1",
                }
            )
            universe.append(
                {
                    "security_id": security_id,
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "listed_days": 600 + index,
                    "exchange": "XNAS",
                    "security_type": "common_stock",
                    "is_primary_listing": True,
                    "is_listed": True,
                    "is_delisted": False,
                    "is_halted": False,
                    "industry_code": "TECH",
                    "float_market_cap": 1_000_000_000.0,
                    "close": close,
                    "adv20_usd": 10_000_000.0,
                    "eligible": True,
                }
            )
    return pl.DataFrame(bars), pl.DataFrame(universe)


def test_contract_rejects_eligible_halted_security() -> None:
    bars, universe = synthetic_frames(5)
    validate_bars(bars)
    universe = universe.with_columns(
        pl.when((pl.col("security_id") == "sec-a") & (pl.col("trade_date") == sessions(5)[0]))
        .then(True)
        .otherwise(pl.col("is_halted"))
        .alias("is_halted")
    )
    with pytest.raises(DataContractError, match="eligible despite"):
        validate_universe(universe)


def test_features_do_not_change_when_future_prices_change() -> None:
    bars, universe = synthetic_frames(50)
    cutoff = sessions(50)[30]
    original = build_price_volume_features(validate_bars(bars), validate_universe(universe))
    mutated_bars = bars.with_columns(
        pl.when(pl.col("trade_date") > cutoff)
        .then(pl.col("close") * 100)
        .otherwise(pl.col("close"))
        .alias("close")
    ).with_columns(pl.max_horizontal("high", "close").alias("high"))
    mutated = build_price_volume_features(validate_bars(mutated_bars), validate_universe(universe))
    keys = (pl.col("security_id") == "sec-a") & (pl.col("trade_date") == cutoff)
    assert original.filter(keys).to_dicts() == mutated.filter(keys).to_dicts()


def test_scaler_does_not_fit_validation_or_test_values() -> None:
    bars, universe = synthetic_frames(70)
    cutoff = sessions(70)[35]
    original = build_price_volume_features(validate_bars(bars), validate_universe(universe))
    mutated_bars = bars.with_columns(
        pl.when(pl.col("trade_date") > cutoff)
        .then(pl.col("dollar_volume") * 1_000_000)
        .otherwise(pl.col("dollar_volume"))
        .alias("dollar_volume")
    )
    mutated = build_price_volume_features(validate_bars(mutated_bars), validate_universe(universe))
    original_scaler = fit_train_robust_scaler(original, list(original.columns[3:10]), cutoff)
    mutated_scaler = fit_train_robust_scaler(mutated, list(mutated.columns[3:10]), cutoff)
    assert original_scaler == mutated_scaler


def test_forward_label_matches_execution_definition_and_cross_sectional_benchmark() -> None:
    bars, universe = synthetic_frames(40)
    bars = validate_bars(bars)
    universe = validate_universe(universe)
    labels = build_forward_excess_return_labels(bars, universe, horizon=5)
    asof = sessions(40)[20]
    rows = labels.filter(pl.col("asof_date") == asof).sort("security_id").to_dicts()
    raw_a = math.log((100 + 0.1 * 25) / (100 + 0.1 * 21 - 0.05))
    raw_b = math.log((80 + 0.05 * 25) / (80 + 0.05 * 21 - 0.05))
    benchmark = (raw_a + raw_b) / 2
    assert rows[0]["raw_return"] == pytest.approx(raw_a)
    assert rows[0]["target"] == pytest.approx(raw_a - benchmark)
    assert rows[1]["target"] == pytest.approx(raw_b - benchmark)


def test_embargo_skips_configured_sessions() -> None:
    bars, universe = synthetic_frames(70)
    labels = build_forward_excess_return_labels(
        validate_bars(bars), validate_universe(universe), horizon=5
    )
    calendar = sessions(70)
    split = SplitConfig(
        train_end=calendar[25],
        valid_end=calendar[45],
        test_end=calendar[65],
        embargo_sessions=2,
    )
    assigned = assign_chronological_splits(labels, calendar, split)
    first_valid = assigned.filter(pl.col("split") == "valid")["asof_date"].min()
    first_test = assigned.filter(pl.col("split") == "test")["asof_date"].min()
    assert first_valid == calendar[28]
    assert first_test == calendar[48]
    assert assigned.filter(
        (pl.col("split") == "train") & (pl.col("label_end") > split.train_end)
    ).is_empty()


def test_delisting_return_is_included_in_overlapping_label() -> None:
    bars, universe = synthetic_frames(30)
    calendar = sessions(30)
    bars = bars.filter((pl.col("security_id") != "sec-a") | (pl.col("trade_date") <= calendar[14]))
    universe = universe.with_columns(
        pl.when((pl.col("security_id") == "sec-a") & (pl.col("trade_date") >= calendar[15]))
        .then(False)
        .otherwise(pl.col("is_listed"))
        .alias("is_listed"),
        pl.when((pl.col("security_id") == "sec-a") & (pl.col("trade_date") >= calendar[15]))
        .then(True)
        .otherwise(pl.col("is_delisted"))
        .alias("is_delisted"),
        pl.when((pl.col("security_id") == "sec-a") & (pl.col("trade_date") >= calendar[15]))
        .then(False)
        .otherwise(pl.col("eligible"))
        .alias("eligible"),
    )
    delistings = validate_delistings(
        pl.DataFrame(
            {
                "security_id": ["sec-a"],
                "delist_date": [calendar[15]],
                "last_trade_date": [calendar[14]],
                "delisting_return": [-0.5],
                "terminal_value": [None],
                "known_at": [calendar[15]],
                "source_revision": ["synthetic-v1"],
            },
            schema_overrides={"terminal_value": pl.Float64},
        )
    )
    labels = build_forward_excess_return_labels(
        validate_bars(bars),
        validate_universe(universe),
        delistings=delistings,
        horizon=5,
    )
    row = labels.filter(
        (pl.col("security_id") == "sec-a") & (pl.col("asof_date") == calendar[10])
    ).row(0, named=True)
    expected_terminal = (100 + 0.1 * 14) * 0.5
    expected_entry = 100 + 0.1 * 11 - 0.05
    assert row["crosses_delisting"] is True
    assert row["raw_return"] == pytest.approx(math.log(expected_terminal / expected_entry))


def test_snapshot_build_is_content_addressed_and_idempotent(tmp_path) -> None:
    bars, universe = synthetic_frames(90)
    bronze = tmp_path / "bronze"
    bronze.mkdir()
    bars_path = bronze / "bars.parquet"
    universe_path = bronze / "universe.parquet"
    source_manifest_path = bronze / "source_manifest.json"
    bars.write_parquet(bars_path)
    universe.write_parquet(universe_path)
    source_manifest_path.write_text('{"provider":"synthetic"}\n', encoding="utf-8")
    calendar = sessions(90)
    config = DatasetBuildConfig.model_validate(
        {
            "sources": {
                "bars": bars_path,
                "universe": universe_path,
                "source_manifest": source_manifest_path,
            },
            "output_root": tmp_path / "snapshots",
            "features": {"context_length": 20},
            "split": {
                "train_end": calendar[35],
                "valid_end": calendar[58],
                "test_end": calendar[82],
                "embargo_sessions": 2,
            },
        }
    )

    first_dir, first_manifest = build_dataset_snapshot(config)
    second_dir, second_manifest = build_dataset_snapshot(config)

    assert first_dir == second_dir
    assert first_manifest == second_manifest
    assert first_dir.name == first_manifest["dataset_id"]
    for filename in [
        "features.parquet",
        "labels.parquet",
        "sample_index.parquet",
        "sample_metadata.parquet",
        "inference_index.parquet",
        "audit.json",
        "scaler.json",
        "manifest.json",
        "source_manifest.json",
    ]:
        assert (first_dir / filename).is_file()
    audit = json.loads((first_dir / "audit.json").read_text(encoding="utf-8"))
    assert audit["sample_index"]["rows"] > 0
    assert set(audit["sample_index"]["split_counts"]) == {"test", "train", "valid"}
    assert json.loads((first_dir / "source_manifest.json").read_text())["provider"] == ("synthetic")
    inference_index = pl.read_parquet(first_dir / "inference_index.parquet")
    sample_index = pl.read_parquet(first_dir / "sample_index.parquet")
    assert first_manifest["schema_version"] == 3
    assert "target" not in inference_index.columns
    assert inference_index["asof_date"].max() == calendar[-1]
    assert inference_index["asof_date"].max() > sample_index["asof_date"].max()
    assert audit["inference_index"]["contains_target"] is False

    moved_config = config.model_copy(update={"output_root": tmp_path / "other-snapshots"})
    moved_dir, moved_manifest = build_dataset_snapshot(moved_config)
    assert moved_dir.parent != first_dir.parent
    assert moved_manifest["dataset_id"] == first_manifest["dataset_id"]
