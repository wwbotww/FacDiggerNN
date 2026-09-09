from __future__ import annotations

import json
import math
from datetime import date, timedelta

import polars as pl
import pytest

from facdigger.data.config import (
    DatasetBuildConfig,
    InferenceSnapshotConfig,
    SplitConfig,
)
from facdigger.data.contracts import (
    DataContractError,
    validate_bars,
    validate_delistings,
    validate_universe,
)
from facdigger.data.inference_snapshots import (
    build_inference_snapshot,
    load_inference_snapshot,
)
from facdigger.data.provenance import build_standardization_contract
from facdigger.data.snapshots import build_dataset_snapshot, sha256_file
from facdigger.datasets.splits import assign_chronological_splits
from facdigger.features.cross_sectional import (
    append_cross_sectional_ranks,
    build_market_context_features,
)
from facdigger.features.price_volume import build_price_volume_features
from facdigger.features.scaling import fit_train_robust_scaler
from facdigger.inference.releases import ModelReleaseManifest
from facdigger.labels.forward_return import (
    build_forward_excess_return_labels,
    build_multi_horizon_excess_return_labels,
)


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


def test_cross_sectional_features_rank_only_eligible_rows_and_build_market_state() -> None:
    bars, universe = synthetic_frames(40)
    target_date = sessions(40)[25]
    universe = universe.with_columns(
        pl.when(
            (pl.col("security_id") == "sec-b")
            & (pl.col("trade_date") == target_date)
        )
        .then(False)
        .otherwise(pl.col("eligible"))
        .alias("eligible")
    )
    raw = build_price_volume_features(validate_bars(bars), validate_universe(universe))
    ranked = append_cross_sectional_ranks(raw, validate_universe(universe))
    target_rows = ranked.filter(pl.col("trade_date") == target_date).sort("security_id")
    assert target_rows["rank_r_close"].null_count() == 2

    fully_eligible = synthetic_frames(40)[1]
    ranked = append_cross_sectional_ranks(raw, validate_universe(fully_eligible))
    target_rows = ranked.filter(pl.col("trade_date") == target_date).sort("security_id")
    assert target_rows["rank_r_close"].to_list() == [1.0, -1.0]
    assert target_rows["observed_rank_r_close"].to_list() == [True, True]

    market = build_market_context_features(raw, validate_universe(fully_eligible))
    row = market.filter(pl.col("trade_date") == target_date).row(0, named=True)
    expected_median = sum(target_rows["r_close"].to_list()) / 2
    assert row["market_return_median"] == pytest.approx(expected_median)
    assert row["market_breadth"] == pytest.approx(1.0)
    assert row["market_return_dispersion"] >= 0
    assert row["observed_market_vol20"] is True


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


def test_multi_horizon_labels_keep_primary_target_and_longest_boundary() -> None:
    bars, universe = synthetic_frames(50)
    labels = build_multi_horizon_excess_return_labels(
        validate_bars(bars),
        validate_universe(universe),
        horizons=[1, 5, 20],
        primary_horizon=5,
    )
    row = labels.filter(
        (pl.col("security_id") == "sec-a")
        & (pl.col("asof_date") == sessions(50)[20])
    ).row(0, named=True)
    assert row["target"] == pytest.approx(row["target_5"])
    assert row["raw_return"] == pytest.approx(row["raw_return_5"])
    assert row["label_end"] == row["label_end_20"]
    assert row["label_end_1"] < row["label_end_5"] < row["label_end_20"]


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


def test_equal_train_and_valid_boundary_creates_final_refit_split() -> None:
    bars, universe = synthetic_frames(70)
    labels = build_forward_excess_return_labels(
        validate_bars(bars), validate_universe(universe), horizon=5
    )
    calendar = sessions(70)
    split = SplitConfig(
        train_end=calendar[45],
        valid_end=calendar[45],
        test_end=calendar[65],
        embargo_sessions=2,
    )

    assigned = assign_chronological_splits(labels, calendar, split)

    assert assigned.filter(pl.col("split") == "valid").is_empty()
    assert assigned.filter(pl.col("split") == "train")["label_end"].max() <= calendar[45]
    assert assigned.filter(pl.col("split") == "test")["asof_date"].min() == calendar[48]


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
    source_manifest_path.write_text(
        json.dumps(
            {
                "provider": "synthetic",
                "standardization": build_standardization_contract(
                    {
                        "bars": {
                            "file": bars_path.name,
                            "sha256": sha256_file(bars_path),
                        },
                        "universe": {
                            "file": universe_path.name,
                            "sha256": sha256_file(universe_path),
                        },
                    },
                    research_ready=True,
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
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
    assert first_manifest["schema_version"] == 4
    assert "target" not in inference_index.columns
    assert inference_index["asof_date"].max() == calendar[-1]
    assert inference_index["asof_date"].max() > sample_index["asof_date"].max()
    assert audit["inference_index"]["contains_target"] is False

    moved_config = config.model_copy(update={"output_root": tmp_path / "other-snapshots"})
    moved_dir, moved_manifest = build_dataset_snapshot(moved_config)
    assert moved_dir.parent != first_dir.parent
    assert moved_manifest["dataset_id"] == first_manifest["dataset_id"]


def test_finance_transformer_snapshot_contains_full_context_and_multi_horizon_targets(
    tmp_path,
) -> None:
    bars, universe = synthetic_frames(90)
    bars_path = tmp_path / "bars.parquet"
    universe_path = tmp_path / "universe.parquet"
    bars.write_parquet(bars_path)
    universe.write_parquet(universe_path)
    calendar = sessions(90)
    channels = [
        "r_close",
        "r_gap",
        "r_intraday",
        "range",
        "dlog_volume",
        "vol20",
        "dollar_volume_z20",
    ]
    config = DatasetBuildConfig.model_validate(
        {
            "sources": {"bars": bars_path, "universe": universe_path},
            "output_root": tmp_path / "snapshots",
            "features": {
                "name": "finance_transformer",
                "context_length": 20,
                "channels": [*channels, *[f"rank_{channel}" for channel in channels]],
                "market_channels": [
                    "market_return_median",
                    "market_breadth",
                    "market_return_dispersion",
                    "market_range_median",
                    "market_volume_activity",
                    "market_vol20",
                ],
            },
            "label": {"horizon": 5, "auxiliary_horizons": [1, 20]},
            "split": {
                "train_end": calendar[40],
                "valid_end": calendar[63],
                "test_end": calendar[87],
                "embargo_sessions": 2,
            },
        }
    )

    snapshot, manifest = build_dataset_snapshot(config)

    assert manifest["artifacts"]["market_features"] == "market_features.parquet"
    features = pl.read_parquet(snapshot / "features.parquet")
    market = pl.read_parquet(snapshot / "market_features.parquet")
    sample_index = pl.read_parquet(snapshot / "sample_index.parquet")
    pretraining_index = pl.read_parquet(snapshot / "pretraining_index.parquet")
    scaler = json.loads((snapshot / "scaler.json").read_text(encoding="utf-8"))
    assert set(config.features.channels).issubset(features.columns)
    assert set(config.features.market_channels).issubset(market.columns)
    assert {"target_1", "target_5", "target_20"}.issubset(sample_index.columns)
    assert not any(
        column.startswith("target") for column in pretraining_index.columns
    )
    assert pretraining_index["future_end"].max() <= config.split.train_end
    assert manifest["artifacts"]["pretraining_index"] == "pretraining_index.parquet"
    assert sample_index.null_count().select(
        "target_1", "target_5", "target_20"
    ).sum_horizontal().item() == 0
    assert set(scaler) == {"local", "market", "method", "rank_channels"}


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"], ids=["unix", "windows"])
def test_inference_snapshot_reuses_release_scaler_without_labels_or_fit(
    tmp_path, monkeypatch, newline
) -> None:
    bars, universe = synthetic_frames(90)
    bars_path = tmp_path / "bars.parquet"
    universe_path = tmp_path / "universe.parquet"
    bars.write_parquet(bars_path)
    universe.write_parquet(universe_path)
    calendar = sessions(90)
    training_snapshot, training_manifest = build_dataset_snapshot(
        DatasetBuildConfig.model_validate(
            {
                "sources": {"bars": bars_path, "universe": universe_path},
                "output_root": tmp_path / "training-snapshots",
                "features": {"context_length": 20},
                "split": {
                    "train_end": calendar[35],
                    "valid_end": calendar[58],
                    "test_end": calendar[82],
                    "embargo_sessions": 2,
                },
            }
        )
    )
    scaler = json.loads((training_snapshot / "scaler.json").read_text(encoding="utf-8"))
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    scaler_path = release_dir / "scaler.json"
    scaler_path.write_bytes(
        (training_snapshot / "scaler.json").read_bytes().replace(b"\n", newline)
    )
    (release_dir / "training_dataset_manifest").write_text(json.dumps(training_manifest))
    release = ModelReleaseManifest.model_validate(
        {
            "release_id": "1" * 64,
            "created_at": "2026-08-12T00:00:00+00:00",
            "model_id": "e3-test",
            "model_type": "financial_pretrained_patchtst",
            "forecast_horizon_sessions": 5,
            "objective": "rank",
            "target_transform": "rank",
            "source": {
                "repository": "FacDiggerNN",
                "commit": "1" * 40,
                "run_id": "run-1",
                "run_manifest_sha256": "2" * 64,
                "predictions_sha256": "5" * 64,
                "git_clean": True,
            },
            "training_data": {
                "dataset_id": training_manifest["dataset_id"],
                "dataset_manifest_sha256": sha256_file(training_snapshot / "manifest.json"),
            },
            "feature_contract": {
                "feature_set": "price_volume_v1",
                "channels": list(scaler["channels"]),
                "context_length": 20,
                "scaler_sha256": sha256_file(scaler_path),
                "scaler_contract": "train_global_robust",
                "identity_policy": "provider_neutral_security_id",
            },
            "artifacts": {
                name: {"file": name, "sha256": "4" * 64, "bytes": 1}
                for name in {
                    "checkpoint",
                    "checkpoint_protocol",
                    "resolved_config",
                    "training_dataset_manifest",
                    "source_run_manifest",
                }
            }
            | {
                "scaler": {
                    "file": "scaler.json",
                    "sha256": sha256_file(scaler_path),
                    "bytes": scaler_path.stat().st_size,
                }
            },
        }
    )
    monkeypatch.setattr(
        "facdigger.data.inference_snapshots.load_model_release", lambda _: release
    )
    monkeypatch.setattr(
        "facdigger.features.pipeline.fit_train_robust_scaler",
        lambda *args, **kwargs: pytest.fail("inference path must not fit a scaler"),
    )

    snapshot, manifest = build_inference_snapshot(
        InferenceSnapshotConfig.model_validate(
            {
                "sources": {"bars": bars_path, "universe": universe_path},
                "output_root": tmp_path / "inference-snapshots",
            }
        ),
        release_dir,
    )

    assert manifest["feature_contract"]["scaler_sha256"] == sha256_file(scaler_path)
    assert "labels" not in manifest["artifacts"]
    assert "sample_index" not in manifest["artifacts"]
    assert (snapshot / "delivery_universe.parquet").is_file()
    assert (snapshot / "scaler.json").read_bytes() == scaler_path.read_bytes()
    assert pl.read_parquet(snapshot / "features.parquet").equals(
        pl.read_parquet(training_snapshot / "features.parquet"),
        null_equal=True,
    )

    loaded_manifest, frames = load_inference_snapshot(snapshot, release)
    assert loaded_manifest["snapshot_id"] == manifest["snapshot_id"]
    assert set(frames) == {"inference_index", "delivery_universe"}
    eligible_universe = frames["delivery_universe"].filter(pl.col("eligible")).select(
        "security_id", "symbol", "asof_date"
    )
    assert eligible_universe.equals(
        frames["inference_index"].select("security_id", "symbol", "asof_date"),
        null_equal=True,
    )


def test_inference_snapshot_loader_rejects_eligibility_drift(
    tmp_path, monkeypatch
) -> None:
    bars, universe = synthetic_frames(50)
    bars_path = tmp_path / "bars.parquet"
    universe_path = tmp_path / "universe.parquet"
    bars.write_parquet(bars_path)
    universe.write_parquet(universe_path)
    scaler = fit_train_robust_scaler(
        build_price_volume_features(validate_bars(bars), validate_universe(universe)),
        [
            "r_close",
            "r_gap",
            "r_intraday",
            "range",
            "dlog_volume",
            "vol20",
            "dollar_volume_z20",
        ],
        sessions(50)[30],
    )
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    (release_dir / "training_dataset_manifest").write_text(
        '{"artifacts": {"source_manifest": null}}'
    )
    scaler_path = release_dir / "scaler.json"
    scaler_path.write_text(
        json.dumps(scaler, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    release = ModelReleaseManifest.model_validate(
        {
            "release_id": "1" * 64,
            "created_at": "2026-08-12T00:00:00+00:00",
            "model_id": "e3-test",
            "model_type": "financial_pretrained_patchtst",
            "forecast_horizon_sessions": 5,
            "objective": "rank",
            "target_transform": "rank",
            "source": {
                "repository": "FacDiggerNN",
                "commit": "1" * 40,
                "run_id": "run-1",
                "run_manifest_sha256": "2" * 64,
                "predictions_sha256": "5" * 64,
                "git_clean": True,
            },
            "training_data": {
                "dataset_id": "training-1",
                "dataset_manifest_sha256": "3" * 64,
            },
            "feature_contract": {
                "feature_set": "price_volume_v1",
                "channels": list(scaler["channels"]),
                "context_length": 20,
                "scaler_sha256": sha256_file(scaler_path),
                "scaler_contract": "train_global_robust",
                "identity_policy": "provider_neutral_security_id",
            },
            "artifacts": {
                name: {"file": name, "sha256": "4" * 64, "bytes": 1}
                for name in {
                    "checkpoint",
                    "checkpoint_protocol",
                    "resolved_config",
                    "training_dataset_manifest",
                    "source_run_manifest",
                }
            }
            | {
                "scaler": {
                    "file": "scaler.json",
                    "sha256": sha256_file(scaler_path),
                    "bytes": scaler_path.stat().st_size,
                }
            },
        }
    )
    monkeypatch.setattr(
        "facdigger.data.inference_snapshots.load_model_release", lambda _: release
    )
    snapshot, _ = build_inference_snapshot(
        InferenceSnapshotConfig.model_validate(
            {
                "sources": {"bars": bars_path, "universe": universe_path},
                "output_root": tmp_path / "inference-snapshots",
            }
        ),
        release_dir,
    )
    delivery_path = snapshot / "delivery_universe.parquet"
    delivery = pl.read_parquet(delivery_path)
    changed_key = (pl.col("security_id") == delivery["security_id"][0]) & (
        pl.col("asof_date") == delivery["asof_date"][0]
    )
    delivery.with_columns(
        pl.when(changed_key)
        .then(~pl.col("eligible"))
        .otherwise(pl.col("eligible"))
        .alias("eligible")
    ).write_parquet(delivery_path)
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["delivery_universe"] = sha256_file(delivery_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataContractError, match="eligibility does not exactly match"):
        load_inference_snapshot(snapshot, release)


def test_exact_date_inference_snapshot_is_partitioned_and_contains_only_target(
    tmp_path,
    monkeypatch,
) -> None:
    bars, universe = synthetic_frames(50)
    bars_path = tmp_path / "bars.parquet"
    universe_path = tmp_path / "universe.parquet"
    bars.write_parquet(bars_path)
    universe.write_parquet(universe_path)
    raw = build_price_volume_features(validate_bars(bars), validate_universe(universe))
    scaler = fit_train_robust_scaler(
        raw,
        [
            "r_close",
            "r_gap",
            "r_intraday",
            "range",
            "dlog_volume",
            "vol20",
            "dollar_volume_z20",
        ],
        sessions(50)[30],
    )
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    (release_dir / "training_dataset_manifest").write_text(
        '{"artifacts": {"source_manifest": null}}'
    )
    scaler_path = release_dir / "scaler.json"
    scaler_path.write_text(json.dumps(scaler, sort_keys=True), encoding="utf-8")
    release = ModelReleaseManifest.model_validate(
        {
            "release_id": "1" * 64,
            "created_at": "2026-08-12T00:00:00+00:00",
            "model_id": "e3-test",
            "model_type": "financial_pretrained_patchtst",
            "forecast_horizon_sessions": 5,
            "objective": "rank",
            "target_transform": "rank",
            "source": {
                "repository": "FacDiggerNN",
                "commit": "1" * 40,
                "run_id": "run-1",
                "run_manifest_sha256": "2" * 64,
                "predictions_sha256": "5" * 64,
                "git_clean": True,
            },
            "training_data": {
                "dataset_id": "training-1",
                "dataset_manifest_sha256": "3" * 64,
            },
            "feature_contract": {
                "feature_set": "price_volume_v1",
                "channels": list(scaler["channels"]),
                "context_length": 20,
                "scaler_sha256": sha256_file(scaler_path),
                "scaler_contract": "train_global_robust",
                "identity_policy": "provider_neutral_security_id",
            },
            "artifacts": {
                name: {"file": name, "sha256": "4" * 64, "bytes": 1}
                for name in {
                    "checkpoint",
                    "checkpoint_protocol",
                    "resolved_config",
                    "training_dataset_manifest",
                    "source_run_manifest",
                }
            }
            | {
                "scaler": {
                    "file": "scaler.json",
                    "sha256": sha256_file(scaler_path),
                    "bytes": scaler_path.stat().st_size,
                }
            },
        }
    )
    monkeypatch.setattr(
        "facdigger.data.inference_snapshots.load_model_release",
        lambda _: release,
    )
    target = sessions(50)[-1]

    snapshot, manifest = build_inference_snapshot(
        InferenceSnapshotConfig.model_validate(
            {
                "sources": {"bars": bars_path, "universe": universe_path},
                "output_root": tmp_path / "inference",
            }
        ),
        release_dir,
        asof_date=target,
    )

    assert snapshot.parent.name == target.isoformat()
    assert manifest["config"]["asof_date"] == target.isoformat()
    index = pl.read_parquet(snapshot / "inference_index.parquet")
    delivery = pl.read_parquet(snapshot / "delivery_universe.parquet")
    assert index["asof_date"].unique().to_list() == [target]
    assert delivery["asof_date"].unique().to_list() == [target]
    assert pl.read_parquet(snapshot / "features.parquet").height < raw.height
