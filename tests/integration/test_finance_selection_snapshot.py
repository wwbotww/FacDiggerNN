from __future__ import annotations

import json
from datetime import date

import polars as pl
import pytest
from pydantic import ValidationError

from facdigger.data.config import (
    FINANCE_TRANSFORMER_CHANNELS,
    MARKET_CONTEXT_CHANNELS,
    DatasetBuildConfig,
    semantic_dataset_config,
)
from facdigger.data.snapshots import build_dataset_snapshot, sha256_file
from facdigger.datasets.splits import build_finance_selection_indices
from facdigger.experiments.manifest import sha256_json
from tests.integration.test_dataset_pipeline import sessions, synthetic_frames


def finance_config(tmp_path, *, selection=True):
    bars, universe = synthetic_frames(180)
    bars.write_parquet(tmp_path / "bars.parquet")
    universe.write_parquet(tmp_path / "universe.parquet")
    calendar = sessions(180)
    return DatasetBuildConfig.model_validate(
        {
            "sources": {
                "bars": tmp_path / "bars.parquet",
                "universe": tmp_path / "universe.parquet",
            },
            "output_root": tmp_path / "snapshots",
            "features": {
                "name": "finance_transformer",
                "context_length": 20,
                "channels": FINANCE_TRANSFORMER_CHANNELS,
                "market_channels": MARKET_CONTEXT_CHANNELS,
            },
            "label": {"horizon": 5, "auxiliary_horizons": [1, 20]},
            "split": {
                "train_end": calendar[115],
                "valid_end": calendar[145],
                "test_end": calendar[175],
                "embargo_sessions": 2,
            },
            **(
                {"finance_selection": {"probe_fit_dates": 5, "probe_selection_dates": 4}}
                if selection
                else {}
            ),
        }
    )


def test_finance_snapshot_freezes_scaler_and_ssl_before_probe_selection(tmp_path):
    config = finance_config(tmp_path)
    snapshot, manifest = build_dataset_snapshot(config)
    scaler = json.loads((snapshot / "scaler.json").read_text())
    sample = pl.read_parquet(snapshot / "sample_index.parquet")
    market = pl.read_parquet(snapshot / "market_features.parquet")
    _, _, expected = build_finance_selection_indices(
        sample,
        market["trade_date"].to_list(),
        config.finance_selection,
    )
    assert scaler["local"]["fit_end"] == scaler["market"]["fit_end"] == expected["a_end"]
    artifact = manifest["artifacts"].get("finance_selection_plan")
    assert artifact == "finance_selection_plan.json"
    plan = json.loads((snapshot / artifact).read_text())
    assert plan == expected
    pretraining = pl.read_parquet(snapshot / "pretraining_index.parquet")
    assert pretraining["future_end"].max() <= date.fromisoformat(plan["a_end"])
    assert pretraining["feature_end"].max() < date.fromisoformat(plan["p_start"])
    assert manifest["config"]["finance_selection"] == config.finance_selection.model_dump(
        mode="json"
    )

    # A future-only perturbation must not change a fitted A transformation or
    # any SSL input, even though the old full-Train fit responds to these values.
    bars = pl.read_parquet(config.sources.bars)
    future = pl.col("trade_date") >= date.fromisoformat(plan["p_start"])
    bars = bars.with_columns(
        [
            pl.when(future).then(pl.col(column) * 3).otherwise(pl.col(column)).alias(column)
            for column in ["open", "high", "low", "close", "dollar_volume"]
        ]
    )
    bars.write_parquet(config.sources.bars)
    modified, changed_manifest = build_dataset_snapshot(config)
    assert changed_manifest["dataset_id"] != manifest["dataset_id"]
    assert json.loads((modified / "scaler.json").read_text()) == scaler
    assert json.loads((modified / artifact).read_text()) == plan
    cutoff = date.fromisoformat(plan["a_end"])
    before = pl.read_parquet(snapshot / "features.parquet").filter(pl.col("trade_date") <= cutoff)
    after = pl.read_parquet(modified / "features.parquet").filter(pl.col("trade_date") <= cutoff)
    assert before.equals(after)
    before_market = market.filter(pl.col("trade_date") <= cutoff)
    after_market = pl.read_parquet(modified / "market_features.parquet").filter(
        pl.col("trade_date") <= cutoff
    )
    assert before_market.equals(after_market)


def test_optional_plan_keeps_old_finance_identity_and_scaler(tmp_path):
    config = finance_config(tmp_path, selection=False)
    snapshot, manifest = build_dataset_snapshot(config)
    legacy_config = config.model_dump(
        mode="json", exclude={"sources", "output_root", "finance_selection"}
    )
    expected = sha256_json(
        {
            "schema_version": 5,
            "config": legacy_config,
            "input_file_hashes": {
                name: sha256_file(path) if path else None
                for name, path in config.sources.__dict__.items()
            },
        }
    )
    assert manifest["dataset_id"] == expected
    assert semantic_dataset_config(config) == legacy_config
    assert config.model_dump(mode="json") == {
        **legacy_config,
        "sources": config.sources.model_dump(mode="json"),
        "output_root": str(config.output_root),
    }
    assert config.model_dump(mode="json")["sources"]["corporate_actions"] is None
    assert "finance_selection" not in manifest["config"]
    assert "finance_selection_plan" not in manifest["artifacts"]
    assert (
        json.loads((snapshot / "scaler.json").read_text())["local"]["fit_end"]
        == config.split.train_end.isoformat()
    )


def test_finance_selection_cannot_be_applied_to_non_finance_data(tmp_path):
    config = finance_config(tmp_path).model_dump()
    config["features"] = {"context_length": 20}
    with pytest.raises(ValidationError, match="Finance"):
        DatasetBuildConfig.model_validate(config)
