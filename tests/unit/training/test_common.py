from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from facdigger.training.common import (
    apply_source_readiness_gate,
    load_required_snapshot_features,
    load_training_snapshot,
    split_supervised_training_index,
)


def test_source_provenance_can_block_statistically_evaluable_results() -> None:
    metrics = {
        "cross_section": {
            "research_ready": True,
            "research_ready_rule": "statistical rule",
        }
    }

    result = apply_source_readiness_gate(
        metrics,
        {"research_ready": False, "warnings": ["survivorship biased"]},
    )

    assert result["cross_section"]["statistical_ready"] is True
    assert result["cross_section"]["source_research_ready"] is False
    assert result["cross_section"]["research_ready"] is False


def test_missing_source_gate_does_not_block_existing_standard_parquet() -> None:
    metrics = {"cross_section": {"research_ready": True}}

    result = apply_source_readiness_gate(metrics, {"research_ready": None})

    assert result["cross_section"]["research_ready"] is True


def test_supervised_selection_is_inside_train_and_purges_label_overlap() -> None:
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(10)]
    rows = []
    for security_id in ["a", "b"]:
        for asof_date in dates:
            rows.append(
                {
                    "sample_id": f"{security_id}|{asof_date}",
                    "security_id": security_id,
                    "asof_date": asof_date,
                    "label_end": asof_date + timedelta(days=2),
                    "split": "train",
                }
            )
    rows.append(
        {
            "sample_id": "a|valid",
            "security_id": "a",
            "asof_date": date(2021, 1, 1),
            "label_end": date(2021, 1, 3),
            "split": "valid",
        }
    )

    protocol, audit = split_supervised_training_index(
        pl.DataFrame(rows),
        selection_fraction=0.2,
    )

    assert set(protocol["split"].unique()) == {
        "train_fit",
        "inner_selection",
        "valid",
    }
    assert audit["selection_rows"] == 4
    assert audit["purged_rows"] == 4
    assert audit["fit_max_label_end"] < audit["selection_min_asof_date"]
    assert audit["outer_validation_rows_used_for_checkpoint_selection"] == 0


def test_training_snapshot_can_skip_features_and_never_loads_inference_index(
    tmp_path: Path,
) -> None:
    manifest = {
        "schema_version": 3,
        "config": {"features": {"channels": ["x"]}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pl.DataFrame({"sample_id": ["sample"]}).write_parquet(
        tmp_path / "sample_index.parquet"
    )
    pl.DataFrame({"sample_id": ["sample"]}).write_parquet(
        tmp_path / "sample_metadata.parquet"
    )
    pl.DataFrame({"sample_id": ["future"]}).write_parquet(
        tmp_path / "inference_index.parquet"
    )

    _, frames = load_training_snapshot(tmp_path, include_features=False)

    assert set(frames) == {"sample_index", "sample_metadata"}


def test_required_feature_loader_projects_channels_and_window_ranges(
    tmp_path: Path,
) -> None:
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(6)]
    features = pl.DataFrame(
        {
            "security_id": ["A"] * 6 + ["B"] * 6,
            "trade_date": dates * 2,
            "x": [float(index) for index in range(12)],
            "observed_x": [True] * 12,
            "unused": [999.0] * 12,
        }
    )
    features.write_parquet(tmp_path / "features.parquet")
    manifest = {
        "config": {"features": {"channels": ["x"]}},
        "artifacts": {"features": "features.parquet"},
    }
    required_rows = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "feature_start": [dates[1], dates[2]],
            "asof_date": [dates[3], dates[4]],
        }
    )

    selected = load_required_snapshot_features(tmp_path, manifest, required_rows)

    assert selected.columns == ["security_id", "trade_date", "x", "observed_x"]
    assert selected["security_id"].unique().to_list() == ["A"]
    assert selected["trade_date"].to_list() == dates[1:5]
