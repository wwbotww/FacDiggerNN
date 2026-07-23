from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from facdigger.training.common import (
    apply_source_readiness_gate,
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
