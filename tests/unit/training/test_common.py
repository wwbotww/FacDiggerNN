from __future__ import annotations

from facdigger.training.common import apply_source_readiness_gate


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
