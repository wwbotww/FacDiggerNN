from __future__ import annotations

import json
from datetime import date

import polars as pl
import pytest

from facdigger.data.contracts import DataContractError
from facdigger.evaluation.compare import compare_runs


def write_run(path, model_id: str, dataset_id: str = "dataset-1") -> None:
    path.mkdir()
    metrics = {
        "run_id": f"run-{model_id}",
        "model_id": model_id,
        "dataset_id": dataset_id,
        "evaluation_split": "valid",
        "coverage": {"coverage": 1.0},
        "metrics": {
            "raw": {
                "ic": {"mean": 0.1},
                "rank_ic": {"mean": 0.2, "ir": 1.5},
                "portfolio": {
                    "gross_q_high_minus_low": 0.01,
                    "mean_turnover": 0.3,
                },
            },
            "neutralized": None,
            "cross_section": {"research_ready": True},
        },
    }
    (path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    pl.DataFrame(
        {
            "security_id": ["sec-a"],
            "asof_date": [date(2025, 1, 2)],
            "target": [0.01],
        }
    ).write_parquet(path / "predictions.parquet")


def test_compare_requires_identical_dataset_and_samples(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_run(first, "a")
    write_run(second, "b")
    output, comparison = compare_runs([first, second], tmp_path / "comparison")
    assert comparison["sample_rows"] == 1
    assert [run["model_id"] for run in comparison["runs"]] == ["a", "b"]
    assert (output / "comparison.json").is_file()
    assert (output / "comparison.html").is_file()

    third = tmp_path / "third"
    write_run(third, "c", dataset_id="other")
    with pytest.raises(DataContractError, match="same dataset_id"):
        compare_runs([first, third], tmp_path / "invalid")
