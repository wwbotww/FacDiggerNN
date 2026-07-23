from __future__ import annotations

import json
from datetime import date, timedelta

import polars as pl
import pytest

from facdigger.data.contracts import DataContractError
from facdigger.research.aggregate import aggregate_research_runs, write_research_report
from facdigger.research.config import M6ResearchConfig


def _config(tmp_path) -> M6ResearchConfig:
    return M6ResearchConfig.model_validate(
        {
            "research_id": "synthetic-m6",
            "base_dataset_config": tmp_path / "dataset.yaml",
            "output_root": tmp_path / "research",
            "models": {
                "e0": tmp_path / "e0.yaml",
                "e1": tmp_path / "e1.yaml",
                "e2": tmp_path / "e2.yaml",
                "e3": tmp_path / "e3.yaml",
            },
            "seeds": [1, 2, 3],
            "folds": [
                {
                    "fold_id": f"fold-{index}",
                    "train_end": date(2020 + index, 1, 1),
                    "valid_end": date(2020 + index, 6, 1),
                    "test_end": date(2020 + index, 12, 1),
                }
                for index in range(3)
            ],
            "hac_lags": 1,
            "non_overlapping_stride": 2,
        }
    )


def _write_cell(
    root, *, fold_index: int, seed: int, model: str, value: float, source_ready: bool
) -> dict:
    run_dir = root / f"fold-{fold_index}" / model / str(seed)
    run_dir.mkdir(parents=True)
    dates = [date(2020 + fold_index, 2, 1) + timedelta(days=index) for index in range(6)]
    daily_ic = [
        {"asof_date": day.isoformat(), "n": 30, "ic": value, "rank_ic": value}
        for day in dates
    ]
    daily_portfolio = [
        {
            "asof_date": day.isoformat(),
            "n": 30,
            "groups": 5,
            "gross_q_high_minus_low": value,
            "turnover": 0.2,
            "net_20bps": value - 0.0004,
        }
        for day in dates
    ]
    score = {
        "ic": {"mean": value},
        "rank_ic": {"mean": value, "ir": 1.0},
        "portfolio": {
            "gross_q_high_minus_low": value,
            "net_20bps": value - 0.0004,
            "mean_turnover": 0.2,
        },
        "daily_ic": daily_ic,
        "daily_portfolio": daily_portfolio,
    }
    dataset_id = f"dataset-{fold_index}"
    metrics = {
        "run_id": f"{fold_index}-{seed}-{model}",
        "model_id": model,
        "dataset_id": dataset_id,
        "evaluation_split": "valid",
        "coverage": {"coverage": 1.0},
        "metrics": {
            "raw": score,
            "neutralized": score,
            "cross_section": {
                "research_ready": source_ready,
                "source_research_ready": source_ready,
            },
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    pl.DataFrame(
        {
            "security_id": ["a", "b"],
            "asof_date": [dates[0], dates[0]],
            "target": [0.1, -0.1],
        }
    ).write_parquet(run_dir / "predictions.parquet")
    return {
        "fold_id": f"fold-{fold_index}",
        "seed": seed,
        "model_key": model,
        "dataset_id": dataset_id,
        "evaluation_split": "valid",
        "run_dir": str(run_dir),
    }


def _matrix(tmp_path, source_ready: bool = True) -> list[dict]:
    values = {"e0": 0.005, "e1": 0.010, "e2": 0.015, "e3": 0.020}
    return [
        _write_cell(
            tmp_path / "runs",
            fold_index=fold,
            seed=seed,
            model=model,
            value=value,
            source_ready=source_ready,
        )
        for fold in range(3)
        for seed in [1, 2, 3]
        for model, value in values.items()
    ]


def test_research_aggregation_answers_all_incremental_questions(tmp_path) -> None:
    config = _config(tmp_path)
    result = aggregate_research_runs(_matrix(tmp_path), config, evaluation_split="valid")
    write_research_report(result, tmp_path / "report")

    assert result["cell_count"] == 36
    assert result["decisions"]["architecture_e1_vs_e0"]["status"] == "go"
    assert result["decisions"]["external_transfer_e2_vs_e1"]["status"] == "go"
    assert result["decisions"]["financial_pretraining_e3_vs_e2"]["status"] == "go"
    assert result["decisions"]["overall_e3"]["status"] == "go"
    assert (tmp_path / "report" / "research.json").is_file()
    assert (tmp_path / "report" / "research.html").is_file()


def test_research_aggregation_fails_closed_on_missing_cell_or_source_block(tmp_path) -> None:
    config = _config(tmp_path)
    cells = _matrix(tmp_path, source_ready=False)
    with pytest.raises(DataContractError, match="incomplete"):
        aggregate_research_runs(cells[:-1], config, evaluation_split="valid")

    result = aggregate_research_runs(cells, config, evaluation_split="valid")
    assert result["source_readiness"]["explicitly_blocked"] is True
    assert result["decisions"]["overall_e3"]["status"] == "no_go"
