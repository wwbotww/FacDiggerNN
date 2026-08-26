from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from facdigger.data.contracts import DataContractError
from facdigger.experiments.manifest import sha256_json
from facdigger.research.transformer_config import TransformerComparisonConfig
from facdigger.research.transformer_runner import (
    _load_resource_admission,
    _paired_result,
    _same_pretraining_encoder_protocol,
)
from facdigger.training.finance_pretrain_config import (
    load_finance_pretraining_config,
)
from facdigger.training.finance_transformer_config import (
    load_finance_transformer_config,
)


def _predictions(*, stronger: bool) -> pl.DataFrame:
    rows = []
    start = date(2023, 1, 3)
    for date_index in range(6):
        trade_date = start + timedelta(days=date_index)
        for security_index, security in enumerate(["A", "B", "C", "D"]):
            target = float(security_index)
            if stronger or date_index % 3:
                score = target
            else:
                score = -target
            rows.append(
                {
                    "security_id": security,
                    "asof_date": trade_date,
                    "target": target,
                    "score_raw": score,
                }
            )
    return pl.DataFrame(rows)


def test_streamlined_comparison_uses_three_paired_folds(tmp_path) -> None:
    plans = []
    stages = []
    for fold_index in range(3):
        fold_id = f"wf{fold_index + 1}"
        plans.append({"fold_id": fold_id, "dataset_id": f"dataset-{fold_id}"})
        scratch = tmp_path / fold_id / "scratch"
        pretrained = tmp_path / fold_id / "pretrained"
        scratch.mkdir(parents=True)
        pretrained.mkdir(parents=True)
        _predictions(stronger=False).write_parquet(scratch / "predictions.parquet")
        _predictions(stronger=True).write_parquet(pretrained / "predictions.parquet")
        stages.extend(
            [
                {
                    "fold_id": fold_id,
                    "stage": "scratch",
                    "run_dir": str(scratch),
                },
                {
                    "fold_id": fold_id,
                    "stage": "finance_pretrained",
                    "run_dir": str(pretrained),
                },
            ]
        )
    config = TransformerComparisonConfig.model_validate(
        {
            "base_dataset_config": "dataset.yaml",
            "folds": [
                {
                    "fold_id": "wf1",
                    "train_end": "2020-12-31",
                    "valid_end": "2021-12-31",
                    "test_end": "2022-12-30",
                    "embargo_sessions": 20,
                },
                {
                    "fold_id": "wf2",
                    "train_end": "2021-12-31",
                    "valid_end": "2022-12-30",
                    "test_end": "2023-12-29",
                    "embargo_sessions": 20,
                },
                {
                    "fold_id": "wf3",
                    "train_end": "2022-12-30",
                    "valid_end": "2024-12-31",
                    "test_end": "2025-12-31",
                    "embargo_sessions": 20,
                },
            ],
            "experiments": {
                "pretraining": "pretrain.yaml",
                "scratch": "scratch.yaml",
                "pretrained": "pretrained.yaml",
            },
        }
    )

    result = _paired_result(plans, {"stages": stages}, config)

    assert result["matrix"] == {
        "folds": 3,
        "seed": 42,
        "pretraining_runs": 3,
        "supervised_cells": 6,
    }
    assert result["positive_fold_count"] == 3
    assert result["paired_mean_rank_ic_delta"] > 0.001
    assert result["acceptance"]["status"] == "go"


def test_streamlined_comparison_requires_matching_rtx_admission(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[3]
    supervised = load_finance_transformer_config(
        repository / "configs/experiments/finance_patch_transformer_scratch.yaml"
    )
    pretraining = load_finance_pretraining_config(
        repository / "configs/experiments/finance_patch_pretrain.yaml"
    )
    report_path = tmp_path / "admission.json"
    report = {
        "benchmark_optimizer_updates": 100,
        "dataset_id": "largest-fold",
        "supervised_config_hash": sha256_json(supervised.model_dump(mode="json")),
        "pretraining_config_hash": sha256_json(pretraining.model_dump(mode="json")),
        "matrix_projection": {"pretraining_runs": 3, "supervised_cells": 6},
        "admission": {
            "cuda_fp16_verified": True,
            "within_memory_budget": True,
            "within_fourteen_days": True,
            "admitted": True,
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    config = TransformerComparisonConfig.model_validate(
        {
            "base_dataset_config": "dataset.yaml",
            "admission_report": report_path,
            "folds": [
                {
                    "fold_id": "wf1",
                    "train_end": "2020-12-31",
                    "valid_end": "2021-12-31",
                    "test_end": "2022-12-30",
                },
                {
                    "fold_id": "wf2",
                    "train_end": "2021-12-31",
                    "valid_end": "2022-12-30",
                    "test_end": "2023-12-29",
                },
                {
                    "fold_id": "wf3",
                    "train_end": "2022-12-30",
                    "valid_end": "2024-12-31",
                    "test_end": "2025-12-31",
                },
            ],
            "experiments": {
                "pretraining": "pretrain.yaml",
                "scratch": "scratch.yaml",
                "pretrained": "pretrained.yaml",
            },
        }
    )

    loaded = _load_resource_admission(
        config, supervised=supervised, pretraining=pretraining
    )
    assert loaded["dataset_id"] == "largest-fold"
    _same_pretraining_encoder_protocol(supervised, pretraining)

    mismatched = pretraining.model_copy(
        update={
            "model": pretraining.model.model_copy(update={"local_d_model": 32})
        }
    )
    with pytest.raises(DataContractError, match="encoder architecture"):
        _same_pretraining_encoder_protocol(supervised, mismatched)

    report["admission"]["within_memory_budget"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(DataContractError, match="within_memory_budget"):
        _load_resource_admission(
            config, supervised=supervised, pretraining=pretraining
        )
