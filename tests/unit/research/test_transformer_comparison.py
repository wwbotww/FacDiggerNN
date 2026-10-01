from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from facdigger.data.contracts import DataContractError
from facdigger.experiments.manifest import sha256_json
from facdigger.research.transformer_config import (
    TransformerComparisonConfig,
    load_transformer_comparison_config,
)
from facdigger.research.transformer_runner import (
    _load_resource_admission,
    _paired_result,
    _resume_run,
    _same_pretraining_encoder_protocol,
)
from facdigger.training.finance_pretrain_config import (
    load_finance_pretraining_config,
)
from facdigger.training.finance_transformer_config import (
    load_finance_transformer_config,
)


def _predictions(*, stronger: bool, year: int = 2023) -> pl.DataFrame:
    rows = []
    start = date(year, 1, 3)
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
        _predictions(stronger=False, year=2021 + fold_index).write_parquet(
            scratch / "predictions.parquet"
        )
        _predictions(stronger=True, year=2021 + fold_index).write_parquet(
            pretrained / "predictions.parquet"
        )
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


def _paired_fixture(tmp_path, fold_pairs):
    repository = Path(__file__).resolve().parents[3]
    config = load_transformer_comparison_config(
        repository / "configs/research/finance_transformer_streamlined.yaml"
    )
    scores_by_ic = {0.2: [0, 3, 2, 1], 0.8: [0, 1, 3, 2], 1.0: [0, 1, 2, 3]}
    plans, stages = [], []
    for index, pairs in enumerate(fold_pairs):
        fold_id = f"wf{index + 1}"
        plans.append({"fold_id": fold_id, "dataset_id": f"dataset-{fold_id}"})
        for method_index, method in enumerate(["scratch", "finance_pretrained"]):
            root = tmp_path / fold_id / method
            root.mkdir(parents=True)
            rows = [
                {
                    "security_id": str(security_index),
                    "asof_date": date(2021 + index, 1, 4) + timedelta(days=day),
                    "target": float(security_index),
                    "score_raw": float(scores_by_ic[pair[method_index]][security_index]),
                }
                for day, pair in enumerate(pairs)
                for security_index in range(4)
            ]
            pl.DataFrame(rows).write_parquet(root / "predictions.parquet")
            stages.append({"fold_id": fold_id, "stage": method, "run_dir": str(root)})
    return plans, {"stages": stages}, config


def test_comparison_primary_estimate_and_gate_use_the_same_date_weights(tmp_path) -> None:
    plans, matrix, config = _paired_fixture(
        tmp_path, [[(0.8, 1.0)] * 2, [(0.8, 1.0)] * 2, [(1.0, 0.8)] * 8]
    )
    config.decisions.minimum_paired_mean_rank_ic_delta = 0.002

    result = _paired_result(plans, matrix, config)

    assert result["paired_mean_rank_ic_delta"] == pytest.approx(-0.2 / 3)
    assert result["paired_mean_rank_ic_delta"] == result["paired_daily_inference"]["hac"]["mean"]
    assert result["fold_equal_paired_mean_rank_ic_delta"] == pytest.approx(0.2 / 3)
    assert result["acceptance"]["checks"] == {
        "paired_mean_delta": False,
        "positive_folds": True,
        "positive_worst_scratch": True,
        "positive_worst_pretrained": True,
    }
    assert result["acceptance"]["status"] == "no_go"
    assert result["inference_protocol"]["paired_mean_weighting"] == "equal_date"
    assert result["inference_protocol"]["null_mean"] == 0.002
    assert result["paired_daily_inference"]["non_overlapping"]["null_mean"] == 0.002
    assert sum(year["dates"] for year in result["years"]) == 12


def test_comparison_screening_does_not_add_a_significance_gate(tmp_path) -> None:
    pairs = [(0.2, 1.0), (1.0, 0.2), (0.8, 1.0)]
    plans, matrix, config = _paired_fixture(tmp_path, [pairs] * 3)

    result = _paired_result(plans, matrix, config)

    assert result["paired_daily_inference"]["hac"]["p_value_one_sided"] > 0.05
    assert result["acceptance"]["status"] == "go"
    assert result["acceptance"]["kind"] == "screening"
    assert result["acceptance"]["significance_is_hard_gate"] is False


def test_comparison_is_order_invariant_and_rejects_overlapping_fold_dates(tmp_path) -> None:
    plans, matrix, config = _paired_fixture(tmp_path, [[(0.8, 1.0)] * 4] * 3)
    reference = _paired_result(plans, matrix, config)
    for stage in matrix["stages"]:
        path = Path(stage["run_dir"]) / "predictions.parquet"
        pl.read_parquet(path).reverse().write_parquet(path)
    assert _paired_result(plans, matrix, config) == reference

    for stage in matrix["stages"]:
        if stage["fold_id"] == "wf2":
            path = Path(stage["run_dir"]) / "predictions.parquet"
            pl.read_parquet(path).with_columns(
                pl.col("asof_date").dt.offset_by("-1y")
            ).write_parquet(path)
    with pytest.raises(DataContractError, match="overlap"):
        _paired_result(plans, matrix, config)


def test_comparison_rejects_empty_paired_daily_observations(tmp_path) -> None:
    plans, matrix, config = _paired_fixture(tmp_path, [[(0.8, 1.0)] * 4] * 3)
    for stage in matrix["stages"]:
        if stage["fold_id"] == "wf1":
            path = Path(stage["run_dir"]) / "predictions.parquet"
            pl.read_parquet(path).with_columns(pl.lit(1.0).alias("score_raw")).write_parquet(path)
    with pytest.raises(DataContractError, match="no finite paired"):
        _paired_result(plans, matrix, config)


@pytest.mark.parametrize("status", ["paused", "complete"])
def test_date_weighting_rejects_legacy_resume_without_rewriting_reports(tmp_path, status) -> None:
    _, _, config = _paired_fixture(tmp_path / "runs", [[(0.8, 1.0)] * 4] * 3)
    legacy_config = config.model_dump(mode="json")
    legacy_config["decisions"].pop("paired_mean_weighting")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"config_hash": sha256_json(legacy_config), "status": status})
    )
    (tmp_path / "comparison.json").write_text('{"paired_mean_rank_ic_delta":0.123}')
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}

    with pytest.raises(ValueError, match="configuration does not match"):
        _resume_run(tmp_path, config)

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == before


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
        "data_protocols": {
            stage: {
                "dataset_id": "largest-fold",
                "computational_universe": universe,
                "label_support": support,
                "selection_plan": {"a_end": "2020-01-01"},
                "feature_scaler_sha256": "a" * 64,
            }
            for stage, universe, support in (
                (
                    "supervised",
                    "target_free_inference_index",
                    "complete_horizon_labels_on_original_phase_dates",
                ),
                (
                    "pretraining",
                    "pretraining_index_before_probe_selection",
                    "probe_complete_horizon_labels",
                ),
            )
        },
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
