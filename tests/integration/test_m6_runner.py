from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from facdigger.data.contracts import DataContractError
from facdigger.research.config import M6ResearchConfig
from facdigger.research.folds import final_refit_protocol
from facdigger.research.runner import run_m6_research


def _config(tmp_path) -> M6ResearchConfig:
    dataset_path = tmp_path / "dataset.yaml"
    dataset_path.write_text(
        """sources:
  bars: synthetic-bars.parquet
  universe: synthetic-universe.parquet
split:
  train_end: 2019-01-01
  valid_end: 2019-06-01
  test_end: 2019-12-01
""",
        encoding="utf-8",
    )
    model_paths = {}
    for model in ["e0", "e1", "e2", "e3"]:
        path = tmp_path / f"{model}.yaml"
        path.write_text("{}\n", encoding="utf-8")
        model_paths[model] = path
    return M6ResearchConfig.model_validate(
        {
            "research_id": "m6-runner-test",
            "base_dataset_config": dataset_path,
            "output_root": tmp_path / "research",
            "snapshot_output_root": tmp_path / "snapshots",
            "models": model_paths,
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
            "decisions": {
                "require_source_research_ready": True,
                "minimum_daily_observations_per_fold": 8,
                "minimum_non_overlapping_observations_per_fold": 4,
            },
        }
    )


def _fold_builder(config: M6ResearchConfig) -> list[dict]:
    return [
        {
            "fold_id": fold.fold_id,
            "split": fold.model_dump(mode="json", exclude={"fold_id"}),
            "dataset_id": f"dataset-{fold.fold_id}",
            "dataset_path": f"/synthetic/{fold.fold_id}",
            "dataset_manifest_sha256": f"hash-{fold.fold_id}",
            "protocol_hash": f"protocol-{fold.fold_id}",
        }
        for fold in config.folds
    ]


def _final_refit_builder_calls():
    calls = []

    def builder(config: M6ResearchConfig, final_fold: dict) -> dict:
        calls.append(final_fold)
        protocol = final_refit_protocol(config)
        fold_index = len(config.folds) - 1
        return {
            "fold_id": final_fold["fold_id"],
            "purpose": protocol["purpose"],
            "split": protocol["split"],
            "training_data_end": protocol["training_data_end"],
            "outer_validation_is_empty": True,
            "dataset_id": f"dataset-refit-{fold_index}",
            "dataset_path": f"/synthetic/refit-{fold_index}",
            "dataset_manifest_sha256": f"refit-hash-{fold_index}",
            "protocol_hash": protocol["protocol_hash"],
            "source_validation_dataset_id": final_fold["dataset_id"],
            "split_counts": {"train": 100, "test": 20},
        }

    return calls, builder


def _executor_calls(*, source_ready: bool = True):
    calls = []

    def executor(**kwargs):
        calls.append(kwargs)
        run_dir = kwargs["output_root"] / "fake-run"
        run_dir.mkdir(parents=True)
        fold_index = int(kwargs["dataset_path"].name.split("-")[-1])
        base = {"e0": 0.005, "e1": 0.010, "e2": 0.015, "e3": 0.020}[kwargs["model_key"]]
        start = (
            date(2020 + fold_index, 7, 1)
            if kwargs["dataset_path"].name.startswith("refit-")
            else date(2020 + fold_index, 2, 1)
        )
        dates = [start + timedelta(days=index) for index in range(8)]
        factors = [0.7, 1.1, 0.9, 1.3, 0.8, 1.2, 0.95, 1.05]
        daily_values = [base * factor for factor in factors]
        daily_ic = [
            {"asof_date": day.isoformat(), "n": 2, "ic": value, "rank_ic": value}
            for day, value in zip(dates, daily_values, strict=True)
        ]
        daily_portfolio = [
            {
                "asof_date": day.isoformat(),
                "n": 2,
                "groups": 5,
                "gross_q_high_minus_low": value,
                "turnover": 0.2,
                "net_20bps": value - 0.0004,
            }
            for day, value in zip(dates, daily_values, strict=True)
        ]
        score = {
            "ic": {"mean": base},
            "rank_ic": {"mean": base, "ir": 1.0},
            "portfolio": {
                "gross_q_high_minus_low": base,
                "net_20bps": base - 0.0004,
                "mean_turnover": 0.2,
            },
            "daily_ic": daily_ic,
            "daily_portfolio": daily_portfolio,
        }
        metrics = {
            "run_id": f"{kwargs['model_key']}-{kwargs['seed']}",
            "model_id": kwargs["model_key"],
            "dataset_id": f"dataset-{kwargs['dataset_path'].name}",
            "evaluation_split": kwargs["evaluation_split"],
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
            [
                {"security_id": security, "asof_date": day, "target": target}
                for day in dates
                for security, target in [("a", 0.1), ("b", -0.1)]
            ]
        ).write_parquet(run_dir / "predictions.parquet")
        return run_dir

    return calls, executor


def test_m6_requires_separate_explicit_holdout_unlock_and_freezes_hashes(tmp_path) -> None:
    config = _config(tmp_path)
    calls, executor = _executor_calls()
    refit_calls, refit_builder = _final_refit_builder_calls()
    with pytest.raises(ValueError, match="requires --resume-run"):
        run_m6_research(
            config,
            repository_root=tmp_path,
            unlock_final_holdout=True,
            snapshot_builder=_fold_builder,
            cell_executor=executor,
        )
    assert calls == []

    run_dir, validation_manifest = run_m6_research(
        config,
        repository_root=tmp_path,
        snapshot_builder=_fold_builder,
        final_refit_snapshot_builder=refit_builder,
        cell_executor=executor,
    )
    assert validation_manifest["status"] == "validation_complete"
    assert validation_manifest["holdout_unlocked"] is False
    assert len(calls) == 36
    freeze = json.loads((run_dir / "freeze.json").read_text())
    assert freeze["schema_version"] == 3
    assert freeze["validation_decision_status"] == "go"
    assert freeze["holdout_eligible"] is True
    assert freeze["final_refit_protocol"]["training_data_end"] == "2022-06-01"
    assert freeze["holdout_has_been_read"] is False
    assert not (run_dir / "holdout").exists()

    run_dir, final_manifest = run_m6_research(
        config,
        repository_root=tmp_path,
        resume_run=run_dir,
        unlock_final_holdout=True,
        snapshot_builder=_fold_builder,
        final_refit_snapshot_builder=refit_builder,
        cell_executor=executor,
    )
    assert final_manifest["status"] == "complete"
    assert final_manifest["holdout_unlocked"] is True
    assert len(calls) == 48
    assert len(refit_calls) == 1
    assert all(call["unlock_test"] for call in calls[-12:])
    assert all(call["evaluation_split"] == "test" for call in calls[-12:])
    assert all(call["dataset_path"].name == "refit-2" for call in calls[-12:])
    freeze = json.loads((run_dir / "freeze.json").read_text())
    assert freeze["holdout_has_been_read"] is True
    assert (run_dir / "validation" / "research.json").is_file()
    assert (run_dir / "holdout" / "research.json").is_file()
    refit = json.loads((run_dir / "final_refit.json").read_text())
    assert refit["training_data_end"] == "2022-06-01"
    matrix = json.loads((run_dir / "matrix.json").read_text())
    assert matrix["schema_version"] == 2
    assert len(matrix["final_refit"]) == 12
    holdout = json.loads((run_dir / "holdout" / "research.json").read_text())
    assert holdout["final_refit"]["dataset_id"] == "dataset-refit-2"


def test_m6_blocks_holdout_when_frozen_validation_decision_is_no_go(tmp_path) -> None:
    config = _config(tmp_path)
    calls, executor = _executor_calls(source_ready=False)
    refit_calls, refit_builder = _final_refit_builder_calls()
    run_dir, validation_manifest = run_m6_research(
        config,
        repository_root=tmp_path,
        snapshot_builder=_fold_builder,
        cell_executor=executor,
    )
    assert validation_manifest["status"] == "validation_complete"
    assert len(calls) == 36
    freeze = json.loads((run_dir / "freeze.json").read_text())
    assert freeze["validation_decision_status"] == "no_go"
    assert freeze["holdout_eligible"] is False

    with pytest.raises(ValueError, match="does not permit final holdout"):
        run_m6_research(
            config,
            repository_root=tmp_path,
            resume_run=run_dir,
            unlock_final_holdout=True,
            snapshot_builder=_fold_builder,
            final_refit_snapshot_builder=refit_builder,
            cell_executor=executor,
        )

    assert len(calls) == 36
    assert refit_calls == []
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "validation_complete"
    assert not (run_dir / "holdout").exists()


def test_m6_rejects_legacy_freeze_without_mutating_completed_validation(tmp_path) -> None:
    config = _config(tmp_path)
    calls, executor = _executor_calls()
    refit_calls, refit_builder = _final_refit_builder_calls()
    run_dir, _ = run_m6_research(
        config,
        repository_root=tmp_path,
        snapshot_builder=_fold_builder,
        cell_executor=executor,
    )
    freeze_path = run_dir / "freeze.json"
    freeze = json.loads(freeze_path.read_text())
    freeze["schema_version"] = 2
    freeze.pop("final_refit_protocol")
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")

    with pytest.raises(ValueError, match="predates the final-refit protocol"):
        run_m6_research(
            config,
            repository_root=tmp_path,
            resume_run=run_dir,
            unlock_final_holdout=True,
            snapshot_builder=_fold_builder,
            final_refit_snapshot_builder=refit_builder,
            cell_executor=executor,
        )

    assert len(calls) == 36
    assert refit_calls == []
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "validation_complete"


def test_m6_resume_skips_completed_cells_after_interruption(tmp_path) -> None:
    config = _config(tmp_path)
    first_calls, successful_executor = _executor_calls()

    def interrupted_executor(**kwargs):
        if len(first_calls) == 4:
            first_calls.append(kwargs)
            raise RuntimeError("synthetic interruption")
        return successful_executor(**kwargs)

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_m6_research(
            config,
            repository_root=tmp_path,
            snapshot_builder=_fold_builder,
            cell_executor=interrupted_executor,
        )
    assert len(first_calls) == 5
    run_dir = next(config.output_root.iterdir())
    interrupted_matrix = json.loads((run_dir / "matrix.json").read_text())
    assert sum(cell["status"] == "complete" for cell in interrupted_matrix["validation"]) == 4

    resumed_calls, resumed_executor = _executor_calls()
    _, manifest = run_m6_research(
        config,
        repository_root=tmp_path,
        resume_run=run_dir,
        snapshot_builder=_fold_builder,
        cell_executor=resumed_executor,
    )
    assert manifest["status"] == "validation_complete"
    assert len(resumed_calls) == 32


def test_m6_resume_skips_completed_final_refit_cells(tmp_path) -> None:
    config = _config(tmp_path)
    calls, successful_executor = _executor_calls()
    _, refit_builder = _final_refit_builder_calls()
    run_dir, _ = run_m6_research(
        config,
        repository_root=tmp_path,
        snapshot_builder=_fold_builder,
        cell_executor=successful_executor,
    )

    def interrupted_executor(**kwargs):
        completed_test_calls = sum(
            call["evaluation_split"] == "test" for call in calls
        )
        if kwargs["evaluation_split"] == "test" and completed_test_calls == 4:
            calls.append(kwargs)
            raise RuntimeError("synthetic final refit interruption")
        return successful_executor(**kwargs)

    with pytest.raises(RuntimeError, match="final refit interruption"):
        run_m6_research(
            config,
            repository_root=tmp_path,
            resume_run=run_dir,
            unlock_final_holdout=True,
            snapshot_builder=_fold_builder,
            final_refit_snapshot_builder=refit_builder,
            cell_executor=interrupted_executor,
        )

    matrix = json.loads((run_dir / "matrix.json").read_text())
    assert sum(cell["status"] == "complete" for cell in matrix["final_refit"]) == 4
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "holdout_failed"
    interrupted_freeze = json.loads((run_dir / "freeze.json").read_text())
    assert interrupted_freeze["holdout_has_been_read"] is True
    assert "holdout_read_started_at" in interrupted_freeze
    assert interrupted_freeze["final_refit_dataset_id"] == "dataset-refit-2"

    completed_run = Path(
        next(cell for cell in matrix["final_refit"] if cell["status"] == "complete")[
            "run_dir"
        ]
    )
    metrics_path = completed_run / "metrics.json"
    original_metrics = metrics_path.read_bytes()
    metrics_path.write_text("{}\n", encoding="utf-8")
    tamper_calls, tamper_executor = _executor_calls()
    with pytest.raises(DataContractError, match="metrics changed"):
        run_m6_research(
            config,
            repository_root=tmp_path,
            resume_run=run_dir,
            unlock_final_holdout=True,
            snapshot_builder=_fold_builder,
            final_refit_snapshot_builder=refit_builder,
            cell_executor=tamper_executor,
        )
    assert tamper_calls == []
    metrics_path.write_bytes(original_metrics)

    resumed_calls, resumed_executor = _executor_calls()
    _, completed = run_m6_research(
        config,
        repository_root=tmp_path,
        resume_run=run_dir,
        unlock_final_holdout=True,
        snapshot_builder=_fold_builder,
        final_refit_snapshot_builder=refit_builder,
        cell_executor=resumed_executor,
    )

    assert completed["status"] == "complete"
    assert len(resumed_calls) == 8
