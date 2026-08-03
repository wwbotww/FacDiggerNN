from __future__ import annotations

import json
from datetime import date, timedelta

import polars as pl
import pytest

from facdigger.research.config import M6ResearchConfig
from facdigger.research.runner import run_m6_research


def _config(tmp_path) -> M6ResearchConfig:
    model_paths = {}
    for model in ["e0", "e1", "e2", "e3"]:
        path = tmp_path / f"{model}.yaml"
        path.write_text("{}\n", encoding="utf-8")
        model_paths[model] = path
    return M6ResearchConfig.model_validate(
        {
            "research_id": "m6-runner-test",
            "base_dataset_config": tmp_path / "dataset.yaml",
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


def _executor_calls(*, source_ready: bool = True):
    calls = []

    def executor(**kwargs):
        calls.append(kwargs)
        run_dir = kwargs["output_root"] / "fake-run"
        run_dir.mkdir(parents=True)
        fold_index = int(kwargs["dataset_path"].name.split("-")[-1])
        base = {"e0": 0.005, "e1": 0.010, "e2": 0.015, "e3": 0.020}[kwargs["model_key"]]
        dates = [date(2020 + fold_index, 2, 1) + timedelta(days=index) for index in range(8)]
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
        cell_executor=executor,
    )
    assert validation_manifest["status"] == "validation_complete"
    assert validation_manifest["holdout_unlocked"] is False
    assert len(calls) == 36
    freeze = json.loads((run_dir / "freeze.json").read_text())
    assert freeze["schema_version"] == 2
    assert freeze["validation_decision_status"] == "go"
    assert freeze["holdout_eligible"] is True
    assert freeze["holdout_has_been_read"] is False
    assert not (run_dir / "holdout").exists()

    run_dir, final_manifest = run_m6_research(
        config,
        repository_root=tmp_path,
        resume_run=run_dir,
        unlock_final_holdout=True,
        snapshot_builder=_fold_builder,
        cell_executor=executor,
    )
    assert final_manifest["status"] == "complete"
    assert final_manifest["holdout_unlocked"] is True
    assert len(calls) == 48
    assert all(call["unlock_test"] for call in calls[-12:])
    assert all(call["evaluation_split"] == "test" for call in calls[-12:])
    freeze = json.loads((run_dir / "freeze.json").read_text())
    assert freeze["holdout_has_been_read"] is True
    assert (run_dir / "validation" / "research.json").is_file()
    assert (run_dir / "holdout" / "research.json").is_file()


def test_m6_blocks_holdout_when_frozen_validation_decision_is_no_go(tmp_path) -> None:
    config = _config(tmp_path)
    calls, executor = _executor_calls(source_ready=False)
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
            cell_executor=executor,
        )

    assert len(calls) == 36
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "validation_complete"
    assert not (run_dir / "holdout").exists()


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
