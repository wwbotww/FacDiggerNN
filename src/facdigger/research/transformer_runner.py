"""Resumable 3-fold scratch-versus-financial-pretraining comparison."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.environment import collect_environment
from facdigger.evaluation.metrics import daily_information_coefficients
from facdigger.experiments.manifest import collect_git_state, sha256_json
from facdigger.research.statistics import panel_mean_inference
from facdigger.research.transformer_config import (
    TransformerComparisonConfig,
    validate_transformer_experiment_paths,
)
from facdigger.research.transformer_snapshots import build_transformer_snapshots
from facdigger.training.finance_pretrain import run_finance_pretraining
from facdigger.training.finance_pretrain_config import (
    FinancePretrainingExperimentConfig,
    load_finance_pretraining_config,
)
from facdigger.training.finance_transformer import run_finance_transformer
from facdigger.training.finance_transformer_config import (
    FinanceTransformerExperimentConfig,
    load_finance_transformer_config,
)
from facdigger.training.resources import (
    TrainingResourceBudget,
    effective_resource_limits,
    training_hardware,
)
from facdigger.training.run_state import verify_completed_artifacts
from facdigger.training.runtime import (
    TrainingControl,
    TrainingPaused,
    TrainingRuntimeConfig,
    atomic_write_text,
    resolve_dataset,
    run_lock,
)
from facdigger.training.runtime import (
    write_json as _write_json,
)


def _new_run(
    config: TransformerComparisonConfig,
    repository_root: Path,
    admission: dict[str, Any],
    destination: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    created_at = datetime.now(timezone.utc)
    config_payload = config.model_dump(mode="json")
    config_hash = sha256_json(config_payload)
    identity = {
        "config_hash": config_hash,
        "created_at": created_at.isoformat(),
        "nonce": uuid.uuid4().hex,
    }
    run_id = (
        f"{config.research_id}-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{sha256_json(identity)[:8]}"
    )
    run_dir = destination or config.output_root.resolve() / run_id
    run_id = run_dir.name
    run_dir.mkdir(parents=True, exist_ok=destination is not None)
    manifest = {
        "status": "building_snapshots",
        "run_id": run_id,
        "research_id": config.research_id,
        "created_at": created_at.isoformat(),
        "updated_at": created_at.isoformat(),
        "config_hash": config_hash,
        "matrix_contract": {
            "folds": 3,
            "seeds": [42],
            "supervised_methods": ["scratch", "finance_pretrained"],
            "pretraining_runs": 3,
            "supervised_cells": 6,
            "long_stages": 9,
        },
        "repository_root": str(repository_root),
        "git": collect_git_state(repository_root),
        "resource_admission": {
            "report": str(config.admission_report.resolve()),
            "report_sha256": sha256_file(config.admission_report),
            "dataset_id": admission["dataset_id"],
            "projected_days": admission["matrix_projection"]["projected_days"],
            "host_peak_rss_bytes": admission["host_peak_rss_bytes"],
            "cuda_peak_reserved_bytes": max(
                int(admission["supervised"]["cuda_peak_reserved_bytes"] or 0),
                int(admission["pretraining"]["cuda_peak_reserved_bytes"] or 0),
            ),
        },
    }
    atomic_write_text(
        run_dir / "resolved_config.yaml",
        yaml.safe_dump(config_payload, allow_unicode=True, sort_keys=True),
    )
    _write_json(run_dir / "matrix.json", {"stages": []})
    _write_json(run_dir / "manifest.json", manifest)
    return run_dir, manifest


def _resume_run(run_dir: Path, config: TransformerComparisonConfig) -> tuple[Path, dict[str, Any]]:
    root = run_dir.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Transformer comparison manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config_hash"] != sha256_json(config.model_dump(mode="json")):
        raise ValueError("Transformer comparison resume configuration does not match")
    if manifest.get("status") == "complete":
        raise ValueError("cannot resume a complete Transformer comparison")
    return root, manifest


def _same_supervised_protocol(
    scratch: FinanceTransformerExperimentConfig,
    pretrained: FinanceTransformerExperimentConfig,
) -> None:
    left = scratch.model_dump(mode="json")
    right = pretrained.model_dump(mode="json")
    for payload in (left, right):
        for key in (
            "experiment_id",
            "initialization",
            "pretrained_checkpoint",
            "output_root",
        ):
            payload.pop(key)
    if left != right:
        raise DataContractError(
            "scratch and pretrained must share data, architecture and supervised training"
        )
    if scratch.initialization != "scratch":
        raise DataContractError("scratch configuration has the wrong initialization")
    if pretrained.initialization != "finance_pretrained":
        raise DataContractError("pretrained configuration has the wrong initialization")


def _same_pretraining_encoder_protocol(
    supervised: FinanceTransformerExperimentConfig,
    pretraining: FinancePretrainingExperimentConfig,
) -> None:
    if supervised.channels != pretraining.channels or (
        supervised.market_channels != pretraining.market_channels
    ):
        raise DataContractError("pretraining and supervised channel contracts must be identical")
    if supervised.model.model_dump(mode="json") != pretraining.model.model_dump(mode="json"):
        raise DataContractError("pretraining and supervised encoder architecture must be identical")


def _load_resource_admission(
    config: TransformerComparisonConfig,
    *,
    supervised: FinanceTransformerExperimentConfig,
    pretraining: FinancePretrainingExperimentConfig,
    resource_budget: TrainingResourceBudget | None = None,
) -> dict[str, Any]:
    path = config.admission_report.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"RTX admission report is required before transformer-run: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("benchmark_optimizer_updates", 0)) < 100:
        raise DataContractError("RTX admission must measure at least 100 optimizer updates")
    expected_hashes = {
        "supervised_config_hash": sha256_json(supervised.model_dump(mode="json")),
        "pretraining_config_hash": sha256_json(pretraining.model_dump(mode="json")),
    }
    for field, expected in expected_hashes.items():
        if payload.get(field) != expected:
            raise DataContractError(f"RTX admission {field} does not match this matrix")
    projection = payload.get("matrix_projection", {})
    if projection.get("pretraining_runs") != 3 or projection.get("supervised_cells") != 6:
        raise DataContractError("RTX admission does not project the fixed nine stages")
    admission = payload.get("admission", {})
    if resource_budget is not None:
        if payload.get("resource_budget") != resource_budget.model_dump(mode="json"):
            raise DataContractError("resource admission budget does not match")
        hardware = training_hardware()
        if payload.get("hardware") != hardware:
            raise DataContractError("resource admission hardware differs; repeat benchmark")
        limits = effective_resource_limits(resource_budget, hardware)
        gpu_peak = max(
            int(payload[key]["cuda_peak_reserved_bytes"] or 0)
            for key in ("supervised", "pretraining")
        )
        if gpu_peak > limits["cuda_peak_reserved_bytes"] or (
            payload["host_peak_rss_bytes"] is None
            or payload["host_peak_rss_bytes"] > limits["host_peak_rss_bytes"]
        ):
            raise DataContractError("resource admission exceeds current allocation")
        if projection["projected_days"] > resource_budget.projected_days:
            raise DataContractError("resource admission exceeds time budget")
    elif "resource_budget" in payload:
        raise DataContractError("explicit resource admission requires its resource budget")
    required_checks = (
        "cuda_fp16_verified",
        "within_memory_budget",
        "within_time_budget" if resource_budget is not None else "within_fourteen_days",
        "admitted",
    )
    failed = [name for name in required_checks if admission.get(name) is not True]
    if failed:
        raise DataContractError(
            "RTX admission blocks transformer-run; failed checks: " + ", ".join(failed)
        )
    return payload


def _validate_admission_dataset(
    admission: dict[str, Any], fold_plans: list[dict[str, Any]]
) -> None:
    if not fold_plans:
        raise DataContractError("Transformer matrix has no fold snapshots")
    largest_fold = fold_plans[-1]
    if admission.get("dataset_id") != largest_fold["dataset_id"]:
        raise DataContractError(
            "RTX admission dataset_id must match the largest (last) fold snapshot"
        )


def _record(matrix: dict[str, Any], *, fold_id: str, stage: str) -> dict[str, Any]:
    for item in matrix["stages"]:
        if item["fold_id"] == fold_id and item["stage"] == stage:
            return item
    item = {"fold_id": fold_id, "stage": stage, "seed": 42, "status": "pending"}
    matrix["stages"].append(item)
    return item


def _validate_completed_stage(record: dict[str, Any]) -> Path:
    run_dir = Path(record["run_dir"])
    manifest = run_dir / "manifest.json"
    if not manifest.is_file() or sha256_file(manifest) != record["manifest_sha256"]:
        raise DataContractError(f"completed Transformer stage changed: {run_dir}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise DataContractError(f"completed Transformer stage is not complete: {run_dir}")
    return run_dir


def _run_stage(
    record: dict[str, Any],
    matrix: dict[str, Any],
    matrix_path: Path,
    *,
    output_root: Path,
    experiment: Any,
    plan: dict[str, Any],
    dataset_path: Path,
    repository: Path,
    control: TrainingControl,
    trainer: Any,
) -> Path:
    expected = {
        "config_hash": sha256_json(experiment.model_dump(mode="json")),
        "dataset_id": plan["dataset_id"],
        "dataset_manifest_hash": plan["dataset_manifest_sha256"],
    }
    if "run_dir" not in record:
        candidates = list(output_root.glob("*/manifest.json"))
        matches = []
        for path in candidates:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if all(payload.get(key) == value for key, value in expected.items()):
                matches.append(path.parent)
        if candidates and len(matches) != 1:
            raise DataContractError("legacy stage has ambiguous or mismatched child runs")
        record["run_dir"] = str(matches[0] if matches else output_root / "run")
        _write_json(matrix_path, matrix)
    child = Path(record["run_dir"]).resolve()
    if not child.is_relative_to(output_root.resolve()):
        raise DataContractError("bound child run is outside the stage output directory")
    path = child / "manifest.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if any(payload.get(key) != value for key, value in expected.items()):
            raise DataContractError("bound child run protocol or dataset differs")
        if record["status"] == "complete":
            _validate_completed_stage(record)
        if payload.get("status") == "complete":
            verify_completed_artifacts(child, payload)
            _complete_record(record, child)
            _write_json(matrix_path, matrix)
            return child
    elif record["status"] == "complete":
        raise DataContractError("completed stage has no manifest")
    record.update({"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})
    _write_json(matrix_path, matrix)
    try:
        completed, _ = trainer(
            experiment, dataset_path, repository_root=repository, run_dir=child, control=control
        )
    except BaseException as exc:
        record["status"] = "paused" if isinstance(exc, TrainingPaused) else "failed"
        _write_json(matrix_path, matrix)
        raise
    _complete_record(record, completed)
    _write_json(matrix_path, matrix)
    return completed


def _complete_record(record: dict[str, Any], run_dir: Path) -> None:
    record.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir.resolve()),
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "error": None,
        }
    )


def _paired_result(
    fold_plans: list[dict[str, Any]],
    matrix: dict[str, Any],
    config: TransformerComparisonConfig,
) -> dict[str, Any]:
    by_key = {(record["fold_id"], record["stage"]): record for record in matrix["stages"]}
    folds: list[dict[str, Any]] = []
    daily_delta_groups: list[list[float]] = []
    for plan in fold_plans:
        fold_id = plan["fold_id"]
        scratch_path = Path(by_key[(fold_id, "scratch")]["run_dir"])
        pretrained_path = Path(by_key[(fold_id, "finance_pretrained")]["run_dir"])
        scratch = pl.read_parquet(scratch_path / "predictions.parquet").sort(
            ["asof_date", "security_id"]
        )
        pretrained = pl.read_parquet(pretrained_path / "predictions.parquet").sort(
            ["asof_date", "security_id"]
        )
        keys = ["security_id", "asof_date", "target"]
        if not scratch.select(keys).equals(pretrained.select(keys), null_equal=True):
            raise DataContractError(f"paired prediction keys differ for {fold_id}")
        scratch_daily = daily_information_coefficients(scratch).select(
            "asof_date", pl.col("rank_ic").alias("scratch_rank_ic")
        )
        pretrained_daily = daily_information_coefficients(pretrained).select(
            "asof_date", pl.col("rank_ic").alias("pretrained_rank_ic")
        )
        paired = scratch_daily.join(
            pretrained_daily, on="asof_date", how="inner", validate="1:1"
        ).drop_nulls()
        if paired.height != scratch_daily.drop_nulls().height or paired.height != (
            pretrained_daily.drop_nulls().height
        ):
            raise DataContractError(f"paired daily Rank IC dates differ for {fold_id}")
        paired = paired.with_columns(
            (pl.col("pretrained_rank_ic") - pl.col("scratch_rank_ic")).alias("delta")
        )
        delta = paired["delta"].to_list()
        daily_delta_groups.append(delta)
        folds.append(
            {
                "fold_id": fold_id,
                "dataset_id": plan["dataset_id"],
                "dates": paired.height,
                "scratch_mean_rank_ic": float(paired["scratch_rank_ic"].mean()),
                "pretrained_mean_rank_ic": float(paired["pretrained_rank_ic"].mean()),
                "paired_mean_delta": float(paired["delta"].mean()),
            }
        )
    paired_mean = float(np.mean([fold["paired_mean_delta"] for fold in folds]))
    positive_folds = sum(fold["paired_mean_delta"] > 0 for fold in folds)
    worst_scratch = min(fold["scratch_mean_rank_ic"] for fold in folds)
    worst_pretrained = min(fold["pretrained_mean_rank_ic"] for fold in folds)
    decision = config.decisions
    checks = {
        "paired_mean_delta": paired_mean >= decision.minimum_paired_mean_rank_ic_delta,
        "positive_folds": positive_folds >= decision.minimum_positive_folds,
        "positive_worst_scratch": worst_scratch > 0,
        "positive_worst_pretrained": worst_pretrained > 0,
    }
    return {
        "research_id": config.research_id,
        "matrix": {
            "folds": 3,
            "seed": 42,
            "pretraining_runs": 3,
            "supervised_cells": 6,
        },
        "folds": folds,
        "paired_mean_rank_ic_delta": paired_mean,
        "positive_fold_count": positive_folds,
        "worst_fold": {
            "scratch_rank_ic": worst_scratch,
            "pretrained_rank_ic": worst_pretrained,
        },
        "paired_daily_inference": panel_mean_inference(
            daily_delta_groups,
            hac_lags=decision.hac_lags,
            stride=decision.non_overlapping_stride,
            offset=0,
            null_mean=decision.minimum_paired_mean_rank_ic_delta,
        ),
        "acceptance": {
            "checks": checks,
            "status": "go" if all(checks.values()) else "no_go",
        },
    }


def run_transformer_comparison(
    config: TransformerComparisonConfig,
    *,
    repository_root: str | Path,
    resume_run: str | Path | None = None,
    run_dir: str | Path | None = None,
    runtime: TrainingRuntimeConfig | None = None,
    resource_budget: TrainingResourceBudget | None = None,
) -> tuple[Path, dict[str, Any]]:
    if (
        resume_run is not None
        and run_dir is not None
        and Path(resume_run).resolve() != Path(run_dir).resolve()
    ):
        raise ValueError("resume run and run directory differ")
    root = (
        Path(resume_run or run_dir).resolve()
        if (resume_run or run_dir)
        else (config.output_root.resolve() / f"{config.research_id}-{uuid.uuid4().hex[:12]}")
    )
    with run_lock(root / ".research.lock"), TrainingControl(runtime) as control:
        existing = root / "manifest.json"
        if existing.is_file():
            manifest = json.loads(existing.read_text(encoding="utf-8"))
            if manifest.get("status") == "complete":
                if manifest["config_hash"] != sha256_json(config.model_dump(mode="json")):
                    raise DataContractError("completed research configuration differs")
                if sha256_file(root / "comparison.json") != manifest["comparison_sha256"]:
                    raise DataContractError("completed comparison changed")
                for record in json.loads((root / "matrix.json").read_text(encoding="utf-8"))[
                    "stages"
                ]:
                    child = _validate_completed_stage(record)
                    verify_completed_artifacts(
                        child, json.loads((child / "manifest.json").read_text(encoding="utf-8"))
                    )
                return root, manifest
        elif resume_run is not None:
            raise FileNotFoundError(f"Transformer comparison manifest missing: {existing}")
        return _run_transformer_comparison(
            config,
            repository_root=repository_root,
            destination=root,
            resume_run=root if existing.is_file() else None,
            control=control,
            resource_budget=resource_budget,
        )


def _run_transformer_comparison(
    config: TransformerComparisonConfig,
    *,
    repository_root: str | Path,
    destination: Path,
    resume_run: Path | None,
    control: TrainingControl,
    resource_budget: TrainingResourceBudget | None,
) -> tuple[Path, dict[str, Any]]:
    repository = Path(repository_root).resolve()
    paths = validate_transformer_experiment_paths(config)
    scratch_template = load_finance_transformer_config(paths["scratch"])
    pretrained_template = load_finance_transformer_config(paths["pretrained"])
    pretraining_template = load_finance_pretraining_config(paths["pretraining"])
    _same_supervised_protocol(scratch_template, pretrained_template)
    _same_pretraining_encoder_protocol(scratch_template, pretraining_template)
    admission = _load_resource_admission(
        config,
        supervised=scratch_template,
        pretraining=pretraining_template,
        resource_budget=resource_budget,
    )
    if resume_run is None:
        run_dir, manifest = _new_run(config, repository, admission, destination)
    else:
        run_dir, manifest = _resume_run(Path(resume_run), config)
        expected_report_hash = manifest.get("resource_admission", {}).get("report_sha256")
        if expected_report_hash != sha256_file(config.admission_report):
            raise DataContractError("RTX admission report changed after matrix creation")
    matrix_path = run_dir / "matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "running",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "stop_reason": None,
            "error": None,
        }
    )
    manifest.setdefault("attempts", []).append(
        {
            "started_at": manifest["updated_at"],
            "runtime": control.config.model_dump(mode="json"),
            "git": collect_git_state(repository),
            "environment": collect_environment(include_model_dependencies=True),
        }
    )
    _write_json(run_dir / "manifest.json", manifest)
    try:
        folds_path = run_dir / "folds.json"
        if not folds_path.is_file():
            fold_plans = build_transformer_snapshots(config, control.config)
            _validate_admission_dataset(admission, fold_plans)
            _write_json(folds_path, fold_plans)
        else:
            fold_plans = json.loads(folds_path.read_text(encoding="utf-8"))
            _validate_admission_dataset(admission, fold_plans)
        for plan in fold_plans:
            fold_id = plan["fold_id"]
            dataset_path = resolve_dataset(
                Path(plan["dataset_path"]),
                control.config,
                dataset_id=plan["dataset_id"],
                manifest_hash=plan["dataset_manifest_sha256"],
            )
            pretrain_record = _record(matrix, fold_id=fold_id, stage="pretraining")
            output_root = run_dir / "runs" / fold_id / "pretraining"
            pretrain_config = pretraining_template.model_copy(
                update={"seed": 42, "output_root": output_root}
            )
            pretrain_run = _run_stage(
                pretrain_record,
                matrix,
                matrix_path,
                output_root=output_root,
                experiment=pretrain_config,
                plan=plan,
                dataset_path=dataset_path,
                repository=repository,
                control=control,
                trainer=run_finance_pretraining,
            )
            encoder = pretrain_run / "checkpoints" / "best_encoder.pt"
            if not encoder.is_file():
                raise FileNotFoundError(f"pretraining encoder is missing: {encoder}")

            for stage, template in (
                ("scratch", scratch_template),
                ("finance_pretrained", pretrained_template),
            ):
                record = _record(matrix, fold_id=fold_id, stage=stage)
                output_root = run_dir / "runs" / fold_id / stage
                update: dict[str, Any] = {
                    "seed": 42,
                    "output_root": output_root,
                    "evaluation_split": "valid",
                    "unlock_test": False,
                }
                if stage == "finance_pretrained":
                    update["pretrained_checkpoint"] = encoder
                experiment = template.model_copy(update=update)
                _run_stage(
                    record,
                    matrix,
                    matrix_path,
                    output_root=output_root,
                    experiment=experiment,
                    plan=plan,
                    dataset_path=dataset_path,
                    repository=repository,
                    control=control,
                    trainer=run_finance_transformer,
                )
        result = _paired_result(fold_plans, matrix, config)
        _write_json(run_dir / "comparison.json", result)
        manifest.update(
            {
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "comparison_sha256": sha256_file(run_dir / "comparison.json"),
                "acceptance": result["acceptance"],
                "git": collect_git_state(repository),
            }
        )
        _write_json(run_dir / "manifest.json", manifest)
    except BaseException as exc:
        manifest.update(
            {
                "status": "paused" if isinstance(exc, TrainingPaused) else "failed",
                "stop_reason": exc.reason if isinstance(exc, TrainingPaused) else None,
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(matrix_path, matrix)
        raise
    return run_dir, manifest
