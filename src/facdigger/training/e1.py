"""End-to-end E1 random PatchTST training, prediction and evaluation."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.datasets.window import SnapshotWindowDataset
from facdigger.environment import collect_environment
from facdigger.evaluation.contracts import prediction_coverage
from facdigger.evaluation.metrics import evaluate_predictions
from facdigger.evaluation.report import write_evaluation_report
from facdigger.experiments.manifest import collect_git_state, sha256_json
from facdigger.training.common import (
    apply_source_readiness_gate,
    build_prediction_frame,
    load_source_provenance,
    load_training_snapshot,
)
from facdigger.training.e1_config import E1ExperimentConfig
from facdigger.training.e1_engine import predict_e1, train_e1


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _new_run_dir(config: E1ExperimentConfig, dataset_id: str) -> tuple[Path, str, str]:
    created_at = datetime.now(timezone.utc)
    config_payload = config.model_dump(mode="json")
    identity = {
        "config": config_payload,
        "dataset_id": dataset_id,
        "created_at": created_at.isoformat(),
        "nonce": uuid.uuid4().hex,
    }
    run_id = (
        f"{config.experiment_id}-{created_at.strftime('%Y%m%dT%H%M%SZ')}-"
        f"{sha256_json(identity)[:8]}"
    )
    run_dir = config.output_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_id, created_at.isoformat()


def _resume_run_dir(
    resume_from: Path, *, dataset_id: str, config_hash: str
) -> tuple[Path, str, str]:
    checkpoint = resume_from.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"E1 resume checkpoint not found: {checkpoint}")
    if checkpoint.parent.name != "checkpoints":
        raise ValueError("E1 resume checkpoint must be inside a run's checkpoints directory")
    run_dir = checkpoint.parent.parent
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"E1 run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") == "complete":
        raise ValueError("cannot resume an already-complete E1 run")
    if manifest.get("dataset_id") != dataset_id:
        raise ValueError("resume run dataset_id does not match")
    if manifest.get("config_hash") != config_hash:
        raise ValueError("resume run configuration does not match")
    return run_dir, str(manifest["run_id"]), str(manifest["created_at"])


def run_e1(
    config: E1ExperimentConfig,
    dataset_dir: str | Path,
    *,
    repository_root: str | Path,
    resume_from: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    dataset_path = Path(dataset_dir).resolve()
    dataset_manifest, frames = load_training_snapshot(dataset_path)
    dataset_config = dataset_manifest["config"]
    context_length = int(dataset_config["features"]["context_length"])
    dataset_channels = list(dataset_config["features"]["channels"])
    if config.channels != dataset_channels:
        raise DataContractError(
            f"E1 channels must exactly match dataset channels: {dataset_channels}"
        )
    if config.model.patch_length > context_length:
        raise DataContractError("E1 patch_length cannot exceed dataset context_length")

    datasets = {
        split: SnapshotWindowDataset(
            features=frames["features"],
            sample_index=frames["sample_index"],
            channels=config.channels,
            context_length=context_length,
            split=split,
        )
        for split in {"train", "valid", config.evaluation_split}
    }
    train_dataset = datasets["train"]
    valid_dataset = datasets["valid"]
    evaluation_dataset = datasets[config.evaluation_split]
    config_payload = config.model_dump(mode="json")
    config_hash = sha256_json(config_payload)
    resume_path = Path(resume_from).resolve() if resume_from is not None else None
    if resume_path is None:
        run_dir, run_id, created_at = _new_run_dir(config, str(dataset_manifest["dataset_id"]))
    else:
        run_dir, run_id, created_at = _resume_run_dir(
            resume_path,
            dataset_id=str(dataset_manifest["dataset_id"]),
            config_hash=config_hash,
        )

    manifest_path = run_dir / "manifest.json"
    resolved_config_path = run_dir / "resolved_config.yaml"
    initial_manifest = {
        "schema_version": 1,
        "status": "running",
        "run_id": run_id,
        "created_at": created_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": config.experiment_id,
        "model_type": "random_patchtst",
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_path": str(dataset_path),
        "dataset_manifest_hash": sha256_file(dataset_path / "manifest.json"),
        "config_hash": config_hash,
        "seed": config.seed,
        "evaluation_split": config.evaluation_split,
        "test_unlocked": config.unlock_test,
        "resumed_from": str(resume_path) if resume_path is not None else None,
    }
    resolved_config_path.write_text(
        yaml.safe_dump(config_payload, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    _write_json(manifest_path, initial_manifest)

    try:
        model, training_audit = train_e1(
            config,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            dataset_id=str(dataset_manifest["dataset_id"]),
            checkpoint_dir=run_dir / "checkpoints",
            resume_from=resume_path,
        )
        best_checkpoint = run_dir / "checkpoints" / "best.pt"
        checkpoint_hash = sha256_file(best_checkpoint)
        scores = predict_e1(
            model,
            evaluation_dataset,
            batch_size=config.training.batch_size,
            device=training_audit["device"],
            precision=training_audit["precision"],
            num_workers=config.training.num_workers,
        )
        evaluation_rows = evaluation_dataset.sample_rows
        predictions, neutralization_audit = build_prediction_frame(
            evaluation_rows,
            frames["sample_metadata"],
            scores,
            model_id=config.experiment_id,
            checkpoint_hash=checkpoint_hash,
            dataset_id=str(dataset_manifest["dataset_id"]),
        )
        coverage = prediction_coverage(
            predictions,
            frames["sample_index"],
            split=config.evaluation_split,
            minimum=config.minimum_coverage,
        )
        source_provenance = load_source_provenance(dataset_path, dataset_manifest)
        factor_metrics = apply_source_readiness_gate(
            evaluate_predictions(predictions, config.costs_bps), source_provenance
        )
        metrics = {
            "schema_version": 1,
            "run_id": run_id,
            "model_id": config.experiment_id,
            "dataset_id": dataset_manifest["dataset_id"],
            "evaluation_split": config.evaluation_split,
            "coverage": coverage,
            "neutralization": neutralization_audit,
            "source_provenance": source_provenance,
            "metrics": factor_metrics,
        }
        predictions.write_parquet(run_dir / "predictions.parquet")
        _write_json(run_dir / "metrics.json", metrics)
        write_evaluation_report(metrics, run_dir / "report.html")
        manifest = {
            **initial_manifest,
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "architecture": config.model.model_dump(mode="json"),
            "input": {
                "context_length": context_length,
                "channels": config.channels,
                "feature_scaler": dataset_config["features"].get("scaler"),
                "model_internal_scaling": config.model.scaling,
            },
            "row_counts": {
                "train": len(train_dataset),
                "valid": len(valid_dataset),
                "evaluation": len(evaluation_dataset),
            },
            "checkpoint": {
                "file": "checkpoints/best.pt",
                "sha256": checkpoint_hash,
                "last_file": "checkpoints/last.pt",
                "last_sha256": sha256_file(run_dir / "checkpoints" / "last.pt"),
            },
            "training": training_audit,
            "source_provenance": source_provenance,
            "git": collect_git_state(repository_root),
            "environment": collect_environment(include_model_dependencies=True),
            "artifacts": {
                "predictions": "predictions.parquet",
                "metrics": "metrics.json",
                "report": "report.html",
                "resolved_config": "resolved_config.yaml",
            },
        }
        _write_json(manifest_path, manifest)
    except Exception as exc:
        failed_manifest = {
            **initial_manifest,
            "status": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "recoverable_checkpoint": (
                "checkpoints/last.pt" if (run_dir / "checkpoints" / "last.pt").is_file() else None
            ),
        }
        _write_json(manifest_path, failed_manifest)
        raise
    return run_dir, metrics
