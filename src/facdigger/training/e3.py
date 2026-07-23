"""End-to-end E3 financial pretraining, fine-tuning and evaluation."""

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
    split_supervised_training_index,
)
from facdigger.training.e1_engine import predict_e1
from facdigger.training.e2_config import E2ExperimentConfig
from facdigger.training.e2_engine import train_e2
from facdigger.training.e3_config import E3ExperimentConfig
from facdigger.training.e3_engine import (
    PretrainingInitializer,
    initialize_alpha_from_financial_checkpoint,
    split_pretraining_index,
    train_financial_pretraining,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _downstream_config(config: E3ExperimentConfig) -> E2ExperimentConfig:
    return E2ExperimentConfig.model_validate(
        {
            "experiment_id": config.experiment_id,
            "seed": config.seed,
            "output_root": config.output_root,
            "evaluation_split": config.evaluation_split,
            "unlock_test": config.unlock_test,
            "minimum_coverage": config.minimum_coverage,
            "selection_fraction": config.selection_fraction,
            "channels": config.channels,
            "costs_bps": config.costs_bps,
            "model": config.model.model_dump(mode="json"),
            "source": config.source.model_dump(mode="json"),
            "training": config.finetuning.model_dump(mode="json"),
        }
    )


def _new_run_dir(config: E3ExperimentConfig, dataset_id: str) -> tuple[Path, str, str]:
    created_at = datetime.now(timezone.utc)
    identity = {
        "config": config.model_dump(mode="json"),
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
) -> tuple[Path, str, str, str]:
    checkpoint = resume_from.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"E3 resume checkpoint not found: {checkpoint}")
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    family = str(payload.get("experiment_family", ""))
    if family == "e3_pretraining":
        valid_parent = (
            checkpoint.parent.name == "checkpoints"
            and checkpoint.parent.parent.name == "pretraining"
        )
        if not valid_parent:
            raise ValueError("E3 pretraining checkpoint is outside pretraining/checkpoints")
        run_dir = checkpoint.parent.parent.parent
    elif family == "e2":
        if checkpoint.parent.name != "checkpoints":
            raise ValueError("E3 fine-tuning checkpoint is outside checkpoints")
        run_dir = checkpoint.parent.parent
    else:
        raise ValueError("resume checkpoint is not an E3 phase checkpoint")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"E3 run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") == "complete":
        raise ValueError("cannot resume an already-complete E3 run")
    if manifest.get("dataset_id") != dataset_id:
        raise ValueError("resume run dataset_id does not match")
    if manifest.get("config_hash") != config_hash:
        raise ValueError("resume run configuration does not match")
    return run_dir, str(manifest["run_id"]), str(manifest["created_at"]), family


def run_e3(
    config: E3ExperimentConfig,
    dataset_dir: str | Path,
    *,
    repository_root: str | Path,
    resume_from: str | Path | None = None,
    pretraining_initializer: PretrainingInitializer | None = None,
) -> tuple[Path, dict[str, Any]]:
    dataset_path = Path(dataset_dir).resolve()
    dataset_manifest, frames = load_training_snapshot(dataset_path)
    dataset_config = dataset_manifest["config"]
    context_length = int(dataset_config["features"]["context_length"])
    dataset_channels = list(dataset_config["features"]["channels"])
    if config.channels != dataset_channels:
        raise DataContractError(
            f"E3 channels must exactly match dataset channels: {dataset_channels}"
        )
    pretrain_index, selection_index, leakage_audit = split_pretraining_index(
        frames["sample_index"],
        validation_fraction=config.pretraining.validation_fraction,
    )
    pretrain_dataset = SnapshotWindowDataset(
        features=frames["features"],
        sample_index=pretrain_index,
        channels=config.channels,
        context_length=context_length,
        split="pretrain_train",
    )
    selection_dataset = SnapshotWindowDataset(
        features=frames["features"],
        sample_index=selection_index,
        channels=config.channels,
        context_length=context_length,
        split="pretrain_selection",
    )
    protocol_index, supervised_selection_audit = split_supervised_training_index(
        frames["sample_index"],
        selection_fraction=config.selection_fraction,
    )
    training_datasets = {
        split: SnapshotWindowDataset(
            features=frames["features"],
            sample_index=protocol_index,
            channels=config.channels,
            context_length=context_length,
            split=split,
        )
        for split in {"train_fit", "inner_selection"}
    }
    evaluation_dataset = SnapshotWindowDataset(
        features=frames["features"],
        sample_index=frames["sample_index"],
        channels=config.channels,
        context_length=context_length,
        split=config.evaluation_split,
    )
    config_payload = config.model_dump(mode="json")
    config_hash = sha256_json(config_payload)
    resume_path = Path(resume_from).resolve() if resume_from is not None else None
    resume_family: str | None = None
    if resume_path is None:
        run_dir, run_id, created_at = _new_run_dir(
            config, str(dataset_manifest["dataset_id"])
        )
    else:
        run_dir, run_id, created_at, resume_family = _resume_run_dir(
            resume_path,
            dataset_id=str(dataset_manifest["dataset_id"]),
            config_hash=config_hash,
        )

    manifest_path = run_dir / "manifest.json"
    initial_manifest = {
        "schema_version": 1,
        "status": "running",
        "phase": "financial_pretraining" if resume_family != "e2" else "fine_tuning",
        "run_id": run_id,
        "created_at": created_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": config.experiment_id,
        "model_type": "financial_pretrained_patchtst",
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_path": str(dataset_path),
        "dataset_manifest_hash": sha256_file(dataset_path / "manifest.json"),
        "config_hash": config_hash,
        "seed": config.seed,
        "evaluation_split": config.evaluation_split,
        "test_unlocked": config.unlock_test,
        "source_checkpoint": config.source.model_dump(mode="json"),
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "pretraining_leakage_audit": leakage_audit,
        "supervised_selection_audit": supervised_selection_audit,
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config_payload, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    _write_json(manifest_path, initial_manifest)

    current_phase = str(initial_manifest["phase"])
    try:
        pretraining_checkpoint = run_dir / "pretraining" / "checkpoints" / "best.pt"
        if resume_family == "e2":
            if not pretraining_checkpoint.is_file():
                raise FileNotFoundError(
                    "E3 fine-tuning resume has no selected pretraining checkpoint"
                )
            import torch

            pretraining_payload = torch.load(
                pretraining_checkpoint, map_location="cpu", weights_only=False
            )
            pretraining_audit = {
                "device": "already_completed",
                "best_epoch": pretraining_payload["best_epoch"],
                "best_selection_loss": pretraining_payload["best_selection_loss"],
                "epochs_completed": pretraining_payload["epoch"],
                "history": pretraining_payload["history"],
                "leakage_audit": pretraining_payload["leakage_audit"],
                "best_checkpoint": str(pretraining_checkpoint),
            }
            source_audit = pretraining_payload["initialization_audit"]
        else:
            _, pretraining_audit, source_audit = train_financial_pretraining(
                config,
                train_dataset=pretrain_dataset,
                selection_dataset=selection_dataset,
                leakage_audit=leakage_audit,
                dataset_id=str(dataset_manifest["dataset_id"]),
                checkpoint_dir=run_dir / "pretraining" / "checkpoints",
                resume_from=resume_path if resume_family == "e3_pretraining" else None,
                initializer=pretraining_initializer,
            )
        _write_json(run_dir / "pretraining" / "training_audit.json", pretraining_audit)
        _write_json(run_dir / "pretraining" / "weight_load_report.json", source_audit)
        _write_json(
            manifest_path,
            {
                **initial_manifest,
                "phase": "fine_tuning",
                "pretraining": pretraining_audit,
            },
        )
        current_phase = "fine_tuning"
        downstream = _downstream_config(config)

        def initializer() -> tuple[Any, dict[str, Any]]:
            return initialize_alpha_from_financial_checkpoint(
                config,
                context_length=context_length,
                checkpoint_path=pretraining_checkpoint,
            )

        model, finetuning_audit, chain_audit = train_e2(
            downstream,
            train_dataset=training_datasets["train_fit"],
            valid_dataset=training_datasets["inner_selection"],
            dataset_id=str(dataset_manifest["dataset_id"]),
            checkpoint_dir=run_dir / "checkpoints",
            resume_from=resume_path if resume_family == "e2" else None,
            initializer=initializer,
        )
        _write_json(run_dir / "weight_load_report.json", chain_audit)
        best_checkpoint = run_dir / "checkpoints" / "best.pt"
        checkpoint_hash = sha256_file(best_checkpoint)
        scores = predict_e1(
            model,
            evaluation_dataset,
            batch_size=config.finetuning.batch_size,
            device=finetuning_audit["device"],
            precision=finetuning_audit["precision"],
            num_workers=config.finetuning.num_workers,
        )
        predictions, neutralization_audit = build_prediction_frame(
            evaluation_dataset.sample_rows,
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
            "phase": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "architecture": config.model.model_dump(mode="json"),
            "input": {
                "context_length": context_length,
                "channels": config.channels,
                "feature_scaler": dataset_config["features"].get("scaler"),
                "model_internal_scaling": config.model.scaling,
            },
            "row_counts": {
                "pretrain": len(pretrain_dataset),
                "pretrain_selection": len(selection_dataset),
                "train_fit": len(training_datasets["train_fit"]),
                "inner_selection": len(training_datasets["inner_selection"]),
                "evaluation": len(evaluation_dataset),
            },
            "pretraining": pretraining_audit,
            "weight_loading": chain_audit,
            "finetuning": finetuning_audit,
            "checkpoint": {
                "file": "checkpoints/best.pt",
                "sha256": checkpoint_hash,
                "pretraining_file": "pretraining/checkpoints/best.pt",
                "pretraining_sha256": sha256_file(pretraining_checkpoint),
            },
            "source_provenance": source_provenance,
            "git": collect_git_state(repository_root),
            "environment": collect_environment(include_model_dependencies=True),
            "artifacts": {
                "weight_load_report": "weight_load_report.json",
                "pretraining_weight_load_report": "pretraining/weight_load_report.json",
                "pretraining_training_audit": "pretraining/training_audit.json",
                "predictions": "predictions.parquet",
                "metrics": "metrics.json",
                "report": "report.html",
                "resolved_config": "resolved_config.yaml",
            },
        }
        _write_json(manifest_path, manifest)
    except Exception as exc:
        downstream_last = run_dir / "checkpoints" / "last.pt"
        pretraining_last = run_dir / "pretraining" / "checkpoints" / "last.pt"
        recoverable = downstream_last if downstream_last.is_file() else pretraining_last
        _write_json(
            manifest_path,
            {
                **initial_manifest,
                "status": "failed",
                "phase": current_phase,
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "recoverable_checkpoint": (
                    str(recoverable.relative_to(run_dir)) if recoverable.is_file() else None
                ),
            },
        )
        raise
    return run_dir, metrics
