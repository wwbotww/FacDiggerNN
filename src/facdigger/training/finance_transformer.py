"""End-to-end finance-native Transformer training and evaluation."""

from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.datasets.window import (
    FinanceTransformerWindowDataset,
    MarketFeatureStore,
    SecurityFeatureStore,
)
from facdigger.environment import collect_environment
from facdigger.evaluation.contracts import prediction_coverage
from facdigger.evaluation.metrics import evaluate_predictions
from facdigger.evaluation.report import write_evaluation_report
from facdigger.experiments.manifest import collect_git_state
from facdigger.models.finance_scoring import predict_finance_transformer
from facdigger.training.common import (
    apply_source_readiness_gate,
    build_prediction_frame,
    load_required_market_features,
    load_required_snapshot_features,
    load_source_provenance,
    load_training_snapshot,
    split_supervised_training_index,
)
from facdigger.training.finance_transformer_config import (
    FinanceTransformerExperimentConfig,
)
from facdigger.training.finance_transformer_engine import (
    train_finance_transformer,
)
from facdigger.training.run_state import TrainingRun, training_run
from facdigger.training.runtime import (
    TrainingControl,
    TrainingRuntimeConfig,
)
from facdigger.training.runtime import (
    write_json as _write_json,
)


def _append_progress(path: Path, payload: dict[str, Any]) -> None:
    event = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def run_finance_transformer(
    config: FinanceTransformerExperimentConfig,
    dataset_dir: str | Path,
    *,
    repository_root: str | Path,
    resume_from: str | Path | None = None,
    runtime: TrainingRuntimeConfig | None = None,
    run_dir: str | Path | None = None,
    control: TrainingControl | None = None,
) -> tuple[Path, dict[str, Any]]:
    session_control = control or TrainingControl(runtime)
    with nullcontext(session_control) if control is not None else session_control:
        with training_run(
            config,
            dataset_dir,
            repository_root=repository_root,
            model_type="finance_patch_transformer",
            control=session_control,
            resume_from=resume_from,
            run_dir=run_dir,
        ) as run:
            if run.manifest["status"] == "complete":
                return run.path, json.loads((run.path / "metrics.json").read_text(encoding="utf-8"))
            return _run_finance_transformer(
                config, run, repository_root=repository_root, control=session_control
            )


def _run_finance_transformer(
    config: FinanceTransformerExperimentConfig,
    run: TrainingRun,
    *,
    repository_root: str | Path,
    control: TrainingControl,
) -> tuple[Path, dict[str, Any]]:
    dataset_path = run.dataset
    dataset_manifest, frames = load_training_snapshot(dataset_path, include_features=False)
    dataset_config = dataset_manifest["config"]
    feature_config = dataset_config["features"]
    label_config = dataset_config["label"]
    if feature_config.get("name") != "finance_transformer":
        raise DataContractError("finance Transformer requires its dedicated feature set")
    if config.channels != list(feature_config["channels"]):
        raise DataContractError("local channels differ from the dataset snapshot")
    if config.market_channels != list(feature_config["market_channels"]):
        raise DataContractError("market channels differ from the dataset snapshot")
    dataset_horizons = sorted(
        [label_config["horizon"], *label_config.get("auxiliary_horizons", [])]
    )
    if config.horizons != dataset_horizons:
        raise DataContractError("model horizons differ from the dataset snapshot")
    if config.primary_horizon != int(label_config["horizon"]):
        raise DataContractError("primary horizon differs from the dataset snapshot")
    context_length = int(feature_config["context_length"])

    protocol_index, selection_audit = split_supervised_training_index(
        frames["sample_index"], selection_fraction=config.selection_fraction
    )
    required_rows = pl.concat(
        [
            protocol_index.filter(pl.col("split").is_in(["train_fit", "inner_selection"])).select(
                "security_id", "feature_start", "asof_date"
            ),
            frames["sample_index"]
            .filter(pl.col("split") == config.evaluation_split)
            .select("security_id", "feature_start", "asof_date"),
        ],
        how="vertical",
    )
    feature_store = SecurityFeatureStore(
        features=load_required_snapshot_features(dataset_path, dataset_manifest, required_rows),
        channels=config.channels,
        presorted=True,
    )
    market_store = MarketFeatureStore(
        features=load_required_market_features(dataset_path, dataset_manifest, required_rows),
        channels=config.market_channels,
    )

    def dataset(index: pl.DataFrame, split: str) -> FinanceTransformerWindowDataset:
        return FinanceTransformerWindowDataset(
            feature_store=feature_store,
            market_store=market_store,
            sample_index=index,
            channels=config.channels,
            market_channels=config.market_channels,
            context_length=context_length,
            split=split,
            horizons=config.horizons,
            primary_horizon=config.primary_horizon,
        )

    train_dataset = dataset(protocol_index, "train_fit")
    selection_dataset = dataset(protocol_index, "inner_selection")
    evaluation_dataset = dataset(frames["sample_index"], config.evaluation_split)
    resume_path = run.resume
    run_dir = run.path
    manifest_path = run_dir / "manifest.json"
    initial_manifest = {
        **run.manifest,
        "status": "running",
        "run_id": run.manifest["run_id"],
        "created_at": run.manifest["created_at"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": config.experiment_id,
        "model_type": "finance_patch_transformer",
        "initialization": config.initialization,
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_path": str(dataset_path),
        "dataset_manifest_hash": sha256_file(dataset_path / "manifest.json"),
        "config_hash": run.manifest["config_hash"],
        "seed": config.seed,
        "evaluation_split": config.evaluation_split,
        "test_unlocked": config.unlock_test,
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "supervised_selection_audit": selection_audit,
    }
    _write_json(manifest_path, initial_manifest)

    model, training_audit = train_finance_transformer(
        config,
        train_dataset=train_dataset,
        valid_dataset=selection_dataset,
        dataset_id=str(dataset_manifest["dataset_id"]),
        checkpoint_dir=run_dir / "checkpoints",
        resume_from=resume_path,
        control=control,
        progress_callback=lambda event: _append_progress(run_dir / "progress.jsonl", event),
    )
    control.raise_if_stopping(run_dir / "checkpoints" / "last.pt")
    best_checkpoint = run_dir / "checkpoints" / "best.pt"
    checkpoint_hash = sha256_file(best_checkpoint)
    scores = predict_finance_transformer(
        model,
        evaluation_dataset,
        batch_size=config.training.batch_size,
        device=training_audit["device"],
        precision=training_audit["precision"],
        num_workers=config.training.num_workers,
        check_stop=lambda: control.raise_if_stopping(run_dir / "checkpoints" / "last.pt"),
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
        "run_id": run.manifest["run_id"],
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
            "feature_set": feature_config["name"],
            "context_length": context_length,
            "channels": config.channels,
            "market_channels": config.market_channels,
            "horizons": config.horizons,
            "primary_horizon": config.primary_horizon,
            "feature_scaler": feature_config["scaler"],
            "feature_scaler_sha256": sha256_file(dataset_path / "scaler.json"),
            "model_internal_scaling": None,
        },
        "row_counts": {
            "train_fit": len(train_dataset),
            "inner_selection": len(selection_dataset),
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
        "predictions_sha256": sha256_file(run_dir / "predictions.parquet"),
        "git": collect_git_state(repository_root),
        "environment": collect_environment(include_model_dependencies=True),
        "artifacts": {
            "predictions": "predictions.parquet",
            "metrics": "metrics.json",
            "report": "report.html",
            "progress": "progress.jsonl",
            "resolved_config": "resolved_config.yaml",
        },
    }
    control.raise_if_stopping(run_dir / "checkpoints" / "last.pt")
    _write_json(manifest_path, manifest)
    return run_dir, metrics
