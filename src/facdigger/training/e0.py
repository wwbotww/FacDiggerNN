"""End-to-end E0 baseline training, prediction and evaluation."""

from __future__ import annotations

import json
import shutil
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
from facdigger.evaluation.contracts import prediction_coverage
from facdigger.evaluation.metrics import evaluate_predictions
from facdigger.evaluation.report import write_evaluation_report
from facdigger.experiments.manifest import collect_git_state, sha256_json
from facdigger.models.baselines import (
    TabularPreprocessor,
    build_multiscale_features,
    predict_mlp,
    train_lightgbm,
    train_mlp,
)
from facdigger.training.common import (
    apply_source_readiness_gate,
    build_prediction_frame,
    load_source_provenance,
    load_training_snapshot,
    split_supervised_training_index,
)
from facdigger.training.e0_config import E0ExperimentConfig


def run_e0(
    config: E0ExperimentConfig,
    dataset_dir: str | Path,
    *,
    repository_root: str | Path,
) -> tuple[Path, dict[str, Any]]:
    dataset_path = Path(dataset_dir).resolve()
    dataset_manifest, frames = load_training_snapshot(dataset_path)
    dataset_config = dataset_manifest["config"]
    context_length = int(dataset_config["features"]["context_length"])
    dataset_channels = list(dataset_config["features"]["channels"])
    if config.channels != dataset_channels:
        raise DataContractError(
            f"E0 channels must exactly match dataset channels: {dataset_channels}"
        )
    protocol_index, selection_audit = split_supervised_training_index(
        frames["sample_index"],
        selection_fraction=config.selection_fraction,
    )
    tabular, feature_columns = build_multiscale_features(
        frames["features"],
        protocol_index,
        channels=config.channels,
        windows=config.windows,
        context_length=context_length,
    )
    train_rows = tabular.filter(pl.col("split") == "train_fit")
    valid_rows = tabular.filter(pl.col("split") == "inner_selection")
    evaluation_rows = tabular.filter(pl.col("split") == config.evaluation_split)
    if train_rows.is_empty() or valid_rows.is_empty() or evaluation_rows.is_empty():
        raise DataContractError(
            "E0 requires non-empty train_fit, inner_selection and evaluation splits"
        )

    preprocessor = TabularPreprocessor.fit(train_rows, feature_columns)
    train_x = preprocessor.transform(train_rows)
    valid_x = preprocessor.transform(valid_rows)
    evaluation_x = preprocessor.transform(evaluation_rows)
    train_y = train_rows["target"].to_numpy().astype(np.float64)
    valid_y = valid_rows["target"].to_numpy().astype(np.float64)

    created_at = datetime.now(timezone.utc)
    config_payload = config.model_dump(mode="json")
    run_identity = {
        "config": config_payload,
        "dataset_id": dataset_manifest["dataset_id"],
        "created_at": created_at.isoformat(),
    }
    timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{config.experiment_id}-{timestamp}-{sha256_json(run_identity)[:8]}"
    output_root = config.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / run_id
    temporary_dir = output_root / f".tmp-{run_id}-{uuid.uuid4().hex}"
    temporary_dir.mkdir(parents=False, exist_ok=False)
    try:
        checkpoint_name = "best.pt" if config.model_type == "mlp" else "best.txt"
        checkpoint_path = temporary_dir / "checkpoints" / checkpoint_name
        if config.model_type == "mlp":
            model, training_audit = train_mlp(
                train_x,
                train_y,
                valid_x,
                valid_y,
                config=config.mlp,
                seed=config.seed,
                checkpoint_path=checkpoint_path,
                preprocessing=preprocessor.to_dict(),
            )
            scores = predict_mlp(model, evaluation_x, training_audit["device"])
        else:
            scores, training_audit = train_lightgbm(
                train_x,
                train_y,
                valid_x,
                valid_y,
                evaluation_x,
                config=config.lightgbm,
                seed=config.seed,
                checkpoint_path=checkpoint_path,
                preprocessing=preprocessor.to_dict(),
            )
        checkpoint_hash = sha256_file(checkpoint_path)
        preprocessing_artifact = None
        if config.model_type == "lightgbm":
            preprocessing_path = checkpoint_path.with_suffix(".preprocessing.json")
            preprocessing_artifact = {
                "file": str(preprocessing_path.relative_to(temporary_dir)),
                "sha256": sha256_file(preprocessing_path),
            }
        predictions, neutralization_audit = build_prediction_frame(
            evaluation_rows,
            frames["sample_metadata"],
            scores,
            model_id=config.experiment_id,
            checkpoint_hash=checkpoint_hash,
            dataset_id=dataset_manifest["dataset_id"],
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
        predictions.write_parquet(temporary_dir / "predictions.parquet")
        (temporary_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        write_evaluation_report(metrics, temporary_dir / "report.html")
        (temporary_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config_payload, allow_unicode=True, sort_keys=True), encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "run_id": run_id,
            "created_at": created_at.isoformat(),
            "model_id": config.experiment_id,
            "model_type": config.model_type,
            "dataset_id": dataset_manifest["dataset_id"],
            "dataset_path": str(dataset_path),
            "dataset_manifest_hash": sha256_file(dataset_path / "manifest.json"),
            "config_hash": sha256_json(config_payload),
            "seed": config.seed,
            "evaluation_split": config.evaluation_split,
            "test_unlocked": config.unlock_test,
            "feature_columns": feature_columns,
            "input_dimensions_with_masks": train_x.shape[1],
            "row_counts": {
                "train_fit": train_rows.height,
                "inner_selection": valid_rows.height,
                "evaluation": evaluation_rows.height,
            },
            "supervised_selection_audit": selection_audit,
            "checkpoint": {
                "file": f"checkpoints/{checkpoint_name}",
                "sha256": checkpoint_hash,
                "preprocessing": preprocessing_artifact,
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
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        temporary_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return final_dir, metrics
