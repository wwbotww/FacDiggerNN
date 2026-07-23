"""Independent evaluation of an immutable prediction table."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.environment import collect_environment
from facdigger.evaluation.contracts import prediction_coverage, validate_predictions
from facdigger.evaluation.metrics import evaluate_predictions
from facdigger.evaluation.report import write_evaluation_report
from facdigger.training.common import (
    apply_source_readiness_gate,
    load_source_provenance,
    load_training_snapshot,
)


def evaluate_prediction_file(
    predictions_path: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    costs_bps: list[float] | None = None,
    minimum_coverage: float = 1.0,
) -> tuple[Path, dict[str, Any]]:
    """Evaluate existing predictions without importing or executing a model."""

    source = Path(predictions_path).resolve()
    dataset_path = Path(dataset_dir).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"evaluation output already exists: {destination}")
    predictions = validate_predictions(pl.read_parquet(source))
    dataset_manifest, frames = load_training_snapshot(dataset_path, include_features=False)
    dataset_ids = predictions["dataset_id"].unique().to_list()
    if dataset_ids != [dataset_manifest["dataset_id"]]:
        raise DataContractError("prediction dataset_id does not match evaluation snapshot")
    splits = predictions["split"].unique().to_list()
    if len(splits) != 1 or splits[0] not in {"train", "valid", "test"}:
        raise DataContractError("prediction file must contain exactly one recognized split")
    split = str(splits[0])
    expected = frames["sample_index"].filter(pl.col("split") == split).select(
        "security_id", "asof_date", pl.col("target").alias("expected_target")
    )
    checked = predictions.join(
        expected,
        on=["security_id", "asof_date"],
        how="left",
        validate="1:1",
    )
    if checked["expected_target"].null_count():
        raise DataContractError("predictions contain keys absent from the snapshot split")
    if not np.allclose(
        checked["target"].to_numpy(),
        checked["expected_target"].to_numpy(),
        rtol=0,
        atol=1e-12,
    ):
        raise DataContractError("prediction targets differ from immutable snapshot labels")
    coverage = prediction_coverage(
        predictions,
        frames["sample_index"],
        split=split,
        minimum=minimum_coverage,
    )
    provenance = load_source_provenance(dataset_path, dataset_manifest)
    factor_metrics = apply_source_readiness_gate(
        evaluate_predictions(predictions, costs_bps or [0.0, 10.0, 20.0, 50.0]),
        provenance,
    )
    metrics = {
        "schema_version": 1,
        "dataset_id": dataset_manifest["dataset_id"],
        "model_id": predictions["model_id"][0],
        "evaluation_split": split,
        "coverage": coverage,
        "source_provenance": provenance,
        "metrics": factor_metrics,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".tmp-{destination.name}-{uuid.uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        metrics_path = temporary / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        write_evaluation_report(metrics, temporary / "report.html")
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prediction_source": str(source),
            "prediction_sha256": sha256_file(source),
            "dataset_path": str(dataset_path),
            "dataset_id": dataset_manifest["dataset_id"],
            "dataset_manifest_sha256": sha256_file(dataset_path / "manifest.json"),
            "evaluation_split": split,
            "row_count": predictions.height,
            "coverage": coverage,
            "environment": collect_environment(include_model_dependencies=False),
            "artifacts": {
                "metrics": {"file": "metrics.json", "sha256": sha256_file(metrics_path)},
                "report": "report.html",
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            + "\n",
            encoding="utf-8",
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination, manifest
