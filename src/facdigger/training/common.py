"""Shared snapshot and prediction helpers for E0-E3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.evaluation.contracts import validate_predictions
from facdigger.evaluation.neutralization import neutralize_predictions


def load_training_snapshot(
    dataset_dir: Path, *, include_features: bool = True
) -> tuple[dict[str, Any], dict[str, pl.DataFrame]]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version", 0) < 2:
        raise DataContractError(
            "training requires dataset snapshot schema >= 2 with sample_metadata.parquet"
        )
    filenames = {
        "sample_index": "sample_index.parquet",
        "sample_metadata": "sample_metadata.parquet",
    }
    inference_filename = manifest.get("artifacts", {}).get("inference_index")
    if inference_filename:
        filenames["inference_index"] = str(inference_filename)
    if include_features:
        filenames["features"] = "features.parquet"
    frames = {name: pl.read_parquet(dataset_dir / filename) for name, filename in filenames.items()}
    return manifest, frames


def load_source_provenance(dataset_dir: Path, dataset_manifest: dict[str, Any]) -> dict[str, Any]:
    filename = dataset_manifest.get("artifacts", {}).get("source_manifest")
    if not filename:
        return {
            "available": False,
            "research_ready": None,
            "warnings": [],
        }
    path = dataset_dir / str(filename)
    if not path.is_file():
        raise DataContractError(f"snapshot source provenance is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    selection = payload.get("selection") or {}
    return {
        "available": True,
        "provider": payload.get("provider"),
        "source_revision": payload.get("source_revision"),
        "manifest_sha256": sha256_file(path),
        "selection": selection,
        "research_ready": selection.get("research_ready"),
        "warnings": list(payload.get("warnings") or []),
    }


def apply_source_readiness_gate(
    factor_metrics: dict[str, Any], provenance: dict[str, Any]
) -> dict[str, Any]:
    cross_section = factor_metrics["cross_section"]
    statistical_ready = bool(cross_section["research_ready"])
    source_ready = provenance.get("research_ready")
    cross_section["statistical_ready"] = statistical_ready
    cross_section["source_research_ready"] = source_ready
    cross_section["research_ready"] = statistical_ready and source_ready is not False
    cross_section["research_ready_rule"] = (
        "statistical cross-section gate passes and source provenance is not explicitly blocked"
    )
    return factor_metrics


def build_prediction_frame(
    rows: pl.DataFrame,
    metadata: pl.DataFrame,
    scores: np.ndarray,
    *,
    model_id: str,
    checkpoint_hash: str,
    dataset_id: str,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    predictions = (
        rows.select("sample_id", "security_id", "symbol", "asof_date", "split", "target")
        .with_columns(pl.Series("score_raw", scores, dtype=pl.Float64))
        .join(
            metadata.select(
                "sample_id",
                "eligible",
                "industry_code",
                "log_float_market_cap",
            ),
            on="sample_id",
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.lit(None, dtype=pl.Float64).alias("score_neutralized"),
            pl.lit(model_id).alias("model_id"),
            pl.lit(checkpoint_hash).alias("checkpoint_hash"),
            pl.lit(dataset_id).alias("dataset_id"),
        )
        .drop("sample_id")
    )
    predictions, neutralization_audit = neutralize_predictions(predictions)
    return validate_predictions(predictions), neutralization_audit
