"""Shared snapshot and prediction helpers for E0-E3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from facdigger.data.contracts import DataContractError
from facdigger.data.provenance import (
    read_source_provenance_manifest,
    require_accepted_source,
)
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
    if include_features:
        filenames["features"] = str(
            manifest.get("artifacts", {}).get("features", "features.parquet")
        )
    frames = {
        name: pl.read_parquet(dataset_dir / filename)
        for name, filename in filenames.items()
    }
    return manifest, frames


def load_required_snapshot_features(
    dataset_dir: Path,
    dataset_manifest: dict[str, Any],
    required_rows: pl.DataFrame,
) -> pl.DataFrame:
    """Read only feature columns and security/date ranges used by one run."""

    required_columns = {"security_id", "feature_start", "asof_date"}
    missing = sorted(required_columns - set(required_rows.columns))
    if missing:
        raise DataContractError(f"required feature rows missing columns: {missing}")
    if required_rows.is_empty():
        raise DataContractError("cannot load features for an empty sample selection")
    feature_config = dataset_manifest["config"]["features"]
    channels = list(feature_config["channels"])
    observed = [f"observed_{channel}" for channel in channels]
    end_column = "future_end" if "future_end" in required_rows.columns else "asof_date"
    bounds = (
        required_rows.select(
            "security_id",
            "feature_start",
            pl.col(end_column).alias("_required_sample_end"),
        )
        .group_by("security_id")
        .agg(
            pl.col("feature_start").min().alias("_required_start"),
            pl.col("_required_sample_end").max().alias("_required_end"),
        )
    )
    filename = str(
        dataset_manifest.get("artifacts", {}).get("features", "features.parquet")
    )
    path = dataset_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot features do not exist: {path}")
    return (
        pl.scan_parquet(path)
        .select("security_id", "trade_date", *channels, *observed)
        .join(bounds.lazy(), on="security_id", how="inner")
        .filter(
            (pl.col("trade_date") >= pl.col("_required_start"))
            & (pl.col("trade_date") <= pl.col("_required_end"))
        )
        .drop("_required_start", "_required_end")
        .collect()
        .sort(["security_id", "trade_date"])
    )


def load_required_market_features(
    dataset_dir: Path,
    dataset_manifest: dict[str, Any],
    required_rows: pl.DataFrame,
) -> pl.DataFrame:
    """Read the single market sequence range needed by one Transformer run."""

    required_columns = {"feature_start", "asof_date"}
    missing = sorted(required_columns - set(required_rows.columns))
    if missing:
        raise DataContractError(f"required market rows missing columns: {missing}")
    if required_rows.is_empty():
        raise DataContractError("cannot load market features for an empty sample selection")
    feature_config = dataset_manifest["config"]["features"]
    channels = list(feature_config.get("market_channels") or [])
    if not channels:
        raise DataContractError("dataset does not declare market context channels")
    filename = dataset_manifest.get("artifacts", {}).get("market_features")
    if not isinstance(filename, str):
        raise DataContractError("dataset does not contain market context features")
    path = dataset_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot market features do not exist: {path}")
    start = required_rows["feature_start"].min()
    end_column = "future_end" if "future_end" in required_rows.columns else "asof_date"
    end = required_rows[end_column].max()
    observed = [f"observed_{channel}" for channel in channels]
    return (
        pl.scan_parquet(path)
        .select("trade_date", *channels, *observed)
        .filter(pl.col("trade_date").is_between(start, end))
        .collect()
        .sort("trade_date")
    )


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
    provenance = read_source_provenance_manifest(path)
    require_accepted_source(provenance)
    provenance["manifest_sha256"] = sha256_file(path)
    return provenance


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


def load_snapshot_inference_rows(
    dataset_dir: Path,
    dataset_manifest: dict[str, Any],
    *,
    asof_dates: list[Any],
) -> pl.DataFrame:
    """Read target-free computational windows only on requested supervised dates."""

    from facdigger.data.paths import artifact_path

    if not asof_dates:
        raise DataContractError("cannot load a computational universe for no dates")
    filename = dataset_manifest.get("artifacts", {}).get("inference_index")
    if not isinstance(filename, str):
        raise DataContractError("Finance training requires a target-free inference_index")
    path = artifact_path(dataset_dir, filename, "inference_index")
    rows = (
        pl.scan_parquet(path)
        .select(
            "sample_id", "security_id", "symbol", "asof_date",
            "feature_start", "feature_end", "eligible",
        )
        .filter(pl.col("asof_date").is_in(asof_dates))
        .collect()
        .sort("asof_date", "security_id")
    )
    if rows.is_empty() or set(rows["asof_date"].to_list()) != set(asof_dates):
        raise DataContractError("computational universe is missing requested dates")
    if rows["eligible"].null_count() or not rows["eligible"].all():
        raise DataContractError("inference_index contains ineligible computational rows")
    return rows
