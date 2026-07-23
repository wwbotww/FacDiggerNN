"""Build content-addressed, immutable dataset snapshots."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from facdigger.data.adapters import StandardParquetAdapter
from facdigger.data.config import DatasetBuildConfig
from facdigger.datasets.index import build_sample_index
from facdigger.datasets.splits import assign_chronological_splits
from facdigger.experiments.manifest import sha256_json
from facdigger.features.price_volume import build_price_volume_features
from facdigger.features.scaling import apply_robust_scaler, fit_train_robust_scaler
from facdigger.labels.forward_return import build_forward_excess_return_labels


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(config: DatasetBuildConfig) -> dict[str, str | None]:
    paths = {
        "bars": config.sources.bars,
        "universe": config.sources.universe,
        "corporate_actions": config.sources.corporate_actions,
        "delistings": config.sources.delistings,
        "source_manifest": config.sources.source_manifest,
    }
    missing = [name for name, path in paths.items() if path is not None and not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Configured dataset source files do not exist: {missing}")
    return {name: sha256_file(path) if path is not None else None for name, path in paths.items()}


def _feature_audit(features: pl.DataFrame) -> dict[str, Any]:
    observed_columns = [column for column in features.columns if column.startswith("observed_")]
    return {
        "rows": features.height,
        "observed_ratio": {
            column.removeprefix("observed_"): float(features[column].mean() or 0.0)
            for column in observed_columns
        },
    }


def _build_sample_metadata(sample_index: pl.DataFrame, universe: pl.DataFrame) -> pl.DataFrame:
    return (
        sample_index.select("sample_id", "security_id", "symbol", "asof_date", "split")
        .join(
            universe.select(
                "security_id",
                pl.col("trade_date").alias("asof_date"),
                "eligible",
                "industry_code",
                "float_market_cap",
            ),
            on=["security_id", "asof_date"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.when(pl.col("float_market_cap") > 0)
            .then(pl.col("float_market_cap").log())
            .otherwise(None)
            .alias("log_float_market_cap")
        )
        .sort(["asof_date", "security_id"])
    )


def build_dataset_snapshot(config: DatasetBuildConfig) -> tuple[Path, dict[str, Any]]:
    adapter = StandardParquetAdapter(config.sources)
    bundle = adapter.load()
    input_hashes = _source_hashes(config)
    semantic_config = config.model_dump(mode="json")
    source_paths = semantic_config.pop("sources")
    semantic_config.pop("output_root")
    identity = {
        "schema_version": 2,
        "config": semantic_config,
        "input_file_hashes": input_hashes,
    }
    dataset_id = sha256_json(identity)
    output_root = config.output_root.resolve()
    final_dir = output_root / dataset_id
    if final_dir.exists():
        manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
        return final_dir, manifest

    raw_features = build_price_volume_features(bundle.bars, bundle.universe)
    scaler = fit_train_robust_scaler(
        raw_features,
        config.features.channels,
        config.split.train_end,
        winsor_lower=config.features.winsor_lower,
        winsor_upper=config.features.winsor_upper,
    )
    features = apply_robust_scaler(raw_features, scaler)
    labels = build_forward_excess_return_labels(
        bundle.bars,
        bundle.universe,
        delistings=bundle.delistings,
        execution_lag=config.label.execution_lag,
        horizon=config.label.horizon,
    )
    calendar = bundle.universe["trade_date"].unique().sort().to_list()
    labels = assign_chronological_splits(labels, calendar, config.split)
    sample_index = build_sample_index(
        features,
        labels,
        bundle.universe,
        context_length=config.features.context_length,
    )
    sample_metadata = _build_sample_metadata(sample_index, bundle.universe)
    split_counts = {
        row["split"]: row["len"]
        for row in sample_index.group_by("split").len().sort("split").to_dicts()
    }
    audit = {
        "sources": adapter.audit(bundle),
        "features": _feature_audit(features),
        "labels": {
            "rows": labels.height,
            "target_non_null": labels["target"].is_not_null().sum(),
            "crosses_delisting": labels["crosses_delisting"].sum(),
        },
        "sample_index": {
            "rows": sample_index.height,
            "split_counts": split_counts,
        },
    }
    manifest = {
        **identity,
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_paths": source_paths,
        "artifacts": {
            "features": "features.parquet",
            "labels": "labels.parquet",
            "sample_index": "sample_index.parquet",
            "sample_metadata": "sample_metadata.parquet",
            "audit": "audit.json",
            "scaler": "scaler.json",
            "source_manifest": (
                "source_manifest.json" if config.sources.source_manifest is not None else None
            ),
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_root / f".tmp-{dataset_id[:12]}-{uuid.uuid4().hex}"
    temporary_dir.mkdir(parents=False, exist_ok=False)
    try:
        features.write_parquet(temporary_dir / "features.parquet")
        labels.write_parquet(temporary_dir / "labels.parquet")
        sample_index.write_parquet(temporary_dir / "sample_index.parquet")
        sample_metadata.write_parquet(temporary_dir / "sample_metadata.parquet")
        (temporary_dir / "audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary_dir / "scaler.json").write_text(
            json.dumps(scaler, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if config.sources.source_manifest is not None:
            shutil.copyfile(
                config.sources.source_manifest,
                temporary_dir / "source_manifest.json",
            )
        temporary_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return final_dir, manifest
