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
from facdigger.data.config import (
    DatasetBuildConfig,
)
from facdigger.datasets.index import (
    build_finance_pretraining_index,
    build_inference_index,
    build_sample_index,
)
from facdigger.datasets.splits import assign_chronological_splits
from facdigger.experiments.manifest import sha256_json
from facdigger.features.pipeline import (
    apply_feature_scaler,
    build_raw_feature_tables,
    fit_feature_scaler,
)
from facdigger.labels.forward_return import (
    build_forward_excess_return_labels,
    build_multi_horizon_excess_return_labels,
)


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
        # v4 adds the immutable target-free finance pretraining index.  Bumping
        # the identity prevents an existing v3 directory from being mistaken
        # for a snapshot that contains the new artifact.
        "schema_version": 4,
        "config": semantic_config,
        "input_file_hashes": input_hashes,
    }
    dataset_id = sha256_json(identity)
    output_root = config.output_root.resolve()
    final_dir = output_root / dataset_id
    if final_dir.exists():
        manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
        return final_dir, manifest

    output_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_root / f".tmp-{dataset_id[:12]}-{uuid.uuid4().hex}"
    temporary_dir.mkdir(parents=False, exist_ok=False)
    try:
        source_audit = adapter.audit(bundle)
        bars = bundle.bars
        universe = bundle.universe
        delistings = bundle.delistings
        del bundle

        raw_features, raw_market_features = build_raw_feature_tables(
            bars, universe, feature_set=config.features.name
        )
        scaler = fit_feature_scaler(
            raw_features,
            raw_market_features,
            channels=config.features.channels,
            train_end=config.split.train_end,
            winsor_lower=config.features.winsor_lower,
            winsor_upper=config.features.winsor_upper,
        )
        features, market_features = apply_feature_scaler(raw_features, raw_market_features, scaler)
        del raw_features, raw_market_features

        if config.label.auxiliary_horizons:
            labels = build_multi_horizon_excess_return_labels(
                bars,
                universe,
                delistings=delistings,
                execution_lag=config.label.execution_lag,
                horizons=config.label.all_horizons,
                primary_horizon=config.label.horizon,
            )
        else:
            labels = build_forward_excess_return_labels(
                bars,
                universe,
                delistings=delistings,
                execution_lag=config.label.execution_lag,
                horizon=config.label.horizon,
            )
        del bars, delistings
        calendar = universe["trade_date"].unique().sort().to_list()
        labels = assign_chronological_splits(labels, calendar, config.split)
        del calendar

        additional_label_columns = (
            [
                column
                for horizon in config.label.all_horizons
                for column in (
                    f"label_end_{horizon}",
                    f"raw_return_{horizon}",
                    f"benchmark_return_{horizon}",
                    f"target_{horizon}",
                    f"crosses_delisting_{horizon}",
                )
            ]
            if config.label.auxiliary_horizons
            else []
        )
        required_targets = (
            [f"target_{horizon}" for horizon in config.label.all_horizons]
            if config.label.auxiliary_horizons
            else ["target"]
        )
        sample_index = build_sample_index(
            features,
            labels,
            universe,
            context_length=config.features.context_length,
            additional_label_columns=additional_label_columns,
            required_target_columns=required_targets,
        )
        labels_audit = {
            "rows": labels.height,
            "target_non_null": labels["target"].is_not_null().sum(),
            "crosses_delisting": labels["crosses_delisting"].sum(),
            "target_non_null_by_horizon": {
                str(horizon): labels[f"target_{horizon}"].is_not_null().sum()
                for horizon in config.label.all_horizons
                if f"target_{horizon}" in labels.columns
            },
        }
        labels.write_parquet(temporary_dir / "labels.parquet")
        del labels

        sample_metadata = _build_sample_metadata(sample_index, universe)
        sample_metadata.write_parquet(temporary_dir / "sample_metadata.parquet")
        del sample_metadata

        inference_index = build_inference_index(
            features,
            universe,
            context_length=config.features.context_length,
        )
        maximum_inference_date = inference_index["asof_date"].max()
        inference_audit = {
            "rows": inference_index.height,
            "minimum_asof_date": inference_index["asof_date"].min().isoformat(),
            "maximum_asof_date": maximum_inference_date.isoformat(),
            "latest_cross_section_rows": inference_index.filter(
                pl.col("asof_date") == maximum_inference_date
            ).height,
            "contains_target": "target" in inference_index.columns,
        }
        inference_index.write_parquet(temporary_dir / "inference_index.parquet")
        del inference_index

        pretraining_index_audit: dict[str, Any] | None = None
        if config.features.name == "finance_transformer":
            pretraining_index = build_finance_pretraining_index(
                features,
                universe,
                context_length=config.features.context_length,
                future_horizon=5,
                train_end=config.split.train_end,
            )
            pretraining_index_audit = {
                "rows": pretraining_index.height,
                "dates": pretraining_index["asof_date"].n_unique(),
                "minimum_asof_date": pretraining_index["asof_date"].min().isoformat(),
                "maximum_asof_date": pretraining_index["asof_date"].max().isoformat(),
                "maximum_future_end": pretraining_index["future_end"].max().isoformat(),
                "contains_supervised_target": any(
                    column.startswith("target") for column in pretraining_index.columns
                ),
            }
            pretraining_index.write_parquet(temporary_dir / "pretraining_index.parquet")
            del pretraining_index
        del universe

        features_audit = _feature_audit(features)
        features.write_parquet(temporary_dir / "features.parquet")
        del features
        market_features_audit: dict[str, Any] | None = None
        if market_features is not None:
            market_features_audit = _feature_audit(market_features)
            market_features.write_parquet(temporary_dir / "market_features.parquet")
            del market_features

        split_counts = {
            row["split"]: row["len"]
            for row in sample_index.group_by("split").len().sort("split").to_dicts()
        }
        sample_index_audit = {
            "rows": sample_index.height,
            "split_counts": split_counts,
        }
        sample_index.write_parquet(temporary_dir / "sample_index.parquet")
        del sample_index

        audit = {
            "sources": source_audit,
            "features": features_audit,
            "market_features": market_features_audit,
            "labels": labels_audit,
            "sample_index": sample_index_audit,
            "inference_index": inference_audit,
            "pretraining_index": pretraining_index_audit,
        }
        manifest = {
            **identity,
            "dataset_id": dataset_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_paths": source_paths,
            "artifacts": {
                "features": "features.parquet",
                "market_features": (
                    "market_features.parquet" if market_features_audit is not None else None
                ),
                "labels": "labels.parquet",
                "sample_index": "sample_index.parquet",
                "sample_metadata": "sample_metadata.parquet",
                "inference_index": "inference_index.parquet",
                "pretraining_index": (
                    "pretraining_index.parquet" if pretraining_index_audit is not None else None
                ),
                "audit": "audit.json",
                "scaler": "scaler.json",
                "source_manifest": (
                    "source_manifest.json" if config.sources.source_manifest is not None else None
                ),
            },
        }
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
