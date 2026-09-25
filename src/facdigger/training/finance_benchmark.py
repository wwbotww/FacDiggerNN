"""RTX admission benchmark for the fixed nine-stage Transformer matrix."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from facdigger.data.contracts import DataContractError
from facdigger.datasets.window import (
    FinancePretrainingWindowDataset,
    FinanceTransformerWindowDataset,
    MarketFeatureStore,
    SecurityFeatureStore,
)
from facdigger.experiments.manifest import sha256_json
from facdigger.training.common import (
    load_required_market_features,
    load_required_snapshot_features,
    load_training_snapshot,
    split_supervised_training_index,
)
from facdigger.training.finance_pretrain_config import (
    FinancePretrainingExperimentConfig,
)
from facdigger.training.finance_pretrain_engine import (
    benchmark_finance_pretraining_updates,
)
from facdigger.training.finance_transformer_config import (
    FinanceTransformerExperimentConfig,
)
from facdigger.training.finance_transformer_engine import (
    benchmark_finance_transformer_updates,
)
from facdigger.training.resources import (
    TrainingResourceBudget,
    cgroup_memory_limit,
    effective_resource_limits,
    training_hardware,
)

RTX_2070S_RESERVED_MEMORY_LIMIT_BYTES = int(7.2 * 1024**3)
HOST_MEMORY_LIMIT_BYTES = 13 * 1024**3


def _process_peak_rss_bytes() -> int | None:
    """Return the current process high-water RSS on Unix/WSL."""

    try:
        import resource
    except ImportError:  # pragma: no cover - native Windows is not the target runtime
        return None
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def run_finance_training_benchmark(
    supervised_config: FinanceTransformerExperimentConfig,
    pretraining_config: FinancePretrainingExperimentConfig,
    dataset_dir: str | Path,
    *,
    optimizer_updates: int = 100,
    resource_budget: TrainingResourceBudget | None = None,
) -> dict[str, Any]:
    """Benchmark one largest fold and conservatively project all nine stages."""

    if supervised_config.initialization != "scratch":
        raise ValueError("admission benchmark requires the scratch supervised config")
    dataset_path = Path(dataset_dir).resolve()
    manifest, frames = load_training_snapshot(dataset_path, include_features=False)
    if int(manifest.get("schema_version", 0)) < 4:
        raise DataContractError("finance benchmark requires a schema-v4 snapshot")
    feature_config = manifest["config"]["features"]
    label_config = manifest["config"]["label"]
    if feature_config.get("name") != "finance_transformer":
        raise DataContractError("finance benchmark requires finance_transformer features")
    if supervised_config.channels != pretraining_config.channels or (
        supervised_config.market_channels != pretraining_config.market_channels
    ):
        raise DataContractError("supervised and pretraining benchmark channels differ")
    if supervised_config.channels != list(feature_config["channels"]) or (
        supervised_config.market_channels != list(feature_config["market_channels"])
    ):
        raise DataContractError("benchmark configuration channels differ from snapshot")
    pretraining_artifact = manifest.get("artifacts", {}).get("pretraining_index")
    if not isinstance(pretraining_artifact, str):
        raise DataContractError("snapshot has no finance pretraining index")
    pretraining_index = pl.read_parquet(dataset_path / pretraining_artifact)
    protocol_index, _ = split_supervised_training_index(
        frames["sample_index"],
        selection_fraction=supervised_config.selection_fraction,
    )
    train_rows = protocol_index.filter(pl.col("split") == "train_fit")
    required_rows = pl.concat(
        [
            pretraining_index.select("security_id", "feature_start", "asof_date", "future_end"),
            train_rows.select("security_id", "feature_start", "asof_date").with_columns(
                pl.col("asof_date").alias("future_end")
            ),
        ],
        how="vertical",
    )
    feature_store = SecurityFeatureStore(
        features=load_required_snapshot_features(dataset_path, manifest, required_rows),
        channels=supervised_config.channels,
        presorted=True,
    )
    market_store = MarketFeatureStore(
        features=load_required_market_features(dataset_path, manifest, required_rows),
        channels=supervised_config.market_channels,
    )
    context_length = int(feature_config["context_length"])
    supervised_dataset = FinanceTransformerWindowDataset(
        feature_store=feature_store,
        market_store=market_store,
        sample_index=protocol_index,
        channels=supervised_config.channels,
        market_channels=supervised_config.market_channels,
        context_length=context_length,
        split="train_fit",
        horizons=supervised_config.horizons,
        primary_horizon=supervised_config.primary_horizon,
    )
    pretraining_dataset = FinancePretrainingWindowDataset(
        feature_store=feature_store,
        market_store=market_store,
        pretraining_index=pretraining_index,
        channels=pretraining_config.channels,
        market_channels=pretraining_config.market_channels,
        context_length=context_length,
        future_horizon=pretraining_config.future_horizon,
    )
    supervised = benchmark_finance_transformer_updates(
        supervised_config,
        train_dataset=supervised_dataset,
        dataset_id=str(manifest["dataset_id"]),
        optimizer_updates=optimizer_updates,
        warmup_updates=min(10, optimizer_updates - 1),
    )
    pretraining = benchmark_finance_pretraining_updates(
        pretraining_config,
        pretraining_dataset=pretraining_dataset,
        dataset_id=str(manifest["dataset_id"]),
        local_optimizer_updates=optimizer_updates,
        market_optimizer_updates=min(20, optimizer_updates),
        warmup_updates=min(10, optimizer_updates - 1),
    )
    raw_hours = 6.0 * float(supervised["projected_cell_hours"]) + 3.0 * float(
        pretraining["projected_run_hours"]
    )
    projected_hours = raw_hours * 1.1
    projected_days = projected_hours / 24.0
    host_peak_rss_bytes = _process_peak_rss_bytes()
    cuda_peak_reserved_bytes = max(
        int(supervised["cuda_peak_reserved_bytes"] or 0),
        int(pretraining["cuda_peak_reserved_bytes"] or 0),
    )
    device_gate = (
        supervised["device"] == "cuda"
        and supervised["precision"] == "fp16"
        and pretraining["device"] == "cuda"
        and pretraining["precision"] == "fp16"
    )
    budget = resource_budget or TrainingResourceBudget()
    hardware = training_hardware()
    limits = (
        effective_resource_limits(budget, hardware)
        if resource_budget
        else {
            "cuda_peak_reserved_bytes": RTX_2070S_RESERVED_MEMORY_LIMIT_BYTES,
            "host_peak_rss_bytes": HOST_MEMORY_LIMIT_BYTES,
            "projected_days": 14.0,
        }
    )
    memory_gate = (
        host_peak_rss_bytes is not None
        and host_peak_rss_bytes <= limits["host_peak_rss_bytes"]
        and cuda_peak_reserved_bytes <= limits["cuda_peak_reserved_bytes"]
    )
    time_gate = projected_days <= limits["projected_days"]
    return {
        **(
            {
                "resource_budget": budget.model_dump(mode="json"),
                "hardware": hardware,
                "cgroup_memory_limit_bytes": cgroup_memory_limit(),
                "job_time_admission": "requires_loading_probe_selection_checkpoint_measurement",
            }
            if resource_budget is not None
            else {}
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": manifest["dataset_id"],
        "dataset_path": str(dataset_path),
        "context_length": context_length,
        "supervised_train_rows": len(supervised_dataset),
        "pretraining_rows": len(pretraining_dataset),
        "benchmark_optimizer_updates": optimizer_updates,
        "supervised_config_hash": sha256_json(supervised_config.model_dump(mode="json")),
        "pretraining_config_hash": sha256_json(pretraining_config.model_dump(mode="json")),
        "host_peak_rss_bytes": host_peak_rss_bytes,
        "supervised": supervised,
        "pretraining": pretraining,
        "matrix_projection": {
            "pretraining_runs": 3,
            "supervised_cells": 6,
            "raw_hours": raw_hours,
            "overhead_fraction": 0.1,
            "projected_hours": projected_hours,
            "projected_days": projected_days,
            "probe_time_not_measured": True,
            "conservative_largest_fold_applied_to_all_stages": True,
        },
        "admission": {
            "cuda_fp16_verified": device_gate,
            "within_memory_budget": memory_gate,
            "within_fourteen_days": projected_days <= 14.0,
            "within_time_budget": time_gate,
            "admitted": device_gate and memory_gate and time_gate,
            "limits": limits,
            "rule": (
                "CUDA FP16, explicit memory and matrix compute-time budgets; "
                "single-job lifecycle time requires a separate interruption rehearsal"
                if resource_budget
                else "CUDA FP16 must be active, peak memory must fit RTX 2070S/16 GB RAM, "
                "and the largest-fold conservative projection must not exceed 14 days"
            ),
        },
        "quality_preservation": {
            "snapshot_rows_changed": False,
            "context_changed": False,
            "model_changed": False,
            "epoch_budget_changed": False,
            "benchmark_only_stops_early": True,
        },
        "label_protocol": label_config,
    }


def write_finance_training_benchmark(path: str | Path, report: dict[str, Any]) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return destination
