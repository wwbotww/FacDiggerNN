"""Run fold-local financial self-supervision and publish one encoder checkpoint."""

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
    FinancePretrainingWindowDataset,
    FinanceTransformerWindowDataset,
    MarketFeatureStore,
    SecurityFeatureStore,
)
from facdigger.environment import collect_environment
from facdigger.experiments.manifest import collect_git_state
from facdigger.training.common import (
    load_required_market_features,
    load_required_snapshot_features,
    load_training_snapshot,
)
from facdigger.training.finance_pretrain_config import (
    FinancePretrainingExperimentConfig,
)
from facdigger.training.finance_pretrain_engine import train_finance_pretraining
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


def _probe_index(
    sample_index: pl.DataFrame,
    *,
    fit_dates: int,
    selection_dates: int,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    official_train = sample_index.filter(pl.col("split") == "train").sort(
        ["asof_date", "security_id"]
    )
    dates = official_train["asof_date"].unique().sort().to_list()
    if len(dates) < fit_dates + selection_dates:
        raise DataContractError(
            "finance pretraining probe has fewer Train dates than its fixed protocol"
        )
    selection = dates[-selection_dates:]
    selection_start = selection[0]
    fit_candidates = official_train.filter(pl.col("label_end") < selection_start)
    available_fit_dates = fit_candidates["asof_date"].unique().sort().to_list()
    if len(available_fit_dates) < fit_dates:
        raise DataContractError("finance pretraining probe purge leaves too few fit dates")
    fit = available_fit_dates[-fit_dates:]
    probe_fit = official_train.filter(pl.col("asof_date").is_in(fit)).with_columns(
        pl.lit("probe_fit").alias("split")
    )
    probe_selection = official_train.filter(pl.col("asof_date").is_in(selection)).with_columns(
        pl.lit("probe_selection").alias("split")
    )
    if probe_fit.is_empty() or probe_selection.is_empty():
        raise DataContractError("finance pretraining probe produced an empty partition")
    if probe_fit["label_end"].max() >= selection_start:
        raise DataContractError("finance pretraining probe labels overlap selection")
    return (
        pl.concat([probe_fit, probe_selection], how="vertical").sort(["asof_date", "security_id"]),
        {
            "policy": "fixed_train_tail_probe_with_label_overlap_purge",
            "fit_dates": fit_dates,
            "selection_dates": selection_dates,
            "fit_rows": probe_fit.height,
            "selection_rows": probe_selection.height,
            "fit_min_asof_date": probe_fit["asof_date"].min(),
            "fit_max_label_end": probe_fit["label_end"].max(),
            "selection_min_asof_date": selection_start,
            "selection_max_asof_date": probe_selection["asof_date"].max(),
            "outer_validation_rows_used": 0,
            "outer_test_rows_used": 0,
        },
    )


def run_finance_pretraining(
    config: FinancePretrainingExperimentConfig,
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
            model_type="finance_patch_pretrain",
            control=session_control,
            resume_from=resume_from,
            run_dir=run_dir,
        ) as run:
            if run.manifest["status"] == "complete":
                return run.path, json.loads(
                    (run.path / "training_audit.json").read_text(encoding="utf-8")
                )
            return _run_finance_pretraining(
                config, run, repository_root=repository_root, control=session_control
            )


def _run_finance_pretraining(
    config: FinancePretrainingExperimentConfig,
    run: TrainingRun,
    *,
    repository_root: str | Path,
    control: TrainingControl,
) -> tuple[Path, dict[str, Any]]:
    dataset_path = run.dataset
    dataset_manifest, frames = load_training_snapshot(dataset_path, include_features=False)
    if int(dataset_manifest.get("schema_version", 0)) < 4:
        raise DataContractError(
            "finance pretraining requires a schema-v4 snapshot; rebuild the snapshot"
        )
    feature_config = dataset_manifest["config"]["features"]
    label_config = dataset_manifest["config"]["label"]
    if feature_config.get("name") != "finance_transformer":
        raise DataContractError("finance pretraining requires finance_transformer features")
    if config.channels != list(feature_config["channels"]):
        raise DataContractError("pretraining local channels differ from snapshot")
    if config.market_channels != list(feature_config["market_channels"]):
        raise DataContractError("pretraining market channels differ from snapshot")
    dataset_horizons = sorted(
        [label_config["horizon"], *label_config.get("auxiliary_horizons", [])]
    )
    if int(label_config["horizon"]) != 5 or dataset_horizons != [1, 5, 20]:
        raise DataContractError("finance pretraining probe requires 1/5/20 labels with primary 5")
    artifact = dataset_manifest.get("artifacts", {}).get("pretraining_index")
    if not isinstance(artifact, str):
        raise DataContractError("snapshot does not contain a pretraining index")
    pretraining_index = pl.read_parquet(dataset_path / artifact)
    if any(column.startswith("target") for column in pretraining_index.columns):
        raise DataContractError("snapshot pretraining index contains supervised targets")
    probe_index, probe_audit = _probe_index(
        frames["sample_index"],
        fit_dates=config.training.probe.fit_dates,
        selection_dates=config.training.probe.selection_dates,
    )
    context_length = int(feature_config["context_length"])
    if config.model.statistics_windows[-1] > context_length:
        raise DataContractError("model statistics window exceeds snapshot context")
    required_rows = pl.concat(
        [
            pretraining_index.select("security_id", "feature_start", "asof_date", "future_end"),
            probe_index.select("security_id", "feature_start", "asof_date").with_columns(
                pl.col("asof_date").alias("future_end")
            ),
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
    pretraining_dataset = FinancePretrainingWindowDataset(
        feature_store=feature_store,
        market_store=market_store,
        pretraining_index=pretraining_index,
        channels=config.channels,
        market_channels=config.market_channels,
        context_length=context_length,
        future_horizon=config.future_horizon,
    )
    horizons = sorted([label_config["horizon"], *label_config.get("auxiliary_horizons", [])])

    def probe_dataset(split: str) -> FinanceTransformerWindowDataset:
        return FinanceTransformerWindowDataset(
            feature_store=feature_store,
            market_store=market_store,
            sample_index=probe_index,
            channels=config.channels,
            market_channels=config.market_channels,
            context_length=context_length,
            split=split,
            horizons=horizons,
            primary_horizon=int(label_config["horizon"]),
        )

    probe_fit_dataset = probe_dataset("probe_fit")
    probe_selection_dataset = probe_dataset("probe_selection")
    run_dir = run.path
    resume_path = run.resume
    manifest_path = run_dir / "manifest.json"
    initial_manifest = {
        **run.manifest,
        "status": "running",
        "run_id": run.manifest["run_id"],
        "created_at": run.manifest["created_at"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": config.experiment_id,
        "model_type": "finance_patch_pretrain",
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_path": str(dataset_path),
        "dataset_manifest_hash": sha256_file(dataset_path / "manifest.json"),
        "config_hash": run.manifest["config_hash"],
        "seed": config.seed,
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "probe_protocol": probe_audit,
    }
    _write_json(manifest_path, initial_manifest)
    _, training_audit = train_finance_pretraining(
        config,
        pretraining_dataset=pretraining_dataset,
        probe_fit_dataset=probe_fit_dataset,
        probe_selection_dataset=probe_selection_dataset,
        dataset_id=str(dataset_manifest["dataset_id"]),
        checkpoint_dir=run_dir / "checkpoints",
        resume_from=resume_path,
        control=control,
        progress_callback=lambda event: _append_progress(run_dir / "progress.jsonl", event),
    )
    control.raise_if_stopping(run_dir / "checkpoints" / "last.pt")
    best_checkpoint = run_dir / "checkpoints" / "best_encoder.pt"
    _write_json(run_dir / "training_audit.json", training_audit)
    manifest = {
        **initial_manifest,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "architecture": config.model.model_dump(mode="json"),
        "objective": config.training.objective.model_dump(mode="json"),
        "input": {
            "feature_set": feature_config["name"],
            "context_length": context_length,
            "channels": config.channels,
            "market_channels": config.market_channels,
            "future_horizon": config.future_horizon,
            "supervised_target_in_pretraining_index": False,
            "outer_validation_rows_used": 0,
            "outer_test_rows_used": 0,
        },
        "row_counts": {
            "pretraining": len(pretraining_dataset),
            "probe_fit": len(probe_fit_dataset),
            "probe_selection": len(probe_selection_dataset),
        },
        "checkpoint": {
            "file": "checkpoints/best_encoder.pt",
            "sha256": sha256_file(best_checkpoint),
            "last_file": "checkpoints/last.pt",
            "last_sha256": sha256_file(run_dir / "checkpoints" / "last.pt"),
        },
        "training": training_audit,
        "git": collect_git_state(repository_root),
        "environment": collect_environment(include_model_dependencies=True),
        "artifacts": {
            "training_audit": "training_audit.json",
            "resolved_config": "resolved_config.yaml",
            "progress": "progress.jsonl",
        },
    }
    control.raise_if_stopping(run_dir / "checkpoints" / "last.pt")
    _write_json(manifest_path, manifest)
    return run_dir, training_audit
