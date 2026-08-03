"""Build immutable fold-specific snapshots from one frozen dataset specification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from facdigger.data.config import SplitConfig, load_dataset_build_config
from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import build_dataset_snapshot, sha256_file
from facdigger.experiments.manifest import sha256_json
from facdigger.research.config import M6ResearchConfig


def build_walk_forward_snapshots(config: M6ResearchConfig) -> list[dict[str, Any]]:
    base = load_dataset_build_config(config.base_dataset_config)
    plans: list[dict[str, Any]] = []
    for fold in config.folds:
        split = fold.model_dump(exclude={"fold_id"})
        fold_config = base.model_copy(
            update={
                "output_root": config.snapshot_output_root,
                "split": base.split.model_validate(split),
            }
        )
        snapshot_dir, manifest = build_dataset_snapshot(fold_config)
        plans.append(
            {
                "fold_id": fold.fold_id,
                "split": split,
                "dataset_id": manifest["dataset_id"],
                "dataset_path": str(snapshot_dir.resolve()),
                "dataset_manifest_sha256": sha256_file(snapshot_dir / "manifest.json"),
                "protocol_hash": sha256_json(
                    {
                        "base_dataset": base.model_dump(mode="json"),
                        "fold": fold.model_dump(mode="json"),
                    }
                ),
            }
        )
    return plans


def final_refit_protocol(config: M6ResearchConfig) -> dict[str, Any]:
    """Describe the frozen post-selection refit without reading the holdout."""

    base = load_dataset_build_config(config.base_dataset_config)
    final_fold = config.folds[-1]
    split = SplitConfig(
        train_end=final_fold.valid_end,
        valid_end=final_fold.valid_end,
        test_end=final_fold.test_end,
        embargo_sessions=final_fold.embargo_sessions,
    )
    payload = {
        "purpose": "post_validation_full_history_refit",
        "source_fold_id": final_fold.fold_id,
        "training_data_end": final_fold.valid_end.isoformat(),
        "holdout_end": final_fold.test_end.isoformat(),
        "outer_validation_is_empty": True,
        "split": split.model_dump(mode="json"),
        "training_policy": (
            "reuse frozen model config; refit scaler and model on the expanded official "
            "train split; retain train-internal checkpoint selection"
        ),
    }
    return {
        **payload,
        "protocol_hash": sha256_json(
            {
                "base_dataset": base.model_dump(mode="json"),
                "refit": payload,
            }
        ),
    }


def build_final_refit_snapshot(
    config: M6ResearchConfig,
    validation_final_fold: dict[str, Any],
) -> dict[str, Any]:
    """Build a snapshot whose official train ends at the frozen validation boundary."""

    protocol = final_refit_protocol(config)
    if validation_final_fold["fold_id"] != protocol["source_fold_id"]:
        raise DataContractError("final validation fold does not match refit protocol")
    validation_path = Path(validation_final_fold["dataset_path"])
    validation_manifest_path = validation_path / "manifest.json"
    if not validation_manifest_path.is_file():
        raise FileNotFoundError(
            f"final validation snapshot manifest missing: {validation_manifest_path}"
        )
    validation_manifest = json.loads(
        validation_manifest_path.read_text(encoding="utf-8")
    )
    if validation_manifest.get("dataset_id") != validation_final_fold["dataset_id"]:
        raise DataContractError("final validation snapshot dataset_id changed")
    if sha256_file(validation_manifest_path) != validation_final_fold[
        "dataset_manifest_sha256"
    ]:
        raise DataContractError("final validation snapshot manifest changed")

    base = load_dataset_build_config(config.base_dataset_config)
    refit_config = base.model_copy(
        update={
            "output_root": config.snapshot_output_root,
            "split": SplitConfig.model_validate(protocol["split"]),
        }
    )
    snapshot_dir, manifest = build_dataset_snapshot(refit_config)
    if manifest.get("input_file_hashes") != validation_manifest.get("input_file_hashes"):
        raise DataContractError(
            "refit snapshot source hashes differ from frozen validation snapshot"
        )
    audit = json.loads((snapshot_dir / "audit.json").read_text(encoding="utf-8"))
    split_counts = audit["sample_index"]["split_counts"]
    if split_counts.get("valid", 0) != 0:
        raise DataContractError("final refit snapshot unexpectedly contains validation rows")
    if split_counts.get("train", 0) <= 0 or split_counts.get("test", 0) <= 0:
        raise DataContractError("final refit snapshot requires non-empty train and test splits")
    key_columns = ["security_id", "asof_date", "target"]
    validation_test_keys = (
        pl.scan_parquet(validation_path / "sample_index.parquet")
        .filter(pl.col("split") == "test")
        .select(key_columns)
        .sort(["asof_date", "security_id"])
        .collect()
    )
    refit_test_keys = (
        pl.scan_parquet(snapshot_dir / "sample_index.parquet")
        .filter(pl.col("split") == "test")
        .select(key_columns)
        .sort(["asof_date", "security_id"])
        .collect()
    )
    if not validation_test_keys.equals(refit_test_keys, null_equal=True):
        raise DataContractError(
            "final refit snapshot changed the frozen holdout prediction keys"
        )
    return {
        "fold_id": protocol["source_fold_id"],
        "purpose": protocol["purpose"],
        "split": protocol["split"],
        "training_data_end": protocol["training_data_end"],
        "outer_validation_is_empty": True,
        "dataset_id": manifest["dataset_id"],
        "dataset_path": str(snapshot_dir.resolve()),
        "dataset_manifest_sha256": sha256_file(snapshot_dir / "manifest.json"),
        "protocol_hash": protocol["protocol_hash"],
        "source_validation_dataset_id": validation_final_fold["dataset_id"],
        "split_counts": split_counts,
        "holdout_rows": refit_test_keys.height,
    }


def validate_model_config_paths(config: M6ResearchConfig) -> dict[str, Path]:
    paths = {
        key: Path(value).resolve()
        for key, value in config.models.model_dump().items()
    }
    missing = [f"{key}:{path}" for key, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("research model configuration missing: " + ", ".join(missing))
    return paths
