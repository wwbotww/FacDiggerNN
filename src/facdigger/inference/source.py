"""Read and verify portable source run artifacts without importing inference orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from facdigger.data.contracts import DataContractError
from facdigger.data.paths import artifact_path
from facdigger.data.snapshots import sha256_file
from facdigger.experiments.manifest import sha256_json
from facdigger.inference.backends import RELEASABLE_MODEL_TYPES

SUPPORTED_MODEL_TYPES = {*RELEASABLE_MODEL_TYPES, "mlp", "lightgbm"}


def resolve_training_snapshot(
    manifest: dict[str, Any], dataset_dir: str | Path | None = None,
) -> Path:
    recorded = manifest.get("dataset_path")
    locator = dataset_dir if dataset_dir is not None else recorded
    if locator is None:
        raise FileNotFoundError("source training snapshot location is missing; specify --dataset")
    path = Path(locator).resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"source training snapshot is unavailable at {locator}; "
            "specify --dataset <local-snapshot-directory>. "
            "The original manifest must not be rewritten."
        )
    return path


def _load_source_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    root = run_dir.resolve()
    manifest_path = root / "manifest.json"
    config_path = root / "resolved_config.yaml"
    if not manifest_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"run is missing manifest or resolved config: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status", "complete") != "complete":
        raise ValueError("inference requires a complete source run")
    model_type = str(manifest.get("model_type"))
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"unsupported source run model_type: {model_type}")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise ValueError("source resolved configuration must be a mapping")
    if sha256_json(config_payload) != manifest["config_hash"]:
        raise DataContractError("source resolved configuration hash does not match manifest")
    checkpoint_info = manifest.get("checkpoint") or {}
    checkpoint_path = artifact_path(root, str(checkpoint_info["file"]), "checkpoint")
    if sha256_file(checkpoint_path) != checkpoint_info["sha256"]:
        raise DataContractError("source checkpoint hash does not match manifest")
    return manifest, config_payload, checkpoint_path
