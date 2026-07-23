"""Build immutable fold-specific snapshots from one frozen dataset specification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from facdigger.data.config import load_dataset_build_config
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


def validate_model_config_paths(config: M6ResearchConfig) -> dict[str, Path]:
    paths = {
        key: Path(value).resolve()
        for key, value in config.models.model_dump().items()
    }
    missing = [f"{key}:{path}" for key, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("research model configuration missing: " + ", ".join(missing))
    return paths
