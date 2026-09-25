"""Persistent identity and single-writer lifecycle for finance training."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.environment import collect_environment
from facdigger.experiments.manifest import collect_git_state, sha256_json
from facdigger.training.runtime import (
    TrainingControl,
    TrainingPaused,
    atomic_write_text,
    resolve_dataset,
    run_lock,
    write_json,
)


@dataclass
class TrainingRun:
    path: Path
    dataset: Path
    resume: Path | None
    manifest: dict[str, Any]


def verify_completed_artifacts(run_dir: Path, manifest: dict[str, Any]) -> None:
    checkpoint = manifest["checkpoint"]
    for name, digest in (
        (checkpoint["file"], checkpoint["sha256"]),
        (manifest.get("artifacts", {}).get("predictions"), manifest.get("predictions_sha256")),
    ):
        if name is not None:
            path = (run_dir / name).resolve()
            if (
                not path.is_relative_to(run_dir.resolve())
                or not path.is_file()
                or (sha256_file(path) != digest)
            ):
                raise DataContractError(f"completed stage artifact changed: {name}")
    for name in manifest.get("artifacts", {}).values():
        path = (run_dir / name).resolve()
        if not path.is_relative_to(run_dir.resolve()) or not path.is_file():
            raise DataContractError(f"completed stage artifact missing: {name}")


@contextmanager
def training_run(
    config: Any,
    dataset_dir: str | Path,
    *,
    repository_root: str | Path,
    model_type: str,
    control: TrainingControl,
    resume_from: str | Path | None,
    run_dir: str | Path | None,
) -> Iterator[TrainingRun]:
    now = datetime.now(timezone.utc).isoformat()
    payload = config.model_dump(mode="json")
    config_hash = sha256_json(payload)
    resume = Path(resume_from).resolve() if resume_from is not None else None
    if resume is not None:
        if not resume.is_file() or resume.parent.name != "checkpoints":
            raise FileNotFoundError("resume checkpoint must be inside a checkpoints directory")
        root = resume.parent.parent
        if run_dir is not None and Path(run_dir).resolve() != root:
            raise ValueError("resume checkpoint and run directory differ")
    elif run_dir is not None:
        root = Path(run_dir).resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = (
            config.output_root.resolve() / f"{config.experiment_id}-{stamp}-{uuid.uuid4().hex[:8]}"
        )
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root / ".training.lock"):
        manifest_path = root / "manifest.json"
        previous = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else None
        )
        if previous is not None:
            if previous.get("status") == "complete" and resume_from is not None:
                raise ValueError("cannot resume an already-complete training run")
            if previous.get("config_hash") != config_hash:
                raise ValueError("resume run configuration does not match")
            if previous.get("model_type") != model_type:
                raise ValueError("resume run model type does not match")
            if resume is None and previous.get("status") != "complete":
                candidate = root / "checkpoints" / "last.pt"
                if candidate.is_file():
                    resume = candidate
                elif any((root / "checkpoints").glob("*.pt")):
                    raise FileNotFoundError("existing run has weights but no committed last.pt")
        elif resume is not None or (root / "checkpoints").exists():
            raise FileNotFoundError("checkpoint has no bound run manifest")
        dataset = resolve_dataset(
            Path(dataset_dir),
            control.config,
            dataset_id=str(previous["dataset_id"]) if previous else None,
            manifest_hash=previous["dataset_manifest_hash"] if previous else None,
        )
        dataset_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        if previous is not None and previous.get("status") == "complete":
            verify_completed_artifacts(root, previous)
            yield TrainingRun(root, dataset, resume, previous)
            return
        manifest = previous or {
            "run_id": root.name,
            "created_at": now,
            "model_id": config.experiment_id,
            "model_type": model_type,
            "dataset_id": dataset_manifest["dataset_id"],
            "dataset_path": str(dataset),
            "dataset_manifest_hash": sha256_file(dataset / "manifest.json"),
            "config_hash": config_hash,
            "git": collect_git_state(repository_root),
        }
        manifest.update(
            {
                "status": "running",
                "updated_at": now,
                "resumed_from": str(resume) if resume else None,
                "stop_reason": None,
                "error": None,
                "recoverable_checkpoint": None,
            }
        )
        manifest.setdefault("attempts", []).append(
            {
                "started_at": now,
                "dataset_path": str(dataset),
                "runtime": control.config.model_dump(mode="json"),
                "restart_without_checkpoint": previous is not None and resume is None,
                "git": collect_git_state(repository_root),
                "environment": collect_environment(include_model_dependencies=True),
            }
        )
        atomic_write_text(
            root / "resolved_config.yaml",
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=True),
        )
        write_json(manifest_path, manifest)
        try:
            yield TrainingRun(root, dataset, resume, manifest)
        except BaseException as exc:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            checkpoint = root / "checkpoints" / "last.pt"
            paused = isinstance(exc, TrainingPaused)
            current.update(
                {
                    "status": "paused" if paused else "failed",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "stop_reason": exc.reason if paused else None,
                    "error": None if paused else {"type": type(exc).__name__, "message": str(exc)},
                    "recoverable_checkpoint": "checkpoints/last.pt"
                    if checkpoint.is_file()
                    else None,
                }
            )
            write_json(manifest_path, current)
            raise
