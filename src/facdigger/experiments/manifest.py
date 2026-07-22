"""Create a self-contained, machine-readable run manifest."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from facdigger.config import ProjectConfig
from facdigger.environment import collect_environment


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def collect_git_state(cwd: str | Path) -> dict[str, Any]:
    root = Path(cwd).resolve()
    commit = _run_git(["rev-parse", "HEAD"], root)
    status = _run_git(["status", "--porcelain"], root)
    branch = _run_git(["branch", "--show-current"], root)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
        "status_porcelain": status,
    }


def create_run_manifest(
    config: ProjectConfig,
    output_root: str | Path,
    repository_root: str | Path,
    command: str,
    now: datetime | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Create an immutable run directory with resolved config and manifest files."""

    created_at = now or datetime.now(timezone.utc)
    config_payload = config.model_dump(mode="json")
    config_hash = sha256_json(config_payload)
    run_id = f"{created_at.strftime('%Y%m%dT%H%M%SZ')}-{config_hash[:10]}"
    run_dir = Path(output_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "command": command,
        "config_hash": config_hash,
        "git": collect_git_state(repository_root),
        "environment": collect_environment(include_model_dependencies=True),
        "dataset_id": None,
        "source_checkpoint": config.model.source.model_dump(mode="json"),
        "seed": config.seed,
    }

    resolved_config_path = run_dir / "resolved_config.yaml"
    manifest_path = run_dir / "manifest.json"
    resolved_config_path.write_text(
        yaml.safe_dump(config_payload, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir, manifest
