"""Bounded retention for production-only inference snapshots."""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime
from pathlib import Path


def prune_inference_snapshots(
    output_root: str | Path, *, keep_sessions: int, keep_attempts_per_session: int = 2,
) -> list[Path]:
    if keep_sessions < 1:
        raise ValueError("keep_sessions must be positive")
    if keep_attempts_per_session < 1:
        raise ValueError("keep_attempts_per_session must be positive")
    root = Path(output_root).resolve()
    if not root.exists():
        return []
    dated: list[tuple[date, Path]] = []
    for child in root.iterdir():
        if not child.is_dir() or child.is_symlink():
            continue
        try:
            day = date.fromisoformat(child.name)
        except ValueError:
            continue
        dated.append((day, child))
    removed: list[Path] = []
    for _, path in sorted(dated)[:-keep_sessions]:
        path.relative_to(root)
        shutil.rmtree(path)
        removed.append(path)
    for day, directory in sorted(dated)[-keep_sessions:]:
        attempts: list[tuple[datetime, Path]] = []
        for child in directory.iterdir():
            if not child.is_dir() or child.is_symlink():
                continue
            try:
                manifest = json.loads((child / "manifest.json").read_text(encoding="utf-8"))
                if not isinstance(manifest, dict) or not isinstance(manifest.get("config"), dict):
                    continue
                if (
                    manifest.get("contract") != "facdigger.inference_snapshot"
                    or manifest.get("status") != "complete"
                    or manifest.get("snapshot_id") != child.name
                    or manifest.get("config", {}).get("asof_date") != day.isoformat()
                ):
                    continue
                created = datetime.fromisoformat(manifest["created_at"])
                if created.tzinfo is None:
                    continue
            except (OSError, ValueError, KeyError, TypeError):
                continue
            attempts.append((created, child))
        for _, path in sorted(attempts)[:-keep_attempts_per_session]:
            shutil.rmtree(path)
            removed.append(path)
    return removed
