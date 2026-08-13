"""Bounded retention for production-only inference snapshots."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path


def prune_inference_snapshots(output_root: str | Path, *, keep_sessions: int) -> list[Path]:
    if keep_sessions < 1:
        raise ValueError("keep_sessions must be positive")
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
    return removed
