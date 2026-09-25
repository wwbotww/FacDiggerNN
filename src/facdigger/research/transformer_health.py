"""Separate health review for frozen Transformer results; never selects checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.experiments.manifest import sha256_json
from facdigger.research.transformer_config import load_transformer_comparison_config
from facdigger.research.transformer_runner import _validate_completed_stage
from facdigger.training.run_state import verify_completed_artifacts
from facdigger.training.runtime import run_lock


def training_health(training: dict[str, Any]) -> dict[str, Any]:
    """Weight clipping by successful updates, not by equally weighted epoch means."""
    history = training.get("history", [])
    incomplete = {"status": "insufficient_history", "checks": {}}
    if not history:
        return incomplete
    try:
        updates = []
        clipped = []
        previous_step = 0
        for epoch, row in enumerate(history, start=1):
            step = int(row["global_step"])
            ratio = float(row["gradient_clip_ratio"])
            scores = [float(row[f"score_std_{h}"]) for h in (1, 5, 20)]
            if (
                row["epoch"] != epoch
                or step <= previous_step
                or not (math.isfinite(ratio) and 0 <= ratio <= 1)
                or any(not math.isfinite(value) or value < 0 for value in scores)
            ):
                return incomplete
            steps = step - previous_step
            clipped_count = round(ratio * steps)
            if not math.isclose(ratio * steps, clipped_count, rel_tol=1e-9, abs_tol=1e-7):
                return incomplete
            updates.append(steps)
            clipped.append(clipped_count)
            previous_step = step
    except (KeyError, TypeError, ValueError, OverflowError):
        return incomplete
    start = len(history) // 2
    latter = history[start:]
    count = sum(updates[start:])
    clipped_count = sum(clipped[start:])
    ratio = clipped_count / count
    traces = {str(h): [float(row[f"score_std_{h}"]) for row in latter] for h in (1, 5, 20)}
    # Three observations give at least two consecutive increases. Fewer data
    # points are insufficient evidence, rather than an automatic healthy result.
    monotonic = {
        horizon: all(right > left for left, right in zip(values, values[1:], strict=False))
        if len(values) >= 3
        else None
        for horizon, values in traces.items()
    }
    checks = {
        "latter_half_clip_ratio_below_10_percent": clipped_count * 10 < count,
        "primary_score_std_not_monotonically_increasing": (
            not monotonic["5"] if monotonic["5"] is not None else None
        ),
    }
    return {
        "status": "failed"
        if False in checks.values()
        else ("insufficient_history" if None in checks.values() else "passed"),
        "checks": checks,
        "epochs": [row["epoch"] for row in latter],
        "successful_optimizer_updates": count,
        "clipped_optimizer_updates": clipped_count,
        "gradient_clip_ratio": ratio,
        "score_std": traces,
        "score_std_strictly_increasing": monotonic,
        "amp_skipped_optimizer_steps": sum(
            int(row.get("amp_skipped_optimizer_steps", 0)) for row in latter
        ),
    }


def audit_transformer_health(run_dir: Path) -> dict[str, Any]:
    """Verify a complete matrix and report health without rewriting frozen acceptance."""
    root = run_dir.resolve()
    with run_lock(root / ".research.lock"):
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise DataContractError("health audit requires a complete comparison")
        config = load_transformer_comparison_config(root / "resolved_config.yaml")
        if sha256_json(config.model_dump(mode="json")) != manifest["config_hash"]:
            raise DataContractError("comparison configuration changed")
        if sha256_file(root / "comparison.json") != manifest["comparison_sha256"]:
            raise DataContractError("completed comparison changed")
        comparison = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
        records = json.loads((root / "matrix.json").read_text(encoding="utf-8"))["stages"]
        expected = {
            (fold.fold_id, stage)
            for fold in config.folds
            for stage in ("pretraining", "scratch", "finance_pretrained")
        }
        if len(records) != 9 or {(r["fold_id"], r["stage"]) for r in records} != expected:
            raise DataContractError("health audit requires exactly the frozen nine stages")
        cells = []
        for record in records:
            if record["status"] != "complete":
                raise DataContractError("health audit found an incomplete stage")
            child = _validate_completed_stage(record)
            payload = json.loads((child / "manifest.json").read_text(encoding="utf-8"))
            verify_completed_artifacts(child, payload)
            if record["stage"] != "pretraining":
                cells.append(
                    {
                        "fold_id": record["fold_id"],
                        "method": record["stage"],
                        "manifest_sha256": record["manifest_sha256"],
                        **training_health(payload.get("training", {})),
                    }
                )
        statuses = {cell["status"] for cell in cells}
        return {
            "run_dir": str(root),
            "comparison_sha256": manifest["comparison_sha256"],
            "metric_acceptance": comparison["acceptance"],
            "health_checks_status": "failed"
            if "failed" in statuses
            else ("insufficient_history" if "insufficient_history" in statuses else "passed"),
            "cells": cells,
            "requires_manual_review": True,
            "review_notes": [
                "latter half uses the last ceil(completed epochs / 2) epochs",
                "clipping is weighted by successful updates; AMP skipped updates are separate",
                "primary 5-day score std needs at least three latter-half epoch observations",
                "review paired score scale and gradient stability, including auxiliary heads",
                "automated checks do not grant research readiness, holdout access or promotion",
            ],
        }
