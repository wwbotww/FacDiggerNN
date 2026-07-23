"""Read-only readiness gate before the expensive frozen research matrix."""

from __future__ import annotations

import json
from typing import Any

import polars as pl

from facdigger.data.config import load_dataset_build_config
from facdigger.data.snapshots import sha256_file
from facdigger.research.config import M6ResearchConfig
from facdigger.research.folds import validate_model_config_paths


def research_preflight(config: M6ResearchConfig) -> dict[str, Any]:
    """Check frozen inputs and provenance without building datasets or training."""

    model_paths = validate_model_config_paths(config)
    base = load_dataset_build_config(config.base_dataset_config)
    source_paths = {
        "bars": base.sources.bars,
        "universe": base.sources.universe,
        "corporate_actions": base.sources.corporate_actions,
        "delistings": base.sources.delistings,
        "source_manifest": base.sources.source_manifest,
    }
    missing_sources = [
        name for name, path in source_paths.items() if path is not None and not path.is_file()
    ]
    provenance: dict[str, Any] = {
        "available": False,
        "research_ready": None,
        "warnings": [],
    }
    if base.sources.source_manifest is not None and base.sources.source_manifest.is_file():
        payload = json.loads(base.sources.source_manifest.read_text(encoding="utf-8"))
        selection = payload.get("selection") or {}
        provenance = {
            "available": True,
            "provider": payload.get("provider"),
            "source_revision": payload.get("source_revision"),
            "manifest_sha256": sha256_file(base.sources.source_manifest),
            "selection": selection,
            "delistings": payload.get("delistings"),
            "research_ready": selection.get("research_ready"),
            "warnings": list(payload.get("warnings") or []),
        }
    maximum_date = None
    minimum_date = None
    universe_rows = None
    if base.sources.universe.is_file():
        dates = pl.scan_parquet(base.sources.universe).select("trade_date").collect()
        minimum_date = dates["trade_date"].min()
        maximum_date = dates["trade_date"].max()
        universe_rows = dates.height
    required_end = max(fold.test_end for fold in config.folds)
    require_source_ready = config.decisions.require_source_research_ready
    checks = {
        "source_files_exist": not missing_sources,
        "source_provenance_available": provenance["available"],
        "source_readiness_gate": (
            provenance["research_ready"] is True or not require_source_ready
        ),
        "universe_covers_final_fold": maximum_date is not None and maximum_date >= required_end,
        "model_configs_exist": len(model_paths) == 4,
        "three_or_more_folds": len(config.folds) >= 3,
        "three_or_more_seeds": len(config.seeds) >= 3,
    }
    blockers: list[str] = []
    if missing_sources:
        blockers.append(f"missing configured sources: {missing_sources}")
    if not provenance["available"]:
        blockers.append("source provenance manifest is unavailable")
    elif provenance["research_ready"] is not True and require_source_ready:
        blockers.append("source provenance explicitly does not declare research_ready=true")
    if not checks["universe_covers_final_fold"]:
        blockers.append(
            f"universe maximum date {maximum_date} does not cover final fold end {required_end}"
        )
    return {
        "ready": all(checks.values()),
        "research_mode": (
            "formal"
            if config.decisions.require_source_research_ready
            and config.decisions.require_neutralized_positive
            else "engineering"
        ),
        "decision_gates": config.decisions.model_dump(mode="json"),
        "research_id": config.research_id,
        "checks": checks,
        "blockers": blockers,
        "source_paths": {
            name: str(path) if path is not None else None
            for name, path in source_paths.items()
        },
        "source_provenance": provenance,
        "universe": {
            "rows": universe_rows,
            "minimum_date": minimum_date,
            "maximum_date": maximum_date,
            "required_final_fold_end": required_end,
        },
        "model_configs": {name: str(path) for name, path in model_paths.items()},
        "validation_cells": len(config.folds) * len(config.seeds) * 4,
        "final_holdout_cells": len(config.seeds) * 4,
    }
