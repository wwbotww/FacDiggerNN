"""Complete Finance F/S experiments. No command evaluates V or unlocks holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from facdigger.research.finance_experiment_cache import prepare_experiment_cache
from facdigger.research.finance_experiment_config import (
    experiment_cells,
    load_finance_experiment_config,
)
from facdigger.research.finance_experiment_plan import (
    experiment_status,
    freeze_experiment_plan,
)
from facdigger.research.finance_experiment_runner import (
    run_experiment_cell,
    seal_experiment_selections,
)
from facdigger.training.runtime import TrainingRuntimeConfig, load_training_runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    listing = commands.add_parser("list", help="List declared cells without reading snapshots")
    listing.add_argument("--config", type=Path, required=True)
    planning = commands.add_parser("plan", help="Freeze the matrix and verify all F/S inputs")
    planning.add_argument("--config", type=Path, required=True)
    planning.add_argument("--runtime", type=Path, required=True)
    planning.add_argument("--output", type=Path, required=True)
    for name in ("prepare", "run"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--runtime", type=Path)
        command.add_argument("--budget-seconds", type=float)
        command.add_argument("--shutdown-margin-seconds", type=float)
        if name == "prepare":
            command.add_argument("--fold", required=True)
        else:
            command.add_argument("--cell", required=True)
            command.add_argument("--cache", type=Path, help="Read-only existing F/S cache root")
            command.add_argument("--parent-prefix", type=Path)
            command.add_argument("--cumulative-budget-seconds", type=float)
    for name in ("status", "seal"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    if args.action == "list":
        result = {"cells": experiment_cells(load_finance_experiment_config(args.config))}
    elif args.action == "plan":
        result = freeze_experiment_plan(
            load_finance_experiment_config(args.config), load_training_runtime(args.runtime),
            args.output, repository_root=repository_root,
        )
        result = {"plan_hash": result["plan_hash"], "cells": result["identity"]["cells"]}
    elif args.action == "status":
        result = experiment_status(args.output)
    elif args.action == "seal":
        result = seal_experiment_selections(args.output, repository_root=repository_root)
    else:
        values = load_training_runtime(args.runtime).model_dump(mode="json")
        for option, field in (
            ("budget_seconds", "max_walltime_seconds"),
            ("shutdown_margin_seconds", "shutdown_margin_seconds"),
        ):
            value = getattr(args, option)
            if value is not None:
                values[field] = value
        runtime = TrainingRuntimeConfig.model_validate(values)
        if args.action == "prepare":
            result = prepare_experiment_cache(
                args.output, args.fold, repository_root=repository_root, runtime=runtime,
            )
        else:
            audit = run_experiment_cell(
                args.output, args.cell, repository_root=repository_root, runtime=runtime,
                cache=args.cache, parent_prefix=args.parent_prefix,
                cumulative_budget_seconds=args.cumulative_budget_seconds,
            )
            result = {
                key: audit[key] for key in (
                    "status", "phase", "epoch", "global_step", "best_epoch", "stop_reason",
                ) if key in audit
            }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 75 if result.get("status") == "paused" else 0


if __name__ == "__main__":
    raise SystemExit(main())
