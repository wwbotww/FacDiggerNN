"""Prepare deterministic statistics or run one approved two-epoch C candidate."""

import argparse
from pathlib import Path

import yaml

from facdigger.research.finance_prefix_diagnostics import (
    prepare_prefix_statistics,
    run_prefix_diagnostics,
)
from facdigger.training.finance_transformer_config import FinanceTransformerExperimentConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checksums", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--candidate", choices=("finance", "statistics_linear", "statistics_mlp"))
    parser.add_argument("--budget-seconds", type=float, required=True)
    args = parser.parse_args()
    config = FinanceTransformerExperimentConfig.model_validate(
        yaml.safe_load(args.config.read_text())
    )
    if args.action == "prepare":
        prepare_prefix_statistics(
            args.dataset, args.checksums, config, args.output, budget_seconds=args.budget_seconds
        )
    else:
        if args.cache is None or args.candidate is None:
            parser.error("run requires --cache and --candidate")
        result = run_prefix_diagnostics(
            args.dataset,
            args.checksums,
            config,
            args.cache,
            args.output,
            candidate=args.candidate,
            budget_seconds=args.budget_seconds,
            repository_root=Path(__file__).resolve().parents[2],
        )
        print(result["status"], flush=True)
