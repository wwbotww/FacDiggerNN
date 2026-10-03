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
    parser.add_argument("--seed", type=int, help="Explicit seed override, bound to the run config")
    parser.add_argument(
        "--statistics-dropout", type=float,
        help="MLP diagnostic dropout only; default 0.1, independent of Finance model.dropout",
    )
    parser.add_argument("--observation-scope", choices=("all", "fit-selection"), default="all")
    parser.add_argument("--observation-precision", choices=("fp32", "fp16"))
    parser.add_argument("--full-fit", action="store_true", help="Observe all F dates at epochs 1/2")
    parser.add_argument("--shutdown-margin-seconds", type=float, default=180)
    parser.add_argument("--cumulative-budget-seconds", type=float)
    args = parser.parse_args()
    if args.statistics_dropout is not None and (
        args.action != "run" or args.candidate != "statistics_mlp"
    ):
        parser.error("--statistics-dropout requires run --candidate statistics_mlp")
    config = FinanceTransformerExperimentConfig.model_validate(
        yaml.safe_load(args.config.read_text())
    )
    if args.seed is not None:
        config = FinanceTransformerExperimentConfig.model_validate(
            {**config.model_dump(mode="json"), "seed": args.seed}
        )
    if args.action == "prepare":
        if args.observation_scope != "all" or args.observation_precision or args.full_fit:
            parser.error(
                "observation options require run; prepare creates the original F/S/V cache"
            )
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
            observation_scope=args.observation_scope,
            observation_precision=args.observation_precision,
            full_fit=args.full_fit,
            shutdown_margin_seconds=args.shutdown_margin_seconds,
            cumulative_budget_seconds=args.cumulative_budget_seconds,
            statistics_dropout=(
                0.1 if args.statistics_dropout is None else args.statistics_dropout
            ),
        )
        print(result["status"], flush=True)
