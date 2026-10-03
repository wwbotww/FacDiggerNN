"""Complete F and fixed-date gradient diagnostics from the approved C observations."""

import argparse
from pathlib import Path

from facdigger.research.finance_fixed_diagnostics import run_fixed_diagnostics
from facdigger.training.finance_transformer_config import load_finance_transformer_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("dataset", "checksums", "config", "cache", "source", "output"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, required=True)
    parser.add_argument("--cumulative-budget-seconds", type=float)
    parser.add_argument("--shutdown-margin-seconds", type=float, default=120)
    parser.add_argument("--reuse-linear-chunks", type=Path)
    args = parser.parse_args()
    result = run_fixed_diagnostics(
        args.dataset,
        args.checksums,
        load_finance_transformer_config(args.config),
        args.cache,
        args.source,
        args.output,
        budget_seconds=args.budget_seconds,
        repository_root=Path(__file__).resolve().parents[2],
        cumulative_budget_seconds=args.cumulative_budget_seconds,
        shutdown_margin_seconds=args.shutdown_margin_seconds,
        reuse_linear_chunks=args.reuse_linear_chunks,
    )
    print(result["status"], flush=True)
