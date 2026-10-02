"""Run the approved wf3 historical R experiment in an isolated output directory."""

import argparse
from pathlib import Path

from facdigger.research.finance_history_diagnostics import run_history_diagnostics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--budget-seconds", type=float, default=14400)
    args = parser.parse_args()
    audit = run_history_diagnostics(
        args.matrix,
        args.snapshots,
        args.output,
        device=args.device,
        budget_seconds=args.budget_seconds,
    )
    print(audit["status"], flush=True)
