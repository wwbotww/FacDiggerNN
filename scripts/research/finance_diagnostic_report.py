"""CPU-only summary of saved C evidence. Does not score or train models."""

import argparse
from pathlib import Path

from facdigger.research.finance_diagnostic_report import summarize_prefix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(summarize_prefix(args.runs, args.output)["status"])
