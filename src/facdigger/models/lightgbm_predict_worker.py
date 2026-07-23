"""Load and predict a LightGBM checkpoint in an isolated process."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM inference requires: uv sync --extra baseline --extra data"
        ) from exc
    model = lgb.Booster(model_file=str(args.checkpoint))
    features = np.load(args.input)
    scores = model.predict(features, num_iteration=model.best_iteration)
    np.save(args.output, np.asarray(scores, dtype=np.float64))


if __name__ == "__main__":
    main()
