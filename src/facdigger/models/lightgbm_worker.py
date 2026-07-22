"""Native LightGBM worker kept separate from Polars/PyTorch OpenMP runtimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM baseline requires: uv sync --extra baseline --extra data"
        ) from exc
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    config = payload["model"]
    seed = int(payload["seed"])
    arrays = np.load(args.input)
    train_data = lgb.Dataset(arrays["train_x"], label=arrays["train_y"], free_raw_data=False)
    valid_data = lgb.Dataset(
        arrays["valid_x"], label=arrays["valid_y"], reference=train_data, free_raw_data=False
    )
    model = lgb.train(
        {
            "objective": "huber",
            "metric": "l1",
            "learning_rate": config["learning_rate"],
            "num_leaves": config["num_leaves"],
            "min_child_samples": config["min_child_samples"],
            "lambda_l2": config["reg_lambda"],
            "seed": seed,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
            "num_threads": 1,
        },
        train_data,
        num_boost_round=config["n_estimators"],
        valid_sets=[valid_data],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(config["early_stopping_rounds"], verbose=False)],
    )
    model.save_model(str(args.checkpoint))
    scores = model.predict(arrays["evaluation_x"], num_iteration=model.best_iteration)
    np.save(args.scores, np.asarray(scores, dtype=np.float64))
    args.audit.write_text(
        json.dumps(
            {"best_iteration": model.best_iteration, "best_score": model.best_score},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
