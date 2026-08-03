"""Native LightGBM worker kept separate from Polars/PyTorch OpenMP runtimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from facdigger.training.ranking import mean_grouped_spearman


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-x", type=Path, required=True)
    parser.add_argument("--train-y", type=Path, required=True)
    parser.add_argument("--valid-x", type=Path, required=True)
    parser.add_argument("--valid-y", type=Path, required=True)
    parser.add_argument("--train-target-rank", type=Path, required=True)
    parser.add_argument("--valid-target-rank", type=Path, required=True)
    parser.add_argument("--train-group", type=Path, required=True)
    parser.add_argument("--valid-group", type=Path, required=True)
    parser.add_argument("--evaluation-x", type=Path, required=True)
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
    train_x = np.load(args.train_x, mmap_mode="r")
    train_y = np.load(args.train_y, mmap_mode="r")
    valid_x = np.load(args.valid_x, mmap_mode="r")
    valid_y = np.load(args.valid_y, mmap_mode="r")
    train_target_rank = np.load(args.train_target_rank, mmap_mode="r")
    valid_target_rank = np.load(args.valid_target_rank, mmap_mode="r")
    train_group = np.load(args.train_group, mmap_mode="r")
    valid_group = np.load(args.valid_group, mmap_mode="r")
    if int(train_group.sum()) != len(train_y) or int(valid_group.sum()) != len(valid_y):
        raise ValueError("LightGBM group sizes do not match label rows")
    train_data = lgb.Dataset(
        train_x, label=train_y, group=train_group, free_raw_data=True
    )
    valid_data = lgb.Dataset(
        valid_x,
        label=valid_y,
        group=valid_group,
        reference=train_data,
        free_raw_data=True,
    )

    def rank_ic_metric(predictions: np.ndarray, dataset: object) -> tuple[str, float, bool]:
        if dataset is valid_data:
            targets = valid_target_rank
            groups = valid_group
        elif dataset is train_data:
            targets = train_target_rank
            groups = train_group
        else:
            raise ValueError("unknown LightGBM dataset in Rank IC evaluator")
        return (
            "mean_rank_ic",
            mean_grouped_spearman(predictions, targets, groups),
            True,
        )

    model = lgb.train(
        {
            "objective": "lambdarank",
            "metric": "None",
            "label_gain": list(range(config["relevance_bins"])),
            "lambdarank_truncation_level": config["truncation_level"],
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
        feval=rank_ic_metric,
        callbacks=[
            lgb.early_stopping(
                config["early_stopping_rounds"],
                first_metric_only=True,
                verbose=False,
            )
        ],
    )
    del train_x, train_y, valid_x, valid_y
    model.save_model(str(args.checkpoint))
    evaluation_x = np.load(args.evaluation_x, mmap_mode="r")
    scores = model.predict(evaluation_x, num_iteration=model.best_iteration)
    np.save(args.scores, np.asarray(scores, dtype=np.float64))
    args.audit.write_text(
        json.dumps(
            {
                "objective": "lambdarank",
                "target_transform": (
                    "full_date_average_rank_percentile_minus_one_to_one"
                ),
                "checkpoint_metric": "mean_daily_spearman_rank_ic",
                "best_iteration": model.best_iteration,
                "best_score": model.best_score,
                "train_groups": len(train_group),
                "valid_groups": len(valid_group),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
