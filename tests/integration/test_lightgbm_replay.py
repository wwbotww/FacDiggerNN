from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lightgbm")

from facdigger.models.baselines import (  # noqa: E402
    predict_lightgbm_checkpoint,
    train_lightgbm_from_files,
)
from facdigger.training.e0_config import LightGBMBaselineConfig  # noqa: E402
from facdigger.training.ranking import (  # noqa: E402
    contiguous_group_sizes,
    cross_sectional_rank_targets,
    lightgbm_relevance_grades,
)


def test_lightgbm_checkpoint_replay_matches_training_scores(tmp_path: Path) -> None:
    generator = np.random.default_rng(42)
    train_x = generator.normal(size=(40, 4)).astype(np.float32)
    train_y = (train_x[:, 0] - train_x[:, 1]).astype(np.float64)
    valid_x = generator.normal(size=(12, 4)).astype(np.float32)
    valid_y = (valid_x[:, 0] - valid_x[:, 1]).astype(np.float64)
    evaluation_x = generator.normal(size=(10, 4)).astype(np.float32)
    checkpoint = tmp_path / "best.txt"
    config = LightGBMBaselineConfig(
        n_estimators=8,
        learning_rate=0.1,
        num_leaves=4,
        min_child_samples=2,
        early_stopping_rounds=3,
    )
    train_dates = [f"d{index // 10}" for index in range(len(train_y))]
    valid_dates = [f"v{index // 4}" for index in range(len(valid_y))]
    train_target_rank = cross_sectional_rank_targets(
        train_y, train_dates, minimum_cross_section_size=2
    )
    valid_target_rank = cross_sectional_rank_targets(
        valid_y, valid_dates, minimum_cross_section_size=2
    )
    matrices = {
        "train_x": train_x,
        "train_y": lightgbm_relevance_grades(
            train_target_rank, bins=config.relevance_bins
        ),
        "valid_x": valid_x,
        "valid_y": lightgbm_relevance_grades(
            valid_target_rank, bins=config.relevance_bins
        ),
        "evaluation_x": evaluation_x,
        "train_target_rank": train_target_rank,
        "valid_target_rank": valid_target_rank,
        "train_group": contiguous_group_sizes(train_dates),
        "valid_group": contiguous_group_sizes(valid_dates),
    }
    paths: dict[str, Path] = {}
    for name, values in matrices.items():
        path = tmp_path / f"{name}.npy"
        np.save(path, values)
        paths[name] = path

    original, audit = train_lightgbm_from_files(
        **paths,
        config=config,
        seed=42,
        checkpoint_path=checkpoint,
        preprocessing={
            "feature_columns": ["a", "b"],
            "means": [0.0, 0.0],
            "scales": [1.0, 1.0],
        },
    )
    replayed = predict_lightgbm_checkpoint(checkpoint, evaluation_x)

    np.testing.assert_allclose(replayed, original, rtol=0, atol=0)
    assert audit["objective"] == "lambdarank"
    assert audit["checkpoint_metric"] == "mean_daily_spearman_rank_ic"
    assert audit["train_groups"] == 4
    assert audit["valid_groups"] == 3
