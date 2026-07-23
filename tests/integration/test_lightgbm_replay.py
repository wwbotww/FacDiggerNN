from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lightgbm")

from facdigger.models.baselines import (  # noqa: E402
    predict_lightgbm_checkpoint,
    train_lightgbm,
)
from facdigger.training.e0_config import LightGBMBaselineConfig  # noqa: E402


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

    original, _ = train_lightgbm(
        train_x,
        train_y,
        valid_x,
        valid_y,
        evaluation_x,
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
