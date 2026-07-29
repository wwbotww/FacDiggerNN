from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from facdigger.models.baselines import TabularPreprocessor, build_multiscale_features
from facdigger.training.e0_config import E0ExperimentConfig


def test_test_split_requires_explicit_unlock() -> None:
    try:
        E0ExperimentConfig(evaluation_split="test")
    except ValueError as exc:
        assert "unlock_test=true" in str(exc)
    else:
        raise AssertionError("test split was not gated")


def test_multiscale_features_use_only_past_and_append_missing_masks() -> None:
    dates = [date(2025, 1, 1) + timedelta(days=index) for index in range(8)]
    features = pl.DataFrame(
        {
            "security_id": ["sec-a"] * 8,
            "trade_date": dates,
            "x": [float(index) for index in range(8)],
        }
    )
    samples = pl.DataFrame(
        {
            "sample_id": ["sec-a|2025-01-05"],
            "security_id": ["sec-a"],
            "symbol": ["A"],
            "asof_date": [dates[4]],
            "split": ["train"],
            "target": [0.1],
        }
    )
    original, columns = build_multiscale_features(
        features, samples, channels=["x"], windows=[3], context_length=5
    )
    mutated = features.with_columns(
        pl.when(pl.col("trade_date") > dates[4])
        .then(pl.lit(10_000.0))
        .otherwise(pl.col("x"))
        .alias("x")
    )
    changed, _ = build_multiscale_features(
        mutated, samples, channels=["x"], windows=[3], context_length=5
    )
    assert original.select(columns).to_dicts() == changed.select(columns).to_dicts()
    assert original["x__mean_3"][0] == 3.0
    assert all(original.schema[column] == pl.Float32 for column in columns)

    with_missing = original.with_columns(pl.lit(None).cast(pl.Float64).alias(columns[0]))
    preprocessor = TabularPreprocessor.fit(original, columns)
    matrix = preprocessor.transform(with_missing)
    assert matrix.shape == (1, len(columns) * 2)
    assert matrix[0, 0] == 0.0
    assert matrix[0, len(columns)] == 0.0

    restored = TabularPreprocessor.from_dict(preprocessor.to_dict())
    assert restored == preprocessor


def test_preprocessor_writes_float32_file_backed_matrix(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "x": [1.0, 2.0, None],
            "y": [5.0, 5.0, 5.0],
        }
    )
    preprocessor = TabularPreprocessor.fit(frame, ["x", "y"])

    path = preprocessor.transform_to_npy(frame, tmp_path / "matrix.npy")
    matrix = np.load(path, mmap_mode="r")

    assert isinstance(matrix, np.memmap)
    assert matrix.dtype == np.float32
    assert matrix.shape == (3, 4)
    np.testing.assert_array_equal(matrix[:, 2], [1.0, 1.0, 0.0])
