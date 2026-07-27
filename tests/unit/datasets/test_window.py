from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from facdigger.datasets.sampler import DateGroupedBatchSampler
from facdigger.datasets.window import SecurityFeatureStore, SnapshotWindowDataset


def test_window_dataset_respects_snapshot_bounds_and_missing_mask() -> None:
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(6)]
    features = pl.DataFrame(
        {
            "security_id": ["A"] * 6,
            "trade_date": dates,
            "x": [1.0, 2.0, float("nan"), 4.0, 999.0, 999.0],
            "observed_x": [True, True, True, True, True, True],
        }
    )
    sample_index = pl.DataFrame(
        {
            "sample_id": ["A|3"],
            "security_id": ["A"],
            "symbol": ["A"],
            "asof_date": [dates[3]],
            "feature_start": [dates[1]],
            "split": ["train"],
            "target": [0.25],
        }
    )
    dataset = SnapshotWindowDataset(
        features=features,
        sample_index=sample_index,
        channels=["x"],
        context_length=3,
        split="train",
    )

    sample = dataset[0]
    np.testing.assert_array_equal(sample["values"][:, 0], [2.0, 0.0, 4.0])
    np.testing.assert_array_equal(sample["observed_mask"][:, 0], [True, False, True])
    assert sample["values"].shape == (3, 1)
    assert sample["target"] == np.float32(0.25)


def test_split_datasets_share_one_feature_store() -> None:
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(5)]
    features = pl.DataFrame(
        {
            "security_id": ["A"] * 5,
            "trade_date": dates,
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    sample_index = pl.DataFrame(
        {
            "sample_id": ["A|train", "A|valid"],
            "security_id": ["A", "A"],
            "symbol": ["A", "A"],
            "asof_date": [dates[2], dates[4]],
            "feature_start": [dates[1], dates[3]],
            "split": ["train", "valid"],
            "target": [0.1, 0.2],
        }
    )
    store = SecurityFeatureStore(features=features, channels=["x"])
    train = SnapshotWindowDataset(
        feature_store=store,
        sample_index=sample_index,
        channels=["x"],
        context_length=2,
        split="train",
    )
    valid = SnapshotWindowDataset(
        feature_store=store,
        sample_index=sample_index,
        channels=["x"],
        context_length=2,
        split="valid",
    )

    assert train.feature_store is store
    assert valid.feature_store is store
    assert train.blocks is valid.blocks
    np.testing.assert_array_equal(train[0]["values"][:, 0], [2.0, 3.0])
    np.testing.assert_array_equal(valid[0]["values"][:, 0], [4.0, 5.0])


def test_date_grouped_sampler_is_deterministic_and_keeps_small_dates_whole() -> None:
    dates = ["d1", "d1", "d2", "d2", "d2", "d3"]
    sampler = DateGroupedBatchSampler(dates, batch_size=4, shuffle=True, seed=17)
    sampler.set_epoch(3)
    first = list(sampler)
    second = list(sampler)

    assert first == second
    for indices in first:
        represented = {dates[index] for index in indices}
        for value in represented:
            full_group = {index for index, date_value in enumerate(dates) if date_value == value}
            assert full_group.issubset(indices)

    restored = DateGroupedBatchSampler(dates, batch_size=4, shuffle=True, seed=17)
    restored.load_state_dict(sampler.state_dict())
    assert list(restored) == first
