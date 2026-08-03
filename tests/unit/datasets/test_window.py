from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from facdigger.datasets.sampler import DateGroupedBatchSampler
from facdigger.datasets.window import (
    SecurityFeatureStore,
    SnapshotInferenceWindowDataset,
    SnapshotWindowDataset,
)


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


def test_inference_dataset_retains_target_free_factor_metadata() -> None:
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(3)]
    features = pl.DataFrame(
        {
            "security_id": ["A"] * 3,
            "trade_date": dates,
            "x": [1.0, 2.0, 3.0],
        }
    )
    inference_index = pl.DataFrame(
        {
            "sample_id": ["A|latest"],
            "security_id": ["A"],
            "symbol": ["A"],
            "asof_date": [dates[2]],
            "feature_start": [dates[1]],
            "feature_end": [dates[2]],
            "eligible": [True],
            "industry_code": ["technology"],
            "float_market_cap": [1_000_000.0],
            "log_float_market_cap": [13.8155],
        }
    )

    dataset = SnapshotInferenceWindowDataset(
        features=features,
        inference_index=inference_index,
        channels=["x"],
        context_length=2,
    )

    assert "target" not in dataset.sample_rows.columns
    assert dataset.sample_rows["eligible"].to_list() == [True]
    assert dataset.sample_rows["industry_code"].to_list() == ["technology"]
    assert dataset.sample_rows["log_float_market_cap"].to_list() == [13.8155]


def test_date_grouped_sampler_is_deterministic_and_never_mixes_dates() -> None:
    dates = ["d1", "d1", "d2", "d2", "d2", "d3"]
    sampler = DateGroupedBatchSampler(dates, batch_size=4, shuffle=True, seed=17)
    sampler.set_epoch(3)
    first = list(sampler)
    second = list(sampler)

    assert first == second
    for indices in first:
        represented = {dates[index] for index in indices}
        assert len(represented) == 1

    restored = DateGroupedBatchSampler(dates, batch_size=4, shuffle=True, seed=17)
    restored.load_state_dict(sampler.state_dict())
    assert list(restored) == first


def test_date_grouped_sampler_balances_large_dates_without_dropping_rows() -> None:
    dates = ["d1"] * 1000 + ["d2"] * 65
    sampler = DateGroupedBatchSampler(
        dates,
        batch_size=64,
        shuffle=False,
        seed=17,
        minimum_group_size=32,
    )
    batches = list(sampler)

    assert sorted(index for batch in batches for index in batch) == list(range(len(dates)))
    assert max(map(len, batches)) <= 64
    assert min(map(len, batches)) >= 32
    assert {len(batch) for batch in batches[:16]} == {62, 63}
    assert [len(batch) for batch in batches[16:]] == [33, 32]
