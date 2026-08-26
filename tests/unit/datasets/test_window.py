from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from facdigger.data.contracts import DataContractError
from facdigger.datasets.sampler import (
    DateGroupedBatchSampler,
    DateSecurityBalancedBatchSampler,
    FullDateBatchSampler,
)
from facdigger.datasets.window import (
    FinancePretrainingWindowDataset,
    FinanceTransformerWindowDataset,
    SecurityFeatureStore,
    SnapshotInferenceWindowDataset,
    SnapshotWindowDataset,
)


def test_finance_pretraining_dataset_is_target_free_and_returns_future_windows() -> None:
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(8)]
    features = pl.DataFrame(
        {
            "security_id": ["A"] * 8,
            "trade_date": dates,
            "x": [float(index) for index in range(8)],
            "observed_x": [True] * 8,
        }
    )
    market = pl.DataFrame(
        {
            "trade_date": dates,
            "m": [10.0 + index for index in range(8)],
            "observed_m": [True] * 8,
        }
    )
    index = pl.DataFrame(
        {
            "sample_id": ["A|d"],
            "security_id": ["A"],
            "symbol": ["A"],
            "asof_date": [dates[4]],
            "feature_start": [dates[2]],
            "feature_end": [dates[4]],
            "future_start": [dates[5]],
            "future_end": [dates[7]],
        }
    )
    dataset = FinancePretrainingWindowDataset(
        features=features,
        market_features=market,
        pretraining_index=index,
        channels=["x"],
        market_channels=["m"],
        context_length=3,
        future_horizon=3,
        future_local_channels=1,
    )

    sample = dataset[0]
    assert "target" not in sample
    np.testing.assert_array_equal(sample["values"][:, 0], [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(sample["future_values"][:, 0], [5.0, 6.0, 7.0])
    history, future = dataset.market_pair(dates[4])
    np.testing.assert_array_equal(history.values[:, 0], [12.0, 13.0, 14.0])
    np.testing.assert_array_equal(future.values[:, 0], [15.0, 16.0, 17.0])


def test_date_security_balanced_sampler_uses_every_row_once() -> None:
    sampler = DateSecurityBalancedBatchSampler(
        ["d1", "d1", "d1", "d2", "d2", "d3"],
        batch_size=4,
        seed=17,
    )
    sampler.set_epoch(2)
    batches = list(sampler)

    assert sorted(index for batch in batches for index in batch) == list(range(6))
    assert [len(batch) for batch in batches] == [4, 2]
    assert list(sampler) == batches


def test_finance_transformer_dataset_returns_multi_targets_and_shared_market_window() -> None:
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(5)]
    features = pl.DataFrame(
        {
            "security_id": [security for security in ("A", "B") for _ in dates],
            "trade_date": dates * 2,
            "x": [float(index) for index in range(5)] * 2,
            "observed_x": [True] * 10,
        }
    )
    market_features = pl.DataFrame(
        {
            "trade_date": dates,
            "market_x": [10.0 + index for index in range(5)],
            "observed_market_x": [True] * 5,
        }
    )
    sample_index = pl.DataFrame(
        {
            "sample_id": ["A|d", "B|d"],
            "security_id": ["A", "B"],
            "symbol": ["A", "B"],
            "asof_date": [dates[4], dates[4]],
            "feature_start": [dates[2], dates[2]],
            "feature_end": [dates[4], dates[4]],
            "split": ["train", "train"],
            "target": [0.2, -0.2],
            "target_1": [0.1, -0.1],
            "target_5": [0.2, -0.2],
            "target_20": [0.3, -0.3],
        }
    )
    dataset = FinanceTransformerWindowDataset(
        features=features,
        market_features=market_features,
        sample_index=sample_index,
        channels=["x"],
        market_channels=["market_x"],
        context_length=3,
        split="train",
        horizons=[1, 5, 20],
        primary_horizon=5,
    )

    np.testing.assert_allclose(dataset[0]["targets"], [0.1, 0.2, 0.3])
    window = dataset.market_window_for_sample_indices([0, 1])
    np.testing.assert_array_equal(window.values[:, 0], [12.0, 13.0, 14.0])
    np.testing.assert_array_equal(window.observed_mask[:, 0], [True, True, True])

    second_date = sample_index.with_columns(
        pl.when(pl.col("security_id") == "B")
        .then(pl.lit(dates[3]))
        .otherwise(pl.col("asof_date"))
        .alias("asof_date"),
        pl.when(pl.col("security_id") == "B")
        .then(pl.lit(dates[1]))
        .otherwise(pl.col("feature_start"))
        .alias("feature_start"),
        pl.when(pl.col("security_id") == "B")
        .then(pl.lit(dates[3]))
        .otherwise(pl.col("feature_end"))
        .alias("feature_end"),
    )
    mixed = FinanceTransformerWindowDataset(
        features=features,
        market_features=market_features,
        sample_index=second_date,
        channels=["x"],
        market_channels=["market_x"],
        context_length=3,
        split="train",
        horizons=[1, 5, 20],
        primary_horizon=5,
    )
    with pytest.raises(DataContractError, match="one complete date"):
        mixed.market_window_for_sample_indices([0, 1])


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


def test_full_date_sampler_never_splits_date_at_device_microbatch_boundary() -> None:
    dates = ["d1"] * 7 + ["d2"] * 3 + ["d3"] * 11
    sampler = FullDateBatchSampler(
        dates,
        shuffle=True,
        seed=29,
        minimum_group_size=3,
    )
    sampler.set_epoch(4)
    assert sampler.groups == [(0, 7), (7, 10), (10, 21)]
    first = list(sampler)
    second = list(sampler)

    assert first == second
    assert sorted(index for batch in first for index in batch) == list(range(len(dates)))
    assert sorted(map(len, first)) == [3, 7, 11]
    assert all(len({dates[index] for index in batch}) == 1 for batch in first)

    restored = FullDateBatchSampler(
        dates,
        shuffle=True,
        seed=29,
        minimum_group_size=3,
    )
    restored.load_state_dict(sampler.state_dict())
    assert list(restored) == first


def test_full_date_sampler_rejects_noncontiguous_dates() -> None:
    with pytest.raises(ValueError, match="contiguous date groups"):
        FullDateBatchSampler(["d1", "d2", "d1"], shuffle=False, seed=0)
