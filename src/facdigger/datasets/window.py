"""Lazy fixed-length windows backed by an immutable dataset snapshot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from facdigger.data.contracts import DataContractError


@dataclass(frozen=True)
class SecurityFeatureBlock:
    dates: np.ndarray
    values: np.ndarray
    observed: np.ndarray

    def position(self, trade_date: Any) -> int | None:
        target = np.datetime64(trade_date, "D")
        index = int(np.searchsorted(self.dates, target))
        if index >= len(self.dates) or self.dates[index] != target:
            return None
        return index


class SecurityFeatureStore:
    """One immutable feature representation shared by multiple split views."""

    def __init__(
        self,
        *,
        features: pl.DataFrame,
        channels: list[str],
        presorted: bool = False,
    ) -> None:
        missing = [channel for channel in channels if channel not in features.columns]
        if missing:
            raise DataContractError(f"features missing requested channels: {missing}")
        self.channels = tuple(channels)
        self.blocks: dict[str, SecurityFeatureBlock] = {}
        self._security_ids: list[str] = []
        self._block_index_by_security: dict[str, int] = {}

        ordered = features if presorted else features.sort(["security_id", "trade_date"])
        observed_columns = [f"observed_{channel}" for channel in channels]
        has_observed_columns = all(column in ordered.columns for column in observed_columns)
        for _, block in ordered.group_by("security_id", maintain_order=True):
            security_id = str(block["security_id"][0])
            raw_values = block.select(channels).to_numpy().astype(np.float32)
            finite = np.isfinite(raw_values)
            if has_observed_columns:
                source_observed = block.select(observed_columns).to_numpy().astype(bool)
                observed = finite & source_observed
            else:
                observed = finite
            values = np.where(observed, raw_values, 0.0).astype(np.float32)
            dates = np.asarray(block["trade_date"].to_numpy(), dtype="datetime64[D]")
            self._block_index_by_security[security_id] = len(self._security_ids)
            self._security_ids.append(security_id)
            self.blocks[security_id] = SecurityFeatureBlock(
                dates=dates,
                values=values,
                observed=observed,
            )

    def locate(self, security_id: str, trade_date: Any) -> tuple[int, int] | None:
        block_index = self._block_index_by_security.get(security_id)
        if block_index is None:
            return None
        block = self.blocks[self._security_ids[block_index]]
        position = block.position(trade_date)
        if position is None:
            return None
        return block_index, position

    def block_at(self, index: int) -> SecurityFeatureBlock:
        return self.blocks[self._security_ids[index]]


@dataclass(frozen=True)
class MarketFeatureWindow:
    values: np.ndarray
    observed_mask: np.ndarray


class MarketFeatureStore:
    """One date-indexed market sequence shared by every security on that date."""

    def __init__(self, *, features: pl.DataFrame, channels: list[str]) -> None:
        missing = [channel for channel in channels if channel not in features.columns]
        if missing:
            raise DataContractError(f"market features missing requested channels: {missing}")
        ordered = features.sort("trade_date")
        if ordered["trade_date"].n_unique() != ordered.height:
            raise DataContractError("market features contain duplicate trade dates")
        observed_columns = [f"observed_{channel}" for channel in channels]
        if not all(column in ordered.columns for column in observed_columns):
            raise DataContractError("market features require explicit observed masks")
        raw_values = ordered.select(channels).to_numpy().astype(np.float32)
        source_observed = ordered.select(observed_columns).to_numpy().astype(bool)
        observed = np.isfinite(raw_values) & source_observed
        self.channels = tuple(channels)
        self.dates = np.asarray(ordered["trade_date"].to_numpy(), dtype="datetime64[D]")
        self.values = np.where(observed, raw_values, 0.0).astype(np.float32)
        self.observed = observed

    def window(self, asof_date: Any, *, context_length: int) -> MarketFeatureWindow:
        target = np.datetime64(asof_date, "D")
        end = int(np.searchsorted(self.dates, target))
        if end >= len(self.dates) or self.dates[end] != target:
            raise DataContractError(f"market context has no row for {asof_date}")
        start = end - context_length + 1
        if start < 0:
            raise DataContractError(
                f"market context is shorter than {context_length} sessions at {asof_date}"
            )
        return MarketFeatureWindow(
            values=self.values[start : end + 1],
            observed_mask=self.observed[start : end + 1],
        )

    def future_window(
        self, asof_date: Any, *, future_horizon: int
    ) -> MarketFeatureWindow:
        if future_horizon < 1:
            raise ValueError("future_horizon must be positive")
        target = np.datetime64(asof_date, "D")
        end = int(np.searchsorted(self.dates, target))
        if end >= len(self.dates) or self.dates[end] != target:
            raise DataContractError(f"market context has no row for {asof_date}")
        start = end + 1
        stop = start + future_horizon
        if stop > len(self.dates):
            raise DataContractError(
                f"market context has no {future_horizon}-session future at {asof_date}"
            )
        return MarketFeatureWindow(
            values=self.values[start:stop],
            observed_mask=self.observed[start:stop],
        )


class SnapshotWindowDataset:
    """Materialize one `[context_length, channels]` window per indexed sample."""

    def __init__(
        self,
        *,
        features: pl.DataFrame | None = None,
        feature_store: SecurityFeatureStore | None = None,
        sample_index: pl.DataFrame,
        channels: list[str],
        context_length: int,
        split: str,
        retained_columns: list[str] | None = None,
    ) -> None:
        if context_length < 1:
            raise ValueError("context_length must be positive")
        if feature_store is None:
            if features is None:
                raise ValueError("features or feature_store is required")
            feature_store = SecurityFeatureStore(features=features, channels=channels)
        elif features is not None:
            raise ValueError("features and feature_store are mutually exclusive")
        if tuple(channels) != feature_store.channels:
            raise DataContractError(
                f"window channels must exactly match feature store channels: "
                f"{list(feature_store.channels)}"
            )
        self.channels = list(channels)
        self.context_length = context_length
        self.split = split
        self.feature_store = feature_store
        self.blocks = feature_store.blocks
        selected_columns = (
            retained_columns
            if retained_columns is not None
            else [
                "sample_id",
                "security_id",
                "symbol",
                "asof_date",
                "feature_start",
                "feature_end",
                "split",
                "target",
            ]
        )
        retained_columns = [
            column
            for column in selected_columns
            if column in sample_index.columns
        ]
        self.sample_rows = (
            sample_index.filter(pl.col("split") == split)
            .select(retained_columns)
            .sort(["asof_date", "security_id"])
        )
        if self.sample_rows.is_empty():
            raise DataContractError(f"sample_index has no rows for split={split!r}")
        self._block_indices = np.empty(self.sample_rows.height, dtype=np.int32)
        self._starts = np.empty(self.sample_rows.height, dtype=np.int32)
        for index, row in enumerate(self.sample_rows.iter_rows(named=True)):
            security_id = str(row["security_id"])
            location = feature_store.locate(security_id, row["asof_date"])
            if location is None:
                raise DataContractError(f"sample has no matching feature row: {row['sample_id']}")
            block_index, end = location
            block = feature_store.block_at(block_index)
            start = end - context_length + 1
            if "feature_end" in row and row["feature_end"] != row["asof_date"]:
                raise DataContractError(
                    f"sample feature_end must equal asof_date: {row['sample_id']}"
                )
            feature_start = np.datetime64(row["feature_start"], "D")
            if start < 0 or block.dates[start] != feature_start:
                raise DataContractError(
                    f"sample window bounds disagree with snapshot index: {row['sample_id']}"
                )
            self._block_indices[index] = block_index
            self._starts[index] = start

    @classmethod
    def from_snapshot(
        cls,
        dataset_dir: str | Path,
        *,
        split: str,
        channels: list[str] | None = None,
    ) -> SnapshotWindowDataset:
        root = Path(dataset_dir)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        feature_config = manifest["config"]["features"]
        configured_channels = list(feature_config["channels"])
        selected_channels = channels or configured_channels
        if selected_channels != configured_channels:
            raise DataContractError(
                f"window channels must exactly match snapshot channels: {configured_channels}"
            )
        return cls(
            features=pl.read_parquet(root / "features.parquet"),
            sample_index=pl.read_parquet(root / "sample_index.parquet"),
            channels=selected_channels,
            context_length=int(feature_config["context_length"]),
            split=split,
        )

    @property
    def asof_dates(self) -> list[Any]:
        return self.sample_rows["asof_date"].to_list()

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = int(self._starts[index])
        block = self.feature_store.block_at(int(self._block_indices[index]))
        stop = start + self.context_length
        return {
            "values": block.values[start:stop],
            "observed_mask": block.observed[start:stop],
            "target": np.float32(self.sample_rows["target"][index]),
            "sample_index": index,
        }


class SnapshotInferenceWindowDataset(SnapshotWindowDataset):
    """Materialize target-free windows selected from a schema-v3 inference index."""

    def __init__(
        self,
        *,
        features: pl.DataFrame,
        inference_index: pl.DataFrame,
        channels: list[str],
        context_length: int,
    ) -> None:
        required = {
            "sample_id",
            "security_id",
            "asof_date",
            "feature_start",
            "feature_end",
        }
        missing = sorted(required - set(inference_index.columns))
        if missing:
            raise DataContractError(f"inference_index missing required columns: {missing}")
        if "target" in inference_index.columns:
            raise DataContractError("inference_index must not contain target")
        rows = inference_index.with_columns(pl.lit("inference").alias("split"))
        super().__init__(
            features=features,
            sample_index=rows,
            channels=channels,
            context_length=context_length,
            split="inference",
            retained_columns=[
                "sample_id",
                "security_id",
                "symbol",
                "asof_date",
                "feature_start",
                "feature_end",
                "split",
                "eligible",
                "industry_code",
                "float_market_cap",
                "log_float_market_cap",
            ],
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = int(self._starts[index])
        block = self.feature_store.block_at(int(self._block_indices[index]))
        stop = start + self.context_length
        return {
            "values": block.values[start:stop],
            "observed_mask": block.observed[start:stop],
            "sample_index": index,
        }


class FinanceTransformerWindowDataset(SnapshotWindowDataset):
    """Supervised local windows plus one shared market lookup per complete date."""

    def __init__(
        self,
        *,
        features: pl.DataFrame | None = None,
        feature_store: SecurityFeatureStore | None = None,
        market_features: pl.DataFrame | None = None,
        market_store: MarketFeatureStore | None = None,
        sample_index: pl.DataFrame,
        channels: list[str],
        market_channels: list[str],
        context_length: int,
        split: str,
        horizons: list[int],
        primary_horizon: int,
    ) -> None:
        ordered_horizons = sorted(set(horizons))
        if ordered_horizons != horizons:
            raise ValueError("finance transformer horizons must be sorted and unique")
        if primary_horizon not in ordered_horizons:
            raise ValueError("primary_horizon must be included in horizons")
        target_columns = [f"target_{horizon}" for horizon in ordered_horizons]
        missing_targets = sorted(set(target_columns) - set(sample_index.columns))
        if missing_targets:
            raise DataContractError(
                f"sample_index missing multi-horizon targets: {missing_targets}"
            )
        super().__init__(
            features=features,
            feature_store=feature_store,
            sample_index=sample_index,
            channels=channels,
            context_length=context_length,
            split=split,
            retained_columns=[
                "sample_id",
                "security_id",
                "symbol",
                "asof_date",
                "feature_start",
                "feature_end",
                "split",
                "target",
                *target_columns,
            ],
        )
        self.horizons = tuple(ordered_horizons)
        self.primary_horizon = primary_horizon
        self.target_columns = tuple(target_columns)
        if market_store is None:
            if market_features is None:
                raise ValueError("market_features or market_store is required")
            market_store = MarketFeatureStore(
                features=market_features,
                channels=market_channels,
            )
        elif market_features is not None:
            raise ValueError("market_features and market_store are mutually exclusive")
        if tuple(market_channels) != market_store.channels:
            raise DataContractError(
                "market channels must exactly match the shared market feature store"
            )
        self.market_store = market_store
        self.market_channels = tuple(market_channels)

    @classmethod
    def from_snapshot(
        cls,
        dataset_dir: str | Path,
        *,
        split: str,
    ) -> FinanceTransformerWindowDataset:
        root = Path(dataset_dir)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        feature_config = manifest["config"]["features"]
        if feature_config["name"] != "finance_transformer":
            raise DataContractError("snapshot is not a finance_transformer feature set")
        market_artifact = manifest["artifacts"].get("market_features")
        if not isinstance(market_artifact, str):
            raise DataContractError("finance transformer snapshot has no market features")
        label_config = manifest["config"]["label"]
        horizons = sorted(
            [label_config["horizon"], *label_config.get("auxiliary_horizons", [])]
        )
        return cls(
            features=pl.read_parquet(root / manifest["artifacts"]["features"]),
            market_features=pl.read_parquet(root / market_artifact),
            sample_index=pl.read_parquet(root / manifest["artifacts"]["sample_index"]),
            channels=list(feature_config["channels"]),
            market_channels=list(feature_config["market_channels"]),
            context_length=int(feature_config["context_length"]),
            split=split,
            horizons=horizons,
            primary_horizon=int(label_config["horizon"]),
        )

    def market_window_for_sample_indices(
        self, sample_indices: np.ndarray | list[int]
    ) -> MarketFeatureWindow:
        indices = np.asarray(sample_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) < 1:
            raise ValueError("sample_indices must be a non-empty one-dimensional array")
        dates = self.sample_rows[indices.tolist(), "asof_date"].unique().to_list()
        if len(dates) != 1:
            raise DataContractError("market context lookup requires one complete date")
        return self.market_store.window(dates[0], context_length=self.context_length)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        sample["targets"] = np.asarray(
            [self.sample_rows[column][index] for column in self.target_columns],
            dtype=np.float32,
        )
        return sample


class FinancePretrainingWindowDataset(SnapshotWindowDataset):
    """Target-free local histories with self-supervised future feature windows."""

    def __init__(
        self,
        *,
        features: pl.DataFrame | None = None,
        feature_store: SecurityFeatureStore | None = None,
        market_features: pl.DataFrame | None = None,
        market_store: MarketFeatureStore | None = None,
        pretraining_index: pl.DataFrame,
        channels: list[str],
        market_channels: list[str],
        context_length: int,
        future_horizon: int,
        future_local_channels: int = 7,
    ) -> None:
        if future_horizon < 1:
            raise ValueError("future_horizon must be positive")
        if not 1 <= future_local_channels <= len(channels):
            raise ValueError("future_local_channels must fit inside local channels")
        missing = sorted(
            {"future_start", "future_end"} - set(pretraining_index.columns)
        )
        if missing:
            raise DataContractError(
                f"pretraining index missing future bounds: {missing}"
            )
        if any(column.startswith("target") for column in pretraining_index.columns):
            raise DataContractError("pretraining index must not contain supervised targets")
        indexed = pretraining_index.with_columns(
            pl.lit("finance_pretrain").alias("split")
        )
        super().__init__(
            features=features,
            feature_store=feature_store,
            sample_index=indexed,
            channels=channels,
            context_length=context_length,
            split="finance_pretrain",
            retained_columns=[
                "sample_id",
                "security_id",
                "symbol",
                "asof_date",
                "feature_start",
                "feature_end",
                "future_start",
                "future_end",
                "split",
            ],
        )
        self.future_horizon = future_horizon
        self.future_local_channels = future_local_channels
        self._future_starts = np.empty(len(self), dtype=np.int32)
        for index, row in enumerate(self.sample_rows.iter_rows(named=True)):
            block = self.feature_store.block_at(int(self._block_indices[index]))
            location = block.position(row["future_start"])
            if location is None:
                raise DataContractError(
                    f"pretraining sample has no future_start: {row['sample_id']}"
                )
            stop = location + future_horizon
            if stop > len(block.dates) or block.dates[stop - 1] != np.datetime64(
                row["future_end"], "D"
            ):
                raise DataContractError(
                    "pretraining future bounds disagree with feature grid: "
                    f"{row['sample_id']}"
                )
            self._future_starts[index] = location

        if market_store is None:
            if market_features is None:
                raise ValueError("market_features or market_store is required")
            market_store = MarketFeatureStore(
                features=market_features,
                channels=market_channels,
            )
        elif market_features is not None:
            raise ValueError("market_features and market_store are mutually exclusive")
        if tuple(market_channels) != market_store.channels:
            raise DataContractError(
                "market channels must exactly match the shared market feature store"
            )
        self.market_store = market_store
        self.market_channels = tuple(market_channels)

    @property
    def unique_asof_dates(self) -> list[Any]:
        return self.sample_rows["asof_date"].unique(maintain_order=True).to_list()

    def market_pair(
        self, asof_date: Any
    ) -> tuple[MarketFeatureWindow, MarketFeatureWindow]:
        return (
            self.market_store.window(asof_date, context_length=self.context_length),
            self.market_store.future_window(
                asof_date, future_horizon=self.future_horizon
            ),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = int(self._starts[index])
        block = self.feature_store.block_at(int(self._block_indices[index]))
        stop = start + self.context_length
        future_start = int(self._future_starts[index])
        future_stop = future_start + self.future_horizon
        return {
            "values": block.values[start:stop],
            "observed_mask": block.observed[start:stop],
            "future_values": block.values[
                future_start:future_stop, : self.future_local_channels
            ],
            "future_observed_mask": block.observed[
                future_start:future_stop, : self.future_local_channels
            ],
            "sample_index": index,
        }
