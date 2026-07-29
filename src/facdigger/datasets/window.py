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
