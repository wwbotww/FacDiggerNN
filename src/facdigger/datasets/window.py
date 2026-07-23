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
    dates: list[Any]
    values: np.ndarray
    observed: np.ndarray
    position_by_date: dict[Any, int]


class SnapshotWindowDataset:
    """Materialize one `[context_length, channels]` window per indexed sample."""

    def __init__(
        self,
        *,
        features: pl.DataFrame,
        sample_index: pl.DataFrame,
        channels: list[str],
        context_length: int,
        split: str,
    ) -> None:
        if context_length < 1:
            raise ValueError("context_length must be positive")
        missing = [channel for channel in channels if channel not in features.columns]
        if missing:
            raise DataContractError(f"features missing requested channels: {missing}")
        self.channels = list(channels)
        self.context_length = context_length
        self.split = split
        self.sample_rows = sample_index.filter(pl.col("split") == split).sort(
            ["asof_date", "security_id"]
        )
        if self.sample_rows.is_empty():
            raise DataContractError(f"sample_index has no rows for split={split!r}")
        self.blocks: dict[str, SecurityFeatureBlock] = {}
        ordered = features.sort(["security_id", "trade_date"])
        for block in ordered.partition_by("security_id", maintain_order=True):
            security_id = str(block["security_id"][0])
            raw_values = block.select(channels).to_numpy().astype(np.float32)
            finite = np.isfinite(raw_values)
            observed_columns = [f"observed_{channel}" for channel in channels]
            if all(column in block.columns for column in observed_columns):
                source_observed = block.select(observed_columns).to_numpy().astype(bool)
                observed = finite & source_observed
            else:
                observed = finite
            values = np.where(observed, raw_values, 0.0).astype(np.float32)
            dates = block["trade_date"].to_list()
            self.blocks[security_id] = SecurityFeatureBlock(
                dates=dates,
                values=values,
                observed=observed,
                position_by_date={trade_date: index for index, trade_date in enumerate(dates)},
            )
        self._locations: list[tuple[str, int]] = []
        for row in self.sample_rows.iter_rows(named=True):
            security_id = str(row["security_id"])
            block = self.blocks.get(security_id)
            if block is None or row["asof_date"] not in block.position_by_date:
                raise DataContractError(f"sample has no matching feature row: {row['sample_id']}")
            end = block.position_by_date[row["asof_date"]]
            start = end - context_length + 1
            if "feature_end" in row and row["feature_end"] != row["asof_date"]:
                raise DataContractError(
                    f"sample feature_end must equal asof_date: {row['sample_id']}"
                )
            if start < 0 or block.dates[start] != row["feature_start"]:
                raise DataContractError(
                    f"sample window bounds disagree with snapshot index: {row['sample_id']}"
                )
            self._locations.append((security_id, start))

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
        return len(self._locations)

    def __getitem__(self, index: int) -> dict[str, Any]:
        security_id, start = self._locations[index]
        block = self.blocks[security_id]
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
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        security_id, start = self._locations[index]
        block = self.blocks[security_id]
        stop = start + self.context_length
        return {
            "values": block.values[start:stop],
            "observed_mask": block.observed[start:stop],
            "sample_index": index,
        }
