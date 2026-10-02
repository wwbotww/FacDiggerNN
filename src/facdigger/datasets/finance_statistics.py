"""Bounded deterministic statistics cache for the matched Finance diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.datasets.window import MarketFeatureWindow
from facdigger.models.finance_patch_transformer import MultiScaleStatisticsEncoder
from facdigger.training.runtime import write_json


def cache_statistics(
    dataset: Any, path: Path, *, windows: tuple[int, ...], identity: dict, check_stop=None
) -> None:
    """Process at most 128 windows at once. No fitted imputer/standardizer is added."""
    path.mkdir(parents=True, exist_ok=False)
    shape = (len(dataset), len(dataset.channels) * (5 * len(windows) + 1))
    values = np.lib.format.open_memmap(path / "local.npy", mode="w+", dtype=np.float32, shape=shape)
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=128, num_workers=0, shuffle=False):
            if check_stop is not None:
                check_stop()
            stats = MultiScaleStatisticsEncoder.statistics(
                batch["values"],
                batch["observed_mask"],
                num_asset_channels=len(dataset.channels),
                windows=windows,
            ).numpy()
            if not np.isfinite(stats).all():
                raise DataContractError("non-finite statistics; no silent imputation")
            values[batch["sample_index"].numpy()] = stats
    values.flush()
    del values
    dates = dataset.sample_rows["asof_date"].unique().sort().to_list()
    market = []
    for day in dates:
        window = dataset.market_store.window(day, context_length=dataset.context_length)
        market.append(
            MultiScaleStatisticsEncoder.statistics(
                torch.from_numpy(window.values).unsqueeze(0),
                torch.from_numpy(window.observed_mask).unsqueeze(0),
                num_asset_channels=len(dataset.market_channels),
                windows=windows,
            )[0].numpy()
        )
    market_array = np.stack(market)
    if not np.isfinite(market_array).all():
        raise DataContractError("non-finite market statistics")
    np.save(path / "market.npy", market_array)
    dataset.sample_rows.write_parquet(path / "index.parquet")
    write_json(
        path / "manifest.json",
        {
            "identity": identity,
            "windows": list(windows),
            "rows": len(dataset),
            "channels": dataset.channels,
            "market_channels": list(dataset.market_channels),
            "context_length": dataset.context_length,
            "primary_horizon": dataset.primary_horizon,
            "market_dates": dates,
            "preprocessing": "snapshot A scaler only; no additional fit",
            "sha256": {
                file: sha256_file(path / file)
                for file in ("local.npy", "market.npy", "index.parquet")
            },
        },
    )


class FinanceStatisticsDataset:
    """A full C view over a verified cache; target projection remains in the trainer."""

    def __init__(
        self, path: Path, *, identity: dict, windows: tuple[int, ...], expected_rows: pl.DataFrame
    ):
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["identity"] != identity or manifest["windows"] != list(windows):
            raise DataContractError("statistics cache data/window identity differs")
        for file, expected in manifest["sha256"].items():
            if sha256_file(path / file) != expected:
                raise DataContractError("statistics cache checksum differs")
        self.sample_rows = pl.read_parquet(path / "index.parquet")
        if not self.sample_rows.equals(expected_rows.sort("asof_date", "security_id")):
            raise DataContractError("statistics cache computational universe differs")
        self.values = np.load(path / "local.npy", mmap_mode="r")
        self.market = np.load(path / "market.npy", mmap_mode="r")
        self.market_positions = {day: i for i, day in enumerate(manifest["market_dates"])}
        self.context_length = manifest["context_length"]
        self.primary_horizon = manifest["primary_horizon"]
        self.channels = manifest["channels"]
        self.market_channels = manifest["market_channels"]
        self.windows = tuple(manifest["windows"])
        self.positions = np.arange(self.sample_rows.height)

    def subset(self, dates: list) -> FinanceStatisticsDataset:
        from copy import copy

        result = copy(self)
        rows = self.sample_rows.with_row_index("_cache_row").filter(
            pl.col("asof_date").is_in(dates)
        )
        result.positions = self.positions[rows["_cache_row"].to_numpy()]
        result.sample_rows = rows.drop("_cache_row")
        return result

    @property
    def asof_dates(self) -> list:
        return self.sample_rows["asof_date"].to_list()

    def __len__(self) -> int:
        return self.sample_rows.height

    def __getitem__(self, index: int) -> dict:
        values = self.values[self.positions[index]][None, :].copy()
        return {
            "values": values,
            "observed_mask": np.ones_like(values, dtype=bool),
            "sample_index": index,
        }

    def market_window_for_sample_indices(self, indices: Any) -> MarketFeatureWindow:
        dates = self.sample_rows[np.asarray(indices).tolist(), "asof_date"].unique().to_list()
        if len(dates) != 1:
            raise DataContractError("statistics market lookup requires one complete date")
        values = self.market[self.market_positions[str(dates[0])]][None, :].copy()
        return MarketFeatureWindow(values, np.ones_like(values, dtype=bool))
