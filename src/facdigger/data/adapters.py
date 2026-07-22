"""Provider-neutral adapter for standardized Parquet inputs."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from facdigger.data.config import ParquetSourceConfig
from facdigger.data.contracts import VALIDATORS, DataBundle, DataContractError, table_audit


class StandardParquetAdapter:
    def __init__(self, config: ParquetSourceConfig) -> None:
        self.config = config

    @staticmethod
    def _read(path: Path | None, table: str, required: bool) -> pl.DataFrame | None:
        if path is None:
            if required:
                raise DataContractError(f"Required source path is not configured: {table}")
            return None
        if not path.is_file():
            raise DataContractError(f"Source Parquet does not exist: {path}")
        try:
            frame = pl.read_parquet(path)
        except Exception as exc:
            raise DataContractError(f"Cannot read {table} from {path}: {exc}") from exc
        return VALIDATORS[table](frame)

    def load(self) -> DataBundle:
        return DataBundle(
            bars=self._read(self.config.bars, "bars", required=True),
            universe=self._read(self.config.universe, "universe", required=True),
            corporate_actions=self._read(
                self.config.corporate_actions, "corporate_actions", required=False
            ),
            delistings=self._read(self.config.delistings, "delistings", required=False),
        )

    def audit(self, bundle: DataBundle | None = None) -> dict:
        loaded = bundle or self.load()
        result = {
            "bars": table_audit(loaded.bars, "trade_date"),
            "universe": table_audit(loaded.universe, "trade_date"),
            "corporate_actions": None,
            "delistings": None,
        }
        if loaded.corporate_actions is not None:
            result["corporate_actions"] = table_audit(loaded.corporate_actions, "ex_date")
        if loaded.delistings is not None:
            result["delistings"] = table_audit(loaded.delistings, "delist_date")
        return result
