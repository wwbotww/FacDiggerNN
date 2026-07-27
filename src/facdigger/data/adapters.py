"""Provider-neutral adapter for standardized Parquet inputs."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from facdigger.data.config import ParquetSourceConfig
from facdigger.data.contracts import VALIDATORS, DataBundle, DataContractError, table_audit
from facdigger.data.provenance import (
    read_source_provenance_manifest,
    require_accepted_source,
    validate_source_table_bindings,
)


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
        self._validate_source_manifest()
        return DataBundle(
            bars=self._read(self.config.bars, "bars", required=True),
            universe=self._read(self.config.universe, "universe", required=True),
            corporate_actions=self._read(
                self.config.corporate_actions, "corporate_actions", required=False
            ),
            delistings=self._read(self.config.delistings, "delistings", required=False),
        )

    def _validate_source_manifest(self) -> None:
        """Validate the provider-neutral proof and bind it to configured tables."""

        path = self.config.source_manifest
        if path is None:
            return
        if not path.is_file():
            raise DataContractError(f"Configured source manifest does not exist: {path}")
        provenance = read_source_provenance_manifest(path)
        require_accepted_source(provenance)
        validate_source_table_bindings(
            provenance,
            {
                "bars": self.config.bars,
                "universe": self.config.universe,
                "corporate_actions": self.config.corporate_actions,
                "delistings": self.config.delistings,
            },
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
