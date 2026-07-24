"""Provider-neutral adapter for standardized Parquet inputs."""

from __future__ import annotations

import hashlib
import json
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
        """Honor provider-declared fail-closed gates without coupling table schemas."""

        path = self.config.source_manifest
        if path is None:
            return
        if not path.is_file():
            raise DataContractError(f"Configured source manifest does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise DataContractError(f"Cannot read source manifest {path}: {exc}") from exc
        selection = payload.get("selection") or {}
        if payload.get("provider") != "eodhd" or selection.get("mode") != "historical_liquid":
            return
        gate = ((payload.get("quality") or {}).get("gate") or {})
        if gate.get("status") != "passed":
            raise DataContractError(
                "EODHD historical source manifest does not contain a passed quality gate; "
                "re-run ingestion with the current adapter before building a dataset"
            )
        configured = {
            "bars": self.config.bars,
            "universe": self.config.universe,
            "corporate_actions": self.config.corporate_actions,
            "delistings": self.config.delistings,
        }
        tables = payload.get("tables") or {}
        for name, table_path in configured.items():
            if table_path is None:
                continue
            expected = (tables.get(name) or {}).get("sha256")
            if not expected:
                raise DataContractError(
                    f"EODHD historical source manifest has no SHA-256 for {name}"
                )
            if not table_path.is_file():
                raise DataContractError(f"Source Parquet does not exist: {table_path}")
            digest = hashlib.sha256()
            with table_path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != expected:
                raise DataContractError(
                    f"EODHD historical source hash mismatch for {name}: "
                    f"manifest={expected}, actual={actual}"
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
