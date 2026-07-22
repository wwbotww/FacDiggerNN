"""EODHD ingestion orchestrator producing provider-neutral standard Parquet."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from facdigger.data.contracts import table_audit
from facdigger.data.providers.base import ProviderIngestResult
from facdigger.data.providers.eodhd.client import DailyCallBudget, EODHDClient, EODHDError
from facdigger.data.providers.eodhd.config import EODHDConfig
from facdigger.data.providers.eodhd.mapper import (
    build_metadata_index,
    build_universe,
    map_corporate_actions,
    map_eod_bars,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class EODHDProvider:
    name = "eodhd"

    def __init__(self, config: EODHDConfig, client: EODHDClient | None = None) -> None:
        self.config = config
        self._client = client
        self._token_source = "injected_client" if client is not None else None

    def _resolve_token(self) -> tuple[str, str]:
        token = os.environ.get(self.config.api_token_env)
        if token:
            return token, self.config.api_token_env
        if self.config.allow_demo_token:
            return "demo", "demo"
        raise EODHDError(
            f"Set {self.config.api_token_env} in the environment; tokens are never read from YAML"
        )

    def _get_client(self) -> EODHDClient:
        if self._client is not None:
            return self._client
        token, token_source = self._resolve_token()
        self._token_source = token_source
        budget = DailyCallBudget(
            self.config.state_dir / "daily_call_budget.json", self.config.max_calls_per_day
        )
        self._client = EODHDClient(
            base_url=str(self.config.base_url),
            api_token=token,
            cache_dir=self.config.cache_dir,
            budget=budget,
            timeout_seconds=self.config.timeout_seconds,
            cache_ttl_hours=self.config.cache_ttl_hours,
            max_retries=self.config.max_retries,
            refresh=self.config.refresh,
        )
        return self._client

    def _metadata(self, client: EODHDClient, warnings: list[str]) -> dict[str, dict[str, Any]]:
        rows = [override.model_dump() for override in self.config.metadata_overrides]
        metadata = build_metadata_index(rows, self.config.exchange_code)
        if self.config.fetch_symbol_metadata:
            try:
                payload = client.get_json(
                    f"exchange-symbol-list/{self.config.exchange_code}",
                    {"type": "common_stock", "delisted": 1},
                )
                if not isinstance(payload, list):
                    raise EODHDError("EODHD symbol-list response is not an array")
                metadata.update(build_metadata_index(payload, self.config.exchange_code))
            except EODHDError as exc:
                if self.config.metadata_failure_policy == "fail":
                    raise
                warnings.append(f"symbol metadata unavailable: {exc}")
        missing = [symbol for symbol in self.config.symbols if symbol not in metadata]
        if missing:
            warnings.append("security_id uses provider-symbol fallback for: " + ", ".join(missing))
        return metadata

    def probe(self) -> dict[str, Any]:
        client = self._get_client()
        start, end = self.config.resolved_dates()
        payload = client.get_json(
            f"eod/{self.config.symbols[0]}",
            {"from": start.isoformat(), "to": end.isoformat(), "period": "d", "order": "a"},
        )
        if not isinstance(payload, list):
            raise EODHDError("EODHD EOD response is not an array")
        fields = sorted({key for row in payload[:10] if isinstance(row, dict) for key in row})
        return {
            "provider": self.name,
            "symbol": self.config.symbols[0],
            "resolved_start": start.isoformat(),
            "resolved_end": end.isoformat(),
            "rows": len(payload),
            "fields": fields,
            "sample": payload[-1] if payload else None,
            "requests": client.request_log,
            "budget": client.budget.status(),
        }

    def ingest(self) -> ProviderIngestResult:
        client = self._get_client()
        start, end = self.config.resolved_dates()
        ingested_at = datetime.now(timezone.utc)
        source_revision = f"eodhd:{ingested_at.isoformat()}"
        warnings = [
            "adjusted_close is converted to one OHLC adjustment factor and includes "
            "splits/dividends",
            "trade status is inferred only from an observed daily bar",
            "historical industry and float market cap are unavailable in this EOD-only ingestion",
            "no delistings table is emitted without a reliable terminal return or value",
        ]
        metadata = self._metadata(client, warnings)
        eod: dict[str, list[dict[str, Any]]] = {}
        dividends: dict[str, list[dict[str, Any]]] = {}
        splits: dict[str, list[dict[str, Any]]] = {}
        date_params = {"from": start.isoformat(), "to": end.isoformat()}
        for symbol in self.config.symbols:
            payload = client.get_json(f"eod/{symbol}", {**date_params, "period": "d", "order": "a"})
            if not isinstance(payload, list):
                raise EODHDError(f"EODHD EOD response for {symbol} is not an array")
            eod[symbol] = payload
            if self.config.include_corporate_actions:
                div_payload = client.get_json(f"div/{symbol}", date_params)
                split_payload = client.get_json(f"splits/{symbol}", date_params)
                if not isinstance(div_payload, list) or not isinstance(split_payload, list):
                    raise EODHDError(
                        f"EODHD corporate-action response for {symbol} is not an array"
                    )
                dividends[symbol] = div_payload
                splits[symbol] = split_payload

        bars = map_eod_bars(
            eod,
            metadata,
            source_revision=source_revision,
            ingested_at=ingested_at,
        )
        universe = build_universe(
            bars,
            min_listed_sessions=self.config.min_listed_sessions,
            min_price=self.config.min_price,
            min_adv20_usd=self.config.min_adv20_usd,
        )
        actions = map_corporate_actions(
            dividends_by_symbol=dividends,
            splits_by_symbol=splits,
            metadata=metadata,
            source_revision=source_revision,
        )

        output_dir = self.config.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        frames: dict[str, pl.DataFrame] = {"bars": bars, "universe": universe}
        filenames = {"bars": "bars_daily.parquet", "universe": "universe_daily.parquet"}
        if actions is not None:
            frames["corporate_actions"] = actions
            filenames["corporate_actions"] = "corporate_actions.parquet"
        files: dict[str, Path] = {}
        for name, frame in frames.items():
            final_path = output_dir / filenames[name]
            temporary = output_dir / f".{final_path.name}.{uuid.uuid4().hex}.tmp"
            frame.write_parquet(temporary)
            temporary.replace(final_path)
            files[name] = final_path

        manifest = {
            "schema_version": 1,
            "provider": self.name,
            "ingested_at": ingested_at.isoformat(),
            "source_revision": source_revision,
            "token_source": self._token_source,
            "resolved_start": start.isoformat(),
            "resolved_end": end.isoformat(),
            "symbols": self.config.symbols,
            "requests": client.request_log,
            "budget": client.budget.status(),
            "warnings": warnings,
            "tables": {
                name: {
                    **table_audit(
                        frame, "ex_date" if name == "corporate_actions" else "trade_date"
                    ),
                    "file": files[name].name,
                    "sha256": _sha256(files[name]),
                }
                for name, frame in frames.items()
            },
            "delistings": {
                "emitted": False,
                "reason": (
                    "EODHD listing status does not provide a defensible terminal return/value"
                ),
            },
        }
        manifest_path = output_dir / "eodhd_ingestion_manifest.json"
        temporary_manifest = output_dir / f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest_path)
        files["manifest"] = manifest_path
        return ProviderIngestResult(self.name, output_dir, files, manifest)
