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

from facdigger.data.contracts import (
    table_audit,
    validate_bars,
    validate_corporate_actions,
)
from facdigger.data.providers.base import ProviderIngestResult
from facdigger.data.providers.eodhd.client import DailyCallBudget, EODHDClient, EODHDError
from facdigger.data.providers.eodhd.config import EODHDConfig
from facdigger.data.providers.eodhd.mapper import (
    build_imputed_delistings,
    build_metadata_index,
    build_universe,
    map_corporate_actions,
    map_eod_bars,
)
from facdigger.data.providers.eodhd.universe import (
    discover_historical_symbols,
    select_top_liquid_symbols,
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

    def _resolve_symbols_and_metadata(
        self, client: EODHDClient, warnings: list[str]
    ) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
        override_rows = [override.model_dump() for override in self.config.metadata_overrides]
        if self.config.universe.mode == "historical_liquid":
            active = client.get_json(
                f"exchange-symbol-list/{self.config.exchange_code}",
                {"type": "common_stock", "delisted": 0},
            )
            delisted = client.get_json(
                f"exchange-symbol-list/{self.config.exchange_code}",
                {"type": "common_stock", "delisted": 1},
            )
            if not isinstance(active, list) or not isinstance(delisted, list):
                raise EODHDError("EODHD historical symbol-list response is not an array")
            symbols, metadata_rows, selection_audit = discover_historical_symbols(
                active,
                delisted,
                exchange_code=self.config.exchange_code,
                config=self.config.universe,
            )
            metadata = build_metadata_index(
                [*metadata_rows, *override_rows],
                self.config.exchange_code,
            )
            warnings.append(
                "delisting terminal returns are conservatively imputed by exchange; "
                "results require sensitivity review"
            )
            return symbols, metadata, selection_audit
        if self.config.universe.mode == "top_liquid":
            payload = client.get_json(
                f"exchange-symbol-list/{self.config.exchange_code}",
                {"type": "common_stock", "delisted": 0},
            )
            if not isinstance(payload, list):
                raise EODHDError("EODHD active symbol-list response is not an array")
            bulk = client.get_json(
                f"eod-bulk-last-day/{self.config.exchange_code}",
                {"filter": "extended"},
                call_cost=100,
            )
            if not isinstance(bulk, list):
                raise EODHDError("EODHD bulk EOD response is not an array")
            symbols, selection_audit = select_top_liquid_symbols(
                payload,
                bulk,
                exchange_code=self.config.exchange_code,
                config=self.config.universe,
            )
            metadata = build_metadata_index(
                [
                    *[{**row, "_is_delisted": False} for row in payload],
                    *override_rows,
                ],
                self.config.exchange_code,
            )
            warnings.append(str(selection_audit["bias_warning"]))
            return symbols, metadata, selection_audit

        symbols = list(self.config.symbols)
        rows = list(override_rows)
        if self.config.fetch_symbol_metadata:
            try:
                for delisted in (0, 1):
                    payload = client.get_json(
                        f"exchange-symbol-list/{self.config.exchange_code}",
                        {"type": "common_stock", "delisted": delisted},
                    )
                    if not isinstance(payload, list):
                        raise EODHDError("EODHD symbol-list response is not an array")
                    rows.extend(
                        {**row, "_is_delisted": bool(delisted)} for row in payload
                    )
            except EODHDError as exc:
                if self.config.metadata_failure_policy == "fail":
                    raise
                warnings.append(f"symbol metadata unavailable: {exc}")
        metadata = build_metadata_index(rows, self.config.exchange_code)
        missing = [symbol for symbol in self.config.symbols if symbol not in metadata]
        if missing:
            warnings.append("security_id uses provider-symbol fallback for: " + ", ".join(missing))
        return (
            symbols,
            metadata,
            {
                "mode": "explicit",
                "research_ready": None,
                "selected_count": len(symbols),
            },
        )

    def probe(self) -> dict[str, Any]:
        client = self._get_client()
        start, end = self.config.resolved_dates()
        warnings: list[str] = []
        symbols, _, selection_audit = self._resolve_symbols_and_metadata(client, warnings)
        payload = client.get_json(
            f"eod/{symbols[0]}",
            {"from": start.isoformat(), "to": end.isoformat(), "period": "d", "order": "a"},
        )
        if not isinstance(payload, list):
            raise EODHDError("EODHD EOD response is not an array")
        fields = sorted({key for row in payload[:10] if isinstance(row, dict) for key in row})
        return {
            "provider": self.name,
            "symbol": symbols[0],
            "resolved_start": start.isoformat(),
            "resolved_end": end.isoformat(),
            "rows": len(payload),
            "fields": fields,
            "sample": payload[-1] if payload else None,
            "requests": client.request_log,
            "budget": client.budget.status(),
            "selection": selection_audit,
            "warnings": warnings,
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
        ]
        if not self.config.delisting_imputation.enabled:
            warnings.append(
                "no delistings table is emitted without a reliable terminal return or value"
            )
        symbols, metadata, selection_audit = self._resolve_symbols_and_metadata(client, warnings)
        eod_batch: dict[str, list[dict[str, Any]]] = {}
        dividend_batch: dict[str, list[dict[str, Any]]] = {}
        split_batch: dict[str, list[dict[str, Any]]] = {}
        bar_parts: list[pl.DataFrame] = []
        action_parts: list[pl.DataFrame] = []
        date_params = {"from": start.isoformat(), "to": end.isoformat()}

        def flush_batch() -> None:
            if not eod_batch:
                return
            bar_parts.append(
                map_eod_bars(
                    eod_batch,
                    metadata,
                    source_revision=source_revision,
                    ingested_at=ingested_at,
                )
            )
            if self.config.include_corporate_actions:
                mapped = map_corporate_actions(
                    dividends_by_symbol=dividend_batch,
                    splits_by_symbol=split_batch,
                    metadata=metadata,
                    source_revision=source_revision,
                )
                if mapped is not None:
                    action_parts.append(mapped)
            eod_batch.clear()
            dividend_batch.clear()
            split_batch.clear()

        for symbol in symbols:
            payload = client.get_json(f"eod/{symbol}", {**date_params, "period": "d", "order": "a"})
            if not isinstance(payload, list):
                raise EODHDError(f"EODHD EOD response for {symbol} is not an array")
            if payload:
                eod_batch[symbol] = payload
            if self.config.include_corporate_actions:
                div_payload = client.get_json(f"div/{symbol}", date_params)
                split_payload = client.get_json(f"splits/{symbol}", date_params)
                if not isinstance(div_payload, list) or not isinstance(split_payload, list):
                    raise EODHDError(
                        f"EODHD corporate-action response for {symbol} is not an array"
                    )
                dividend_batch[symbol] = div_payload
                split_batch[symbol] = split_payload
            if len(eod_batch) >= 250:
                flush_batch()
        flush_batch()
        if not bar_parts:
            raise EODHDError("EODHD returned no usable EOD histories")
        bars = validate_bars(pl.concat(bar_parts, how="vertical_relaxed"))
        universe = build_universe(
            bars,
            min_listed_sessions=self.config.min_listed_sessions,
            min_price=self.config.min_price,
            min_adv20_usd=self.config.min_adv20_usd,
            max_daily_symbols=(
                self.config.universe.max_symbols
                if self.config.universe.mode == "historical_liquid"
                else None
            ),
        )
        actions = (
            validate_corporate_actions(pl.concat(action_parts, how="vertical_relaxed"))
            if action_parts
            else None
        )
        delistings = None
        if self.config.delisting_imputation.enabled:
            imputation = self.config.delisting_imputation
            delistings = build_imputed_delistings(
                bars,
                universe,
                exchange_returns={
                    "XNAS": imputation.nasdaq_return,
                    "XNYS": imputation.nyse_return,
                    "XASE": imputation.nyse_american_return,
                },
                default_return=imputation.default_return,
                source_revision=source_revision,
            )

        output_dir = self.config.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        frames: dict[str, pl.DataFrame] = {"bars": bars, "universe": universe}
        filenames = {"bars": "bars_daily.parquet", "universe": "universe_daily.parquet"}
        if actions is not None:
            frames["corporate_actions"] = actions
            filenames["corporate_actions"] = "corporate_actions.parquet"
        if delistings is not None:
            frames["delistings"] = delistings
            filenames["delistings"] = "delistings.parquet"
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
            "symbols": symbols,
            "selection": selection_audit,
            "requests": client.request_log,
            "budget": client.budget.status(),
            "warnings": warnings,
            "tables": {
                name: {
                    **table_audit(
                        frame,
                        (
                            "ex_date"
                            if name == "corporate_actions"
                            else "delist_date"
                            if name == "delistings"
                            else "trade_date"
                        ),
                    ),
                    "file": files[name].name,
                    "sha256": _sha256(files[name]),
                }
                for name, frame in frames.items()
            },
            "delistings": {
                "emitted": delistings is not None,
                "rows": delistings.height if delistings is not None else 0,
                "policy": self.config.delisting_imputation.model_dump(mode="json"),
                "warning": (
                    "returns are imputed policy assumptions, not provider-observed terminal values"
                    if delistings is not None
                    else "no delisting terminal return/value is available"
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
