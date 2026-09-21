"""EODHD daily revisions for the production-standard data store."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from typing import Any

import polars as pl

from facdigger.data.contracts import DataContractError, validate_bars
from facdigger.data.market_calendar import (
    regular_session_frame,
    regular_sessions,
)
from facdigger.data.providers.eodhd.client import EODHDClient, EODHDError
from facdigger.data.providers.eodhd.config import EODHDConfig
from facdigger.data.providers.eodhd.mapper import (
    build_metadata_index,
    consolidate_bars,
    filter_valid_eod_rows,
    map_eod_bars,
)
from facdigger.data.providers.eodhd.quality import (
    filter_to_regular_sessions,
    quarantine_suspicious_identities,
)
from facdigger.data.providers.eodhd.universe import discover_historical_symbols


class DailyDataNotReady(EODHDError):
    """Raised for provider state which can safely be retried before the cutoff."""


@dataclass(frozen=True)
class EODHDDailyRevision:
    target_date: date
    revision_start: date
    bars: pl.DataFrame
    metadata_rows: tuple[dict[str, Any], ...]
    symbol_count: int
    rejected_rows: int
    request_log: tuple[dict[str, Any], ...]
    source_revision: str
    ingested_at: datetime
    backfilled_provider_symbols: tuple[str, ...] = ()
    rejected_rows_by_symbol: tuple[tuple[str, int], ...] = ()
    raw_quality_audits: tuple[dict[str, Any], ...] = ()


def _discover(
    client: EODHDClient,
    config: EODHDConfig,
) -> tuple[list[str], list[dict[str, Any]]]:
    active = client.get_json(
        f"exchange-symbol-list/{config.exchange_code}",
        {"type": "common_stock", "delisted": 0},
    )
    delisted = client.get_json(
        f"exchange-symbol-list/{config.exchange_code}",
        {"type": "common_stock", "delisted": 1},
    )
    if not isinstance(active, list) or not isinstance(delisted, list):
        raise EODHDError("EODHD historical symbol-list response is not an array")
    symbols, rows, _ = discover_historical_symbols(
        active,
        delisted,
        exchange_code=config.exchange_code,
        config=config.universe,
    )
    return symbols, rows


def _mapped_rows(
    rows_by_symbol: dict[str, list[dict[str, Any]]],
    metadata: dict[str, dict[str, Any]],
    *,
    source_revision: str,
    ingested_at: datetime,
) -> tuple[pl.DataFrame | None, int, dict[str, int]]:
    valid_by_symbol: dict[str, list[dict[str, Any]]] = {}
    rejected = 0
    rejected_by_symbol: dict[str, int] = {}
    for symbol, rows in rows_by_symbol.items():
        valid, invalid = filter_valid_eod_rows(rows)
        rejected += len(invalid)
        if invalid:
            rejected_by_symbol[symbol] = len(invalid)
        if valid:
            valid_by_symbol[symbol] = valid
    if not valid_by_symbol:
        return None, rejected, rejected_by_symbol
    return (
        map_eod_bars(
            valid_by_symbol,
            metadata,
            source_revision=source_revision,
            ingested_at=ingested_at,
            consolidate_aliases=False,
        ),
        rejected,
        rejected_by_symbol,
    )


def _quality_checked_bars(
    bars: pl.DataFrame, config: EODHDConfig, calendar: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Inspect raw aliases before consolidation can hide conflicting observations."""
    bars, calendar_audit = filter_to_regular_sessions(bars, calendar)
    gate = config.quality_gate
    bars, identity_audit = quarantine_suspicious_identities(
        bars, calendar,
        max_adjusted_price_ratio=gate.max_adjusted_price_ratio,
        max_alias_overlap_relative_diff=gate.max_alias_overlap_relative_diff,
        max_quarantined_security_fraction=gate.max_quarantined_security_fraction,
    )
    return validate_bars(consolidate_bars(bars)), {
        "calendar": calendar_audit, "identity": identity_audit,
    }


def fetch_daily_revision(
    client: EODHDClient,
    config: EODHDConfig,
    *,
    revision_start: date,
    target_date: date,
) -> EODHDDailyRevision:
    """Fetch one fresh inclusive window without writing provider-neutral storage."""

    if config.universe.mode != "historical_liquid":
        raise ValueError("daily production requires historical_liquid universe semantics")
    if revision_start > target_date:
        raise ValueError("daily revision start must not follow its target")
    symbols, metadata_rows = _discover(client, config)
    metadata = build_metadata_index(metadata_rows, config.exchange_code)
    ingested_at = datetime.now(timezone.utc)
    source_revision = f"eodhd-daily:{target_date.isoformat()}"
    parts: list[pl.DataFrame] = []
    rejected = 0
    rejected_by_symbol: dict[str, int] = {}
    expected_symbols = set(symbols)
    for session in regular_sessions(revision_start, target_date):
        payload = client.get_json(
            f"eod-bulk-last-day/{config.exchange_code}",
            {"date": session.isoformat()},
            call_cost=100,
        )
        if not isinstance(payload, list):
            raise EODHDError("EODHD bulk EOD response is not an array")
        by_symbol: dict[str, list[dict[str, Any]]] = {}
        for row in payload:
            if not isinstance(row, dict):
                rejected += 1
                continue
            code = str(row.get("code", "")).strip().upper()
            symbol = f"{code}.{config.exchange_code}" if code else ""
            if symbol in expected_symbols:
                by_symbol.setdefault(symbol, []).append(row)
        mapped, invalid, rejected_symbols = _mapped_rows(
            by_symbol,
            metadata,
            source_revision=source_revision,
            ingested_at=ingested_at,
        )
        rejected += invalid
        for symbol, count in rejected_symbols.items():
            rejected_by_symbol[symbol] = rejected_by_symbol.get(symbol, 0) + count
        if mapped is not None:
            parts.append(mapped)
    if not parts:
        raise DailyDataNotReady("EODHD returned no usable daily revision bars")
    bars = pl.concat(parts, how="vertical_relaxed")
    calendar = regular_session_frame(revision_start, target_date)
    bars, raw_quality = _quality_checked_bars(bars, config, calendar)
    maximum = bars["trade_date"].max()
    if maximum != target_date:
        raise DailyDataNotReady(
            "EODHD daily revision is not ready for target date "
            f"{target_date.isoformat()}; latest={maximum}"
        )
    return EODHDDailyRevision(
        target_date=target_date,
        revision_start=revision_start,
        bars=bars,
        metadata_rows=tuple(metadata_rows),
        symbol_count=len(symbols),
        rejected_rows=rejected,
        request_log=tuple(client.request_log),
        source_revision=source_revision,
        ingested_at=ingested_at,
        rejected_rows_by_symbol=tuple(sorted(rejected_by_symbol.items())),
        raw_quality_audits=({"stage": "bulk_revision", **raw_quality},),
    )


def backfill_adjusted_histories(
    client: EODHDClient,
    config: EODHDConfig,
    revision: EODHDDailyRevision,
    *,
    provider_symbols: list[str],
    history_start: date,
) -> EODHDDailyRevision:
    """Replace changed identities with fresh full-window per-symbol histories."""

    requested = sorted(set(provider_symbols))
    if not requested:
        return revision
    metadata = build_metadata_index(list(revision.metadata_rows), config.exchange_code)
    backfill_parts: list[pl.DataFrame] = []
    rejected = revision.rejected_rows
    rejected_by_symbol = dict(revision.rejected_rows_by_symbol)
    for symbol in requested:
        payload = client.get_json(
            f"eod/{symbol}",
            {
                "from": history_start.isoformat(),
                "to": revision.target_date.isoformat(),
                "period": "d",
                "order": "a",
            },
        )
        if not isinstance(payload, list):
            raise EODHDError("EODHD targeted history response is not an array")
        mapped, invalid, rejected_symbols = _mapped_rows(
            {symbol: [row for row in payload if isinstance(row, dict)]},
            metadata,
            source_revision=revision.source_revision,
            ingested_at=revision.ingested_at,
        )
        rejected += invalid + sum(not isinstance(row, dict) for row in payload)
        for rejected_symbol, count in rejected_symbols.items():
            rejected_by_symbol[rejected_symbol] = rejected_by_symbol.get(rejected_symbol, 0) + count
        if mapped is None:
            raise DailyDataNotReady(f"EODHD returned no targeted history for {symbol}")
        backfill_parts.append(mapped)
    backfill, raw_quality = _quality_checked_bars(
        pl.concat(backfill_parts), config,
        regular_session_frame(history_start, revision.target_date),
    )
    if raw_quality["identity"]["quarantined_securities"]:
        # An unsafe replacement cannot repair an adjustment boundary. Never keep
        # the old history and pretend that the requested full backfill succeeded.
        raise DailyDataNotReady("targeted adjustment history has conflicting security identities")
    merged = validate_bars(
        consolidate_bars(
            pl.concat([revision.bars, backfill], how="vertical_relaxed")
        )
    )
    return replace(
        revision,
        bars=merged,
        rejected_rows=rejected,
        request_log=tuple(client.request_log),
        backfilled_provider_symbols=tuple(requested),
        rejected_rows_by_symbol=tuple(sorted(rejected_by_symbol.items())),
        raw_quality_audits=(
            *revision.raw_quality_audits, {"stage": "adjustment_backfill", **raw_quality},
        ),
    )


def daily_revision_audit(revision: EODHDDailyRevision) -> dict[str, Any]:
    return {
        "target_date": revision.target_date.isoformat(),
        "revision_start": revision.revision_start.isoformat(),
        "rows": revision.bars.height,
        "securities": revision.bars["security_id"].n_unique(),
        "symbols_requested": revision.symbol_count,
        "rejected_rows": revision.rejected_rows,
        "rejected_rows_by_symbol": dict(revision.rejected_rows_by_symbol),
        "requests": list(revision.request_log),
        "backfilled_provider_symbols": list(revision.backfilled_provider_symbols),
        "raw_quality": list(revision.raw_quality_audits),
    }


def require_fresh_daily_requests(revision: EODHDDailyRevision) -> None:
    """Daily production may never publish from provider cache alone."""

    eod_requests = [
        request
        for request in revision.request_log
        if str(request.get("path", "")).startswith(("eod-bulk-last-day/", "eod/"))
    ]
    if not eod_requests or any(request.get("cache_hit") is True for request in eod_requests):
        raise DataContractError("daily production requires fresh EODHD EOD responses")
