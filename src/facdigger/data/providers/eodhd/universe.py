"""Pure paid-plan universe discovery helpers."""

from __future__ import annotations

import math
from typing import Any

from facdigger.data.providers.eodhd.config import EODHDUniverseSelection
from facdigger.data.providers.eodhd.mapper import normalize_security_type, provider_symbol


def discover_historical_symbols(
    active_rows: list[dict[str, Any]],
    delisted_rows: list[dict[str, Any]],
    *,
    exchange_code: str,
    config: EODHDUniverseSelection,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Discover all in-scope active and delisted candidates without current-liquidity ranking."""

    allowed_exchanges = {value.strip().upper() for value in config.exchanges}
    allowed_types = {normalize_security_type(value) for value in config.security_types}
    selected: dict[str, dict[str, Any]] = {}
    active_symbols: set[str] = set()
    delisted_symbols: set[str] = set()
    for rows, is_delisted in ((active_rows, False), (delisted_rows, True)):
        for source in rows:
            code = str(source.get("Code", "")).strip().upper()
            if not code:
                continue
            if str(source.get("Exchange", "")).strip().upper() not in allowed_exchanges:
                continue
            if normalize_security_type(source.get("Type")) not in allowed_types:
                continue
            symbol = provider_symbol(code, exchange_code)
            row = {**source, "_is_delisted": is_delisted}
            selected[symbol] = row
            (delisted_symbols if is_delisted else active_symbols).add(symbol)
    symbols = sorted(selected)
    if not symbols:
        raise ValueError("historical_liquid discovery found no eligible candidates")
    metadata = [selected[symbol] for symbol in symbols]
    audit = {
        "mode": "historical_liquid",
        "research_ready": False,
        "research_ready_reason": (
            "dynamic point-in-time liquidity is enabled, but delisting returns are imputed "
            "and point-in-time risk exposures are unavailable"
        ),
        "active_metadata_rows": len(active_rows),
        "delisted_metadata_rows": len(delisted_rows),
        "active_candidates": len(active_symbols),
        "delisted_candidates": len(delisted_symbols),
        "candidate_count": len(symbols),
        "daily_max_symbols": config.max_symbols,
        "ranking": "daily_adv20_usd_desc_then_security_id",
        "selection_uses_current_liquidity": False,
    }
    return symbols, metadata, audit


def select_top_liquid_symbols(
    metadata_rows: list[dict[str, Any]],
    bulk_rows: list[dict[str, Any]],
    *,
    exchange_code: str,
    config: EODHDUniverseSelection,
) -> tuple[list[str], dict[str, Any]]:
    """Join metadata to latest bulk EOD and rank an engineering pilot universe."""

    allowed_exchanges = {value.strip().upper() for value in config.exchanges}
    allowed_types = {normalize_security_type(value) for value in config.security_types}
    metadata: dict[str, dict[str, Any]] = {}
    for row in metadata_rows:
        code = str(row.get("Code", "")).strip().upper()
        if not code:
            continue
        if str(row.get("Exchange", "")).strip().upper() not in allowed_exchanges:
            continue
        if normalize_security_type(row.get("Type")) not in allowed_types:
            continue
        metadata[code] = row

    candidates: list[dict[str, Any]] = []
    for row in bulk_rows:
        code = str(row.get("code", "")).strip().upper()
        if code not in metadata:
            continue
        try:
            close = float(row["close"])
            average_volume = float(row["avgvol_200d"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(close) or not math.isfinite(average_volume):
            continue
        if close < config.min_price or average_volume < config.min_avg_volume_200d:
            continue
        candidates.append(
            {
                "code": code,
                "provider_symbol": provider_symbol(code, exchange_code),
                "close": close,
                "avgvol_200d": average_volume,
                "dollar_liquidity_200d": close * average_volume,
                "asof_date": row.get("date"),
            }
        )
    candidates.sort(key=lambda row: (-row["dollar_liquidity_200d"], row["code"]))
    selected = candidates[: config.max_symbols]
    if len(selected) < config.max_symbols:
        raise ValueError(
            f"top_liquid selection found {len(selected)} eligible symbols, "
            f"below requested {config.max_symbols}"
        )
    dates = sorted({str(row["asof_date"]) for row in selected if row["asof_date"]})
    audit = {
        "mode": "top_liquid",
        "research_ready": False,
        "bias_warning": (
            "selected from current active listings and current liquidity; "
            "survivorship/look-ahead biased for historical research"
        ),
        "metadata_rows": len(metadata_rows),
        "metadata_rows_after_exchange_type_filter": len(metadata),
        "bulk_rows": len(bulk_rows),
        "eligible_candidates": len(candidates),
        "selected_count": len(selected),
        "selected_asof_dates": dates,
        "ranking": "close_times_avgvol_200d_desc_then_code",
        "minimum_selected_dollar_liquidity_200d": selected[-1]["dollar_liquidity_200d"],
        "maximum_selected_dollar_liquidity_200d": selected[0]["dollar_liquidity_200d"],
    }
    return [str(row["provider_symbol"]) for row in selected], audit
