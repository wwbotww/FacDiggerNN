"""Pure mappings from EODHD payloads to FacDigger standard tables."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import polars as pl

from facdigger.data.contracts import (
    validate_bars,
    validate_corporate_actions,
    validate_universe,
)

EXCHANGE_MAP = {
    "NASDAQ": "XNAS",
    "NYSE": "XNYS",
    "NYSE ARCA": "ARCX",
    "NYSE MKT": "XASE",
    "AMEX": "XASE",
    "OTCQX": "OTCM",
    "OTCQB": "OTCM",
    "PINK": "OTCM",
    "US": "US",
}


def normalize_security_type(value: str | None) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", (value or "unknown").strip().lower()).strip("_")
    aliases = {
        "common_stock": "common_stock",
        "common": "common_stock",
        "stock": "common_stock",
        "preferred_stock": "preferred_stock",
        "etf": "etf",
        "fund": "fund",
    }
    return aliases.get(normalized, normalized or "unknown")


def provider_symbol(code: str, exchange_code: str) -> str:
    code = code.strip().upper()
    return code if "." in code else f"{code}.{exchange_code.upper()}"


def display_symbol(value: str) -> str:
    return value.rsplit(".", 1)[0].upper()


def build_metadata_index(
    rows: list[dict[str, Any]], exchange_code: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        code = row.get("Code") or row.get("provider_symbol")
        if not code:
            continue
        ticker = provider_symbol(str(code), exchange_code)
        result[ticker] = {
            "provider_symbol": ticker,
            "isin": row.get("Isin") or row.get("isin"),
            "name": row.get("Name") or row.get("name"),
            "exchange": row.get("Exchange") or row.get("exchange") or exchange_code,
            "currency": row.get("Currency") or row.get("currency") or "USD",
            "security_type": row.get("Type") or row.get("security_type") or "Common Stock",
        }
    return result


def security_identity(ticker: str, metadata: dict[str, Any] | None) -> tuple[str, str]:
    isin = (metadata or {}).get("isin")
    if isinstance(isin, str) and isin.strip():
        return f"eodhd:isin:{isin.strip().upper()}", "isin"
    return f"eodhd:symbol:{ticker.upper()}", "provider_symbol_fallback"


def map_eod_bars(
    rows_by_symbol: dict[str, list[dict[str, Any]]],
    metadata: dict[str, dict[str, Any]],
    *,
    source_revision: str,
    ingested_at: datetime,
) -> pl.DataFrame:
    records: list[dict[str, Any]] = []
    for ticker, rows in rows_by_symbol.items():
        meta = metadata.get(ticker)
        security_id, identity_quality = security_identity(ticker, meta)
        for row in rows:
            close = float(row["close"])
            adjusted_close = float(row.get("adjusted_close", close))
            adjustment = adjusted_close / close if close else None
            volume = float(row["volume"])
            records.append(
                {
                    "security_id": security_id,
                    "symbol": display_symbol(ticker),
                    "trade_date": row["date"],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": close,
                    "volume": volume,
                    "dollar_volume": close * volume,
                    "adj_factor": adjustment,
                    "source_revision": source_revision,
                    "ingested_at": ingested_at,
                    "provider": "eodhd",
                    "provider_symbol": ticker,
                    "adjusted_close": adjusted_close,
                    "adjustment_basis": "adjusted_close_over_raw_close_splits_and_dividends",
                    "identity_quality": identity_quality,
                    "exchange_source": (meta or {}).get("exchange"),
                    "security_type_source": (meta or {}).get("security_type"),
                }
            )
    if not records:
        raise ValueError("EODHD returned no usable EOD rows")
    return validate_bars(pl.DataFrame(records))


def build_universe(
    bars: pl.DataFrame,
    *,
    min_listed_sessions: int,
    min_price: float,
    min_adv20_usd: float,
) -> pl.DataFrame:
    frame = (
        bars.sort(["security_id", "trade_date"])
        .with_columns(
            pl.col("dollar_volume")
            .rolling_mean(window_size=20, min_samples=20)
            .over("security_id")
            .alias("adv20_usd"),
            pl.int_range(1, pl.len() + 1).over("security_id").alias("listed_days"),
            pl.col("exchange_source")
            .fill_null("US")
            .str.to_uppercase()
            .replace_strict(EXCHANGE_MAP, default=pl.col("exchange_source").fill_null("US"))
            .alias("exchange"),
            pl.col("security_type_source")
            .map_elements(normalize_security_type, return_dtype=pl.String)
            .alias("security_type"),
        )
        .with_columns(
            pl.lit(True).alias("is_primary_listing"),
            pl.lit(True).alias("is_listed"),
            pl.lit(False).alias("is_delisted"),
            pl.lit(False).alias("is_halted"),
            pl.lit(None, dtype=pl.String).alias("industry_code"),
            pl.lit(None, dtype=pl.Float64).alias("float_market_cap"),
            pl.lit("observed_bar_assumed_tradable").alias("trade_status_quality"),
        )
        .with_columns(
            (
                (pl.col("security_type") == "common_stock")
                & (pl.col("listed_days") >= min_listed_sessions)
                & (pl.col("close") >= min_price)
                & (pl.col("adv20_usd") >= min_adv20_usd)
            )
            .fill_null(False)
            .alias("eligible")
        )
        .select(
            "security_id",
            "symbol",
            "trade_date",
            "listed_days",
            "exchange",
            "security_type",
            "is_primary_listing",
            "is_listed",
            "is_delisted",
            "is_halted",
            "industry_code",
            "float_market_cap",
            "close",
            "adv20_usd",
            "eligible",
            "trade_status_quality",
            "identity_quality",
            "provider_symbol",
        )
    )
    return validate_universe(frame)


def parse_split_ratio(value: str) -> tuple[float, float]:
    parts = value.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid EODHD split ratio: {value!r}")
    new_shares, old_shares = (float(part) for part in parts)
    if new_shares <= 0 or old_shares <= 0:
        raise ValueError(f"Invalid EODHD split ratio: {value!r}")
    return old_shares / new_shares, new_shares / old_shares


def map_corporate_actions(
    *,
    dividends_by_symbol: dict[str, list[dict[str, Any]]],
    splits_by_symbol: dict[str, list[dict[str, Any]]],
    metadata: dict[str, dict[str, Any]],
    source_revision: str,
) -> pl.DataFrame | None:
    records: list[dict[str, Any]] = []
    symbols = set(dividends_by_symbol) | set(splits_by_symbol)
    for ticker in symbols:
        security_id, _ = security_identity(ticker, metadata.get(ticker))
        for row in dividends_by_symbol.get(ticker, []):
            ex_date = row["date"]
            declared = row.get("declarationDate")
            known_at = declared if declared and declared <= ex_date else ex_date
            records.append(
                {
                    "security_id": security_id,
                    "ex_date": ex_date,
                    "action_type": "cash_dividend",
                    "price_factor": 1.0,
                    "volume_factor": 1.0,
                    "cash_amount": float(row["value"]),
                    "known_at": known_at,
                    "source_revision": source_revision,
                    "provider_symbol": ticker,
                    "currency": row.get("currency"),
                    "known_at_quality": (
                        "declaration_date" if known_at == declared else "ex_date_assumed"
                    ),
                }
            )
        for row in splits_by_symbol.get(ticker, []):
            price_factor, volume_factor = parse_split_ratio(str(row["split"]))
            records.append(
                {
                    "security_id": security_id,
                    "ex_date": row["date"],
                    "action_type": "split",
                    "price_factor": price_factor,
                    "volume_factor": volume_factor,
                    "cash_amount": 0.0,
                    "known_at": row["date"],
                    "source_revision": source_revision,
                    "provider_symbol": ticker,
                    "currency": None,
                    "known_at_quality": "ex_date_assumed",
                }
            )
    if not records:
        return None
    return validate_corporate_actions(pl.DataFrame(records))
