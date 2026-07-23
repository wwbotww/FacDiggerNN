"""Pure mappings from EODHD payloads to FacDigger standard tables."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import polars as pl

from facdigger.data.contracts import (
    validate_bars,
    validate_corporate_actions,
    validate_delistings,
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
            "is_delisted": bool(row.get("_is_delisted", row.get("IsDelisted", False))),
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
                    "is_delisted_source": bool((meta or {}).get("is_delisted", False)),
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
    max_daily_symbols: int | None = None,
) -> pl.DataFrame:
    """Build a full security-session grid and optional daily dynamic-liquidity universe."""

    ordered = bars.sort(["security_id", "trade_date"])
    calendar = ordered.select("trade_date").unique().sort("trade_date")
    maximum_date = calendar["trade_date"].max()
    next_sessions = calendar.with_columns(
        pl.col("trade_date").shift(-1).alias("_next_session")
    )
    securities = (
        ordered.group_by("security_id", maintain_order=True)
        .agg(
            pl.col("symbol").first().alias("_fallback_symbol"),
            pl.col("trade_date").min().alias("_first_trade_date"),
            pl.col("trade_date").max().alias("_last_trade_date"),
            pl.col("exchange_source").drop_nulls().first().alias("_exchange_source"),
            pl.col("security_type_source")
            .drop_nulls()
            .first()
            .alias("_security_type_source"),
            pl.col("identity_quality").first().alias("identity_quality"),
            pl.col("provider_symbol").first().alias("provider_symbol"),
            pl.col("is_delisted_source").max().alias("_eventually_delisted"),
        )
        .join(
            next_sessions,
            left_on="_last_trade_date",
            right_on="trade_date",
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.when(pl.col("_eventually_delisted"))
            .then(pl.coalesce("_next_session", "_last_trade_date"))
            .otherwise(pl.lit(maximum_date))
            .alias("_range_end")
        )
    )
    observations = ordered.select(
        "security_id",
        "trade_date",
        pl.col("symbol").alias("_observed_symbol"),
        "close",
        "dollar_volume",
    )
    frame = (
        securities.join(calendar, how="cross")
        .filter(
            (pl.col("trade_date") >= pl.col("_first_trade_date"))
            & (pl.col("trade_date") <= pl.col("_range_end"))
        )
        .join(
            observations,
            on=["security_id", "trade_date"],
            how="left",
            validate="1:1",
        )
        .sort(["security_id", "trade_date"])
        .with_columns(
            pl.col("_observed_symbol")
            .forward_fill()
            .backward_fill()
            .over("security_id")
            .fill_null(pl.col("_fallback_symbol"))
            .alias("symbol"),
            pl.col("dollar_volume")
            .rolling_mean(window_size=20, min_samples=20)
            .over("security_id")
            .alias("adv20_usd"),
            pl.int_range(1, pl.len() + 1).over("security_id").alias("listed_days"),
            pl.col("_exchange_source")
            .fill_null("US")
            .str.to_uppercase()
            .replace_strict(
                EXCHANGE_MAP,
                default=pl.col("_exchange_source").fill_null("US"),
            )
            .alias("exchange"),
            pl.col("_security_type_source")
            .map_elements(normalize_security_type, return_dtype=pl.String)
            .alias("security_type"),
        )
        .with_columns(
            pl.lit(True).alias("is_primary_listing"),
            (
                ~pl.col("_eventually_delisted")
                | (pl.col("trade_date") <= pl.col("_last_trade_date"))
            ).alias("is_listed"),
            (
                pl.col("_eventually_delisted")
                & (pl.col("trade_date") > pl.col("_last_trade_date"))
            ).alias("is_delisted"),
            (
                (
                    ~pl.col("_eventually_delisted")
                    | (pl.col("trade_date") <= pl.col("_last_trade_date"))
                )
                & pl.col("close").is_null()
            ).alias("is_halted"),
            pl.lit(None, dtype=pl.String).alias("industry_code"),
            pl.lit(None, dtype=pl.Float64).alias("float_market_cap"),
            pl.when(pl.col("close").is_not_null())
            .then(pl.lit("observed_bar"))
            .when(
                pl.col("_eventually_delisted")
                & (pl.col("trade_date") > pl.col("_last_trade_date"))
            )
            .then(pl.lit("imputed_delisted_session"))
            .otherwise(pl.lit("missing_bar_assumed_halt"))
            .alias("trade_status_quality"),
        )
        .with_columns(
            (
                (pl.col("security_type") == "common_stock")
                & (pl.col("listed_days") >= min_listed_sessions)
                & (pl.col("close") >= min_price)
                & (pl.col("adv20_usd") >= min_adv20_usd)
                & pl.col("is_listed")
                & ~pl.col("is_halted")
            )
            .fill_null(False)
            .alias("_eligible_candidate")
        )
    )
    if max_daily_symbols is not None:
        ranks = (
            frame.filter(pl.col("_eligible_candidate"))
            .sort(
                ["trade_date", "adv20_usd", "security_id"],
                descending=[False, True, False],
            )
            .with_columns(
                pl.int_range(1, pl.len() + 1)
                .over("trade_date")
                .alias("liquidity_rank")
            )
            .select("security_id", "trade_date", "liquidity_rank")
        )
        frame = frame.join(
            ranks,
            on=["security_id", "trade_date"],
            how="left",
            validate="1:1",
        ).with_columns(
            (
                pl.col("_eligible_candidate")
                & (pl.col("liquidity_rank") <= max_daily_symbols)
            )
            .fill_null(False)
            .alias("eligible")
        )
    else:
        frame = frame.with_columns(
            pl.col("_eligible_candidate").alias("eligible"),
            pl.lit(None, dtype=pl.Int64).alias("liquidity_rank"),
        )
    frame = frame.select(
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
            "liquidity_rank",
            "_eventually_delisted",
            "_last_trade_date",
        )
    return validate_universe(frame)


def build_imputed_delistings(
    bars: pl.DataFrame,
    universe: pl.DataFrame,
    *,
    exchange_returns: dict[str, float],
    default_return: float,
    source_revision: str,
) -> pl.DataFrame | None:
    """Create explicitly marked, conservative delisting-return imputations."""

    last_rows = (
        bars.filter(pl.col("is_delisted_source"))
        .sort(["security_id", "trade_date"])
        .group_by("security_id", maintain_order=True)
        .tail(1)
        .select(
            "security_id",
            pl.col("trade_date").alias("last_trade_date"),
        )
    )
    if last_rows.is_empty():
        return None
    delist_dates = (
        universe.filter(pl.col("is_delisted"))
        .group_by("security_id")
        .agg(
            pl.col("trade_date").min().alias("delist_date"),
            pl.col("exchange").first().alias("exchange"),
        )
    )
    rows = last_rows.join(
        delist_dates,
        on="security_id",
        how="inner",
        validate="1:1",
    )
    if rows.is_empty():
        return None
    return validate_delistings(
        rows.with_columns(
            pl.col("exchange")
            .replace_strict(exchange_returns, default=default_return)
            .cast(pl.Float64)
            .alias("delisting_return"),
            pl.lit(None, dtype=pl.Float64).alias("terminal_value"),
            pl.col("delist_date").alias("known_at"),
            pl.lit(source_revision).alias("source_revision"),
            pl.lit(True).alias("is_imputed"),
            pl.concat_str(
                pl.lit("exchange_penalty:"),
                "exchange",
            ).alias("imputation_method"),
        ).select(
            "security_id",
            "delist_date",
            "last_trade_date",
            "delisting_return",
            "terminal_value",
            "known_at",
            "source_revision",
            "is_imputed",
            "imputation_method",
            "exchange",
        )
    )


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
