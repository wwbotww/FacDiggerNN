"""Pure mappings from EODHD payloads to FacDigger standard tables."""

from __future__ import annotations

import math
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


def filter_valid_eod_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate usable observations from provider placeholders or corrupt OHLCV rows."""

    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        try:
            open_ = float(row["open"])
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            volume = float(row["volume"])
            adjusted_close = float(row.get("adjusted_close", close))
            adjustment = adjusted_close / close if close else math.nan
            values = (open_, high, low, close, volume, adjusted_close, adjustment)
            is_valid = (
                bool(row.get("date"))
                and all(math.isfinite(value) for value in values)
                and min(open_, high, low, close) > 0
                and high >= max(open_, close, low)
                and low <= min(open_, close, high)
                and volume >= 0
                and adjustment > 0
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            is_valid = False
        (valid if is_valid else rejected).append(row)
    return valid, rejected


def consolidate_bars(frame: pl.DataFrame) -> pl.DataFrame:
    """Resolve overlapping EODHD ticker aliases for one stable security identity.

    EODHD can expose both old and current tickers with the same ISIN and return
    overlapping histories for both endpoints. Prefer an active alias, then the
    alias with the latest coverage, with provider symbol as a deterministic
    final tie-breaker. Non-overlapping history from old tickers is retained.
    """

    if frame.is_empty():
        return frame
    return (
        frame.with_columns(
            pl.col("trade_date")
            .max()
            .over("provider_symbol")
            .alias("_provider_last_trade_date")
        )
        .sort(
            [
                "security_id",
                "trade_date",
                "is_delisted_source",
                "_provider_last_trade_date",
                "provider_symbol",
            ],
            descending=[False, False, False, True, False],
        )
        .unique(subset=["security_id", "trade_date"], keep="first", maintain_order=True)
        .drop("_provider_last_trade_date")
    )


def map_eod_bars(
    rows_by_symbol: dict[str, list[dict[str, Any]]],
    metadata: dict[str, dict[str, Any]],
    *,
    source_revision: str,
    ingested_at: datetime,
    consolidate_aliases: bool = True,
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
    frame = pl.DataFrame(records)
    return validate_bars(consolidate_bars(frame)) if consolidate_aliases else frame


def build_universe(
    bars: pl.DataFrame,
    *,
    min_listed_sessions: int,
    min_price: float,
    min_adv20_usd: float,
    max_daily_symbols: int | None = None,
    calendar: pl.DataFrame | None = None,
    listed_day_offsets: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build a full security-session grid and optional daily dynamic-liquidity universe.

    Provider ingestion must pass an exchange calendar.  The bars-union fallback
    remains available for provider-neutral unit tests and manually supplied data.
    """

    ordered = bars.sort(["security_id", "trade_date"])
    calendar = (
        calendar.select("trade_date").unique().sort("trade_date")
        if calendar is not None
        else ordered.select("trade_date").unique().sort("trade_date")
    )
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
            # An active alias supersedes an old alias marked delisted.  Using
            # max() here falsely delisted still-active identities after ticker changes.
            pl.col("is_delisted_source").min().alias("_eventually_delisted"),
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
    )
    if listed_day_offsets is None:
        frame = frame.with_columns(pl.lit(0, dtype=pl.Int64).alias("_listed_day_offset"))
    else:
        required = {"security_id", "listed_day_offset"}
        if set(listed_day_offsets.columns) != required:
            raise ValueError(
                "listed_day_offsets columns must exactly equal security_id and "
                "listed_day_offset"
            )
        if listed_day_offsets["security_id"].n_unique() != listed_day_offsets.height:
            raise ValueError("listed_day_offsets security_id values must be unique")
        if listed_day_offsets.filter(pl.col("listed_day_offset") < 0).height:
            raise ValueError("listed_day_offsets values must be non-negative")
        frame = frame.join(
            listed_day_offsets.with_columns(pl.col("listed_day_offset").cast(pl.Int64)),
            on="security_id",
            how="left",
            validate="m:1",
        ).with_columns(
            pl.col("listed_day_offset").fill_null(0).alias("_listed_day_offset")
        ).drop("listed_day_offset")
    frame = (
        frame
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
            (
                pl.int_range(1, pl.len() + 1).over("security_id")
                + pl.col("_listed_day_offset")
            ).alias("listed_days"),
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

    lifecycle = (
        universe.filter(pl.col("_eventually_delisted"))
        .group_by("security_id")
        .agg(
            pl.col("_last_trade_date").first().alias("last_trade_date"),
            pl.col("trade_date")
            .filter(pl.col("is_delisted"))
            .min()
            .alias("delist_date"),
            pl.col("exchange").first().alias("exchange"),
        )
        .filter(pl.col("delist_date").is_not_null())
    )
    if lifecycle.is_empty():
        return None
    return validate_delistings(
        lifecycle.with_columns(
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
                    "is_delisted_source": bool(
                        (metadata.get(ticker) or {}).get("is_delisted", False)
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
                    "is_delisted_source": bool(
                        (metadata.get(ticker) or {}).get("is_delisted", False)
                    ),
                }
            )
    if not records:
        return None
    return validate_corporate_actions(pl.DataFrame(records))


def consolidate_corporate_actions(
    frame: pl.DataFrame,
) -> tuple[pl.DataFrame | None, dict[str, Any]]:
    """Deduplicate alias copies and quarantine economically conflicting events.

    Rows with the same stable identity, ex-date and action type are one candidate
    event.  Equal economic terms are collapsed deterministically.  If terms
    disagree, the entire event is excluded rather than guessing which alias is
    correct.
    """

    if frame.is_empty():
        return None, {
            "exact_alias_rows_removed": 0,
            "conflict_groups_dropped": 0,
            "conflict_rows_dropped": 0,
            "conflict_security_ids": [],
            "conflict_examples": [],
        }
    event_key = ["security_id", "ex_date", "action_type"]
    economic_terms = ["price_factor", "volume_factor", "cash_amount", "currency"]
    ranked = frame.sort(
        [
            *event_key,
            "is_delisted_source",
            "known_at",
            "provider_symbol",
        ],
        descending=[False, False, False, False, False, False],
    )
    semantic = ranked.unique(
        subset=[*event_key, *economic_terms],
        keep="first",
        maintain_order=True,
    )
    exact_removed = frame.height - semantic.height
    conflicts = (
        semantic.group_by(event_key)
        .agg(
            pl.len().alias("_variants"),
            pl.col("provider_symbol").alias("_provider_symbols"),
            pl.col("price_factor").alias("_price_factors"),
            pl.col("volume_factor").alias("_volume_factors"),
            pl.col("cash_amount").alias("_cash_amounts"),
        )
        .filter(pl.col("_variants") > 1)
    )
    conflict_keys = conflicts.select(event_key)
    conflict_rows = (
        semantic.join(conflict_keys, on=event_key, how="semi")
        if conflicts.height
        else semantic.head(0)
    )
    clean = (
        semantic.join(conflict_keys, on=event_key, how="anti")
        if conflicts.height
        else semantic
    )
    clean = clean.drop("is_delisted_source")
    result = validate_corporate_actions(clean) if clean.height else None
    return result, {
        "exact_alias_rows_removed": exact_removed,
        "conflict_groups_dropped": conflicts.height,
        "conflict_rows_dropped": conflict_rows.height,
        "conflict_security_ids": sorted(conflict_rows["security_id"].unique().to_list()),
        "conflict_examples": conflicts.head(20).to_dicts(),
    }
