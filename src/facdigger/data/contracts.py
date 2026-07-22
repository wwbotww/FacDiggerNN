"""Strict table contracts for point-in-time US-equity data."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl


class DataContractError(ValueError):
    """Raised when an input table cannot safely enter a dataset snapshot."""


@dataclass(frozen=True)
class DataBundle:
    bars: pl.DataFrame
    universe: pl.DataFrame
    corporate_actions: pl.DataFrame | None = None
    delistings: pl.DataFrame | None = None


BARS_REQUIRED = {
    "security_id",
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dollar_volume",
    "adj_factor",
    "source_revision",
}

UNIVERSE_REQUIRED = {
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
}

CORPORATE_ACTIONS_REQUIRED = {
    "security_id",
    "ex_date",
    "action_type",
    "price_factor",
    "volume_factor",
    "cash_amount",
    "known_at",
    "source_revision",
}

DELISTINGS_REQUIRED = {
    "security_id",
    "delist_date",
    "last_trade_date",
    "delisting_return",
    "terminal_value",
    "known_at",
    "source_revision",
}


def _require_columns(frame: pl.DataFrame, required: set[str], table: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataContractError(f"{table} is missing required columns: {missing}")


def _require_unique(frame: pl.DataFrame, columns: list[str], table: str) -> None:
    duplicates = frame.group_by(columns).len().filter(pl.col("len") > 1)
    if duplicates.height:
        preview = duplicates.head(5).to_dicts()
        raise DataContractError(f"{table} has duplicate keys {columns}: {preview}")


def _require_no_nulls(frame: pl.DataFrame, columns: list[str], table: str) -> None:
    counts = {column: frame[column].null_count() for column in columns}
    bad = {column: count for column, count in counts.items() if count}
    if bad:
        raise DataContractError(f"{table} has nulls in key fields: {bad}")


def _cast_date(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    if frame.schema[column] == pl.Date:
        return frame
    if frame.schema[column] == pl.String:
        return frame.with_columns(pl.col(column).str.to_date(strict=False))
    return frame.with_columns(pl.col(column).cast(pl.Date, strict=False))


def validate_bars(frame: pl.DataFrame) -> pl.DataFrame:
    _require_columns(frame, BARS_REQUIRED, "bars_daily")
    frame = _cast_date(frame, "trade_date").sort(["security_id", "trade_date"])
    _require_no_nulls(
        frame,
        ["security_id", "symbol", "trade_date", "source_revision"],
        "bars_daily",
    )
    _require_unique(frame, ["security_id", "trade_date"], "bars_daily")
    invalid_prices = frame.filter(
        (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0)
        | (pl.col("high") < pl.max_horizontal("open", "close", "low"))
        | (pl.col("low") > pl.min_horizontal("open", "close", "high"))
        | (pl.col("volume") < 0)
        | (pl.col("dollar_volume") < 0)
        | (pl.col("adj_factor") <= 0)
        | ~pl.col("open").is_finite()
        | ~pl.col("high").is_finite()
        | ~pl.col("low").is_finite()
        | ~pl.col("close").is_finite()
        | ~pl.col("volume").is_finite()
        | ~pl.col("dollar_volume").is_finite()
        | ~pl.col("adj_factor").is_finite()
    )
    if invalid_prices.height:
        raise DataContractError(
            f"bars_daily has {invalid_prices.height} invalid OHLCV/adjustment rows"
        )
    return frame


def validate_universe(frame: pl.DataFrame) -> pl.DataFrame:
    _require_columns(frame, UNIVERSE_REQUIRED, "universe_daily")
    frame = _cast_date(frame, "trade_date").sort(["security_id", "trade_date"])
    _require_no_nulls(
        frame,
        [
            "security_id",
            "symbol",
            "trade_date",
            "exchange",
            "security_type",
            "is_primary_listing",
            "is_listed",
            "is_delisted",
            "is_halted",
            "eligible",
        ],
        "universe_daily",
    )
    _require_unique(frame, ["security_id", "trade_date"], "universe_daily")
    inconsistent = frame.filter(
        pl.col("eligible")
        & (
            ~pl.col("is_listed")
            | pl.col("is_delisted")
            | pl.col("is_halted")
            | ~pl.col("is_primary_listing")
            | (pl.col("security_type") != "common_stock")
        )
    )
    if inconsistent.height:
        raise DataContractError(
            f"universe_daily has {inconsistent.height} rows eligible despite ineligible state"
        )
    return frame


def validate_corporate_actions(frame: pl.DataFrame) -> pl.DataFrame:
    _require_columns(frame, CORPORATE_ACTIONS_REQUIRED, "corporate_actions")
    frame = _cast_date(_cast_date(frame, "ex_date"), "known_at").sort(["security_id", "ex_date"])
    _require_no_nulls(
        frame,
        ["security_id", "ex_date", "action_type", "known_at", "source_revision"],
        "corporate_actions",
    )
    invalid = frame.filter(
        (pl.col("price_factor") <= 0)
        | (pl.col("volume_factor") <= 0)
        | (pl.col("cash_amount") < 0)
        | (pl.col("known_at") > pl.col("ex_date"))
    )
    if invalid.height:
        raise DataContractError(
            f"corporate_actions has {invalid.height} invalid or non-point-in-time rows"
        )
    return frame


def validate_delistings(frame: pl.DataFrame) -> pl.DataFrame:
    _require_columns(frame, DELISTINGS_REQUIRED, "delistings")
    frame = _cast_date(_cast_date(frame, "delist_date"), "last_trade_date").sort(
        ["security_id", "delist_date"]
    )
    _require_no_nulls(
        frame,
        ["security_id", "delist_date", "last_trade_date", "known_at", "source_revision"],
        "delistings",
    )
    _require_unique(frame, ["security_id"], "delistings")
    invalid = frame.filter(
        (pl.col("last_trade_date") > pl.col("delist_date"))
        | (pl.col("delisting_return").is_null() & pl.col("terminal_value").is_null())
        | (pl.col("delisting_return") < -1)
        | (pl.col("terminal_value") < 0)
    )
    if invalid.height:
        raise DataContractError(
            f"delistings has {invalid.height} rows without a valid terminal return/value"
        )
    return frame


def table_audit(frame: pl.DataFrame, date_column: str) -> dict[str, Any]:
    minimum = frame[date_column].min()
    maximum = frame[date_column].max()
    return {
        "rows": frame.height,
        "securities": frame["security_id"].n_unique(),
        "date_min": minimum.isoformat() if minimum is not None else None,
        "date_max": maximum.isoformat() if maximum is not None else None,
        "null_counts": {
            column: frame[column].null_count()
            for column in frame.columns
            if frame[column].null_count()
        },
    }


VALIDATORS: dict[str, Callable[[pl.DataFrame], pl.DataFrame]] = {
    "bars": validate_bars,
    "universe": validate_universe,
    "corporate_actions": validate_corporate_actions,
    "delistings": validate_delistings,
}
