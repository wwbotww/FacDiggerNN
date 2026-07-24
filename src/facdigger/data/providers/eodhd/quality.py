"""EODHD-specific source-quality controls before the provider-neutral boundary."""

from __future__ import annotations

from typing import Any

import polars as pl

from facdigger.data.contracts import DataContractError


def filter_to_regular_sessions(
    bars: pl.DataFrame,
    calendar: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if bars.schema["trade_date"] != pl.Date:
        bars = bars.with_columns(pl.col("trade_date").cast(pl.String).str.to_date(strict=False))
    allowed = calendar.select("trade_date").unique()
    rejected = bars.join(allowed, on="trade_date", how="anti")
    filtered = bars.join(allowed, on="trade_date", how="inner")
    examples = (
        rejected.group_by("trade_date")
        .agg(pl.len().alias("bars"))
        .sort("trade_date")
        .head(20)
        .to_dicts()
    )
    return filtered, {
        "off_calendar_rows_dropped": rejected.height,
        "off_calendar_dates_dropped": rejected["trade_date"].n_unique(),
        "off_calendar_securities": rejected["security_id"].n_unique(),
        "off_calendar_examples": examples,
    }


def quarantine_suspicious_identities(
    bars: pl.DataFrame,
    calendar: pl.DataFrame,
    *,
    max_adjusted_price_ratio: float,
    max_alias_overlap_relative_diff: float,
    max_quarantined_security_fraction: float,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Remove identities whose adjusted-price history cannot represent one security."""

    adjusted = bars.select(
        "security_id",
        "provider_symbol",
        "trade_date",
        (pl.col("close") * pl.col("adj_factor")).alias("_adjusted_price"),
    )
    alias_conflicts = (
        adjusted.group_by(["security_id", "trade_date"])
        .agg(
            pl.col("provider_symbol").n_unique().alias("_aliases"),
            pl.col("_adjusted_price").min().alias("_minimum"),
            pl.col("_adjusted_price").max().alias("_maximum"),
        )
        .filter(
            (pl.col("_aliases") > 1)
            & (
                (pl.col("_maximum") / pl.col("_minimum"))
                > (1.0 + max_alias_overlap_relative_diff)
            )
        )
    )

    indexed = calendar.with_row_index("_session_index")
    sequential = (
        adjusted.join(indexed, on="trade_date", how="inner", validate="m:1")
        .sort(["security_id", "trade_date"])
        .with_columns(
            pl.col("_adjusted_price").shift(1).over("security_id").alias("_previous_price"),
            pl.col("_session_index").shift(1).over("security_id").alias("_previous_session"),
        )
        .with_columns(
            (pl.col("_adjusted_price") / pl.col("_previous_price")).alias("_price_ratio")
        )
        .filter(
            (pl.col("_session_index") - pl.col("_previous_session") == 1)
            & (
                (pl.col("_price_ratio") > max_adjusted_price_ratio)
                | (pl.col("_price_ratio") < 1.0 / max_adjusted_price_ratio)
            )
        )
    )
    quarantined = sorted(
        set(alias_conflicts["security_id"].to_list())
        | set(sequential["security_id"].to_list())
    )
    total = bars["security_id"].n_unique()
    fraction = len(quarantined) / total if total else 0.0
    if fraction > max_quarantined_security_fraction:
        raise DataContractError(
            "EODHD quality gate would quarantine "
            f"{len(quarantined)}/{total} securities ({fraction:.2%}), above configured "
            f"{max_quarantined_security_fraction:.2%}; inspect the provider payload before "
            "accepting systemic data loss"
        )
    result = bars.filter(~pl.col("security_id").is_in(quarantined))
    return result, {
        "quarantined_securities": len(quarantined),
        "quarantined_security_fraction": fraction,
        "quarantined_bar_rows": bars.height - result.height,
        "quarantined_security_ids": quarantined,
        "alias_overlap_conflict_groups": alias_conflicts.height,
        "extreme_consecutive_return_rows": sequential.height,
        "alias_overlap_examples": alias_conflicts.head(20).to_dicts(),
        "extreme_return_examples": sequential.select(
            "security_id",
            "provider_symbol",
            "trade_date",
            "_adjusted_price",
            "_previous_price",
            "_price_ratio",
        )
        .head(20)
        .to_dicts(),
    }


def assert_historical_ingestion_quality(
    bars: pl.DataFrame,
    calendar: pl.DataFrame,
    *,
    max_adjusted_price_ratio: float,
) -> dict[str, Any]:
    """Fail closed if a P0 invariant survives standardization."""

    unexpected_dates = bars.select("trade_date").unique().join(
        calendar.select("trade_date").unique(),
        on="trade_date",
        how="anti",
    )
    observed_dates = bars.select("trade_date").unique()
    missing_dates = calendar.join(observed_dates, on="trade_date", how="anti")
    indexed = calendar.with_row_index("_session_index")
    extreme = (
        bars.with_columns((pl.col("close") * pl.col("adj_factor")).alias("_adjusted_price"))
        .join(indexed, on="trade_date", how="inner", validate="m:1")
        .sort(["security_id", "trade_date"])
        .with_columns(
            pl.col("_adjusted_price").shift(1).over("security_id").alias("_previous_price"),
            pl.col("_session_index").shift(1).over("security_id").alias("_previous_session"),
        )
        .with_columns(
            (pl.col("_adjusted_price") / pl.col("_previous_price")).alias("_price_ratio")
        )
        .filter(
            (pl.col("_session_index") - pl.col("_previous_session") == 1)
            & (
                (pl.col("_price_ratio") > max_adjusted_price_ratio)
                | (pl.col("_price_ratio") < 1.0 / max_adjusted_price_ratio)
            )
        )
    )
    if unexpected_dates.height or missing_dates.height or extreme.height:
        raise DataContractError(
            "EODHD historical quality gate failed: "
            f"off_calendar_dates={unexpected_dates.height}, "
            f"missing_market_sessions={missing_dates.height}, "
            f"extreme_consecutive_returns={extreme.height}"
        )
    return {
        "status": "passed",
        "market_sessions": calendar.height,
        "observed_market_sessions": observed_dates.height,
        "off_calendar_dates": 0,
        "missing_market_sessions": 0,
        "remaining_extreme_consecutive_returns": 0,
    }
