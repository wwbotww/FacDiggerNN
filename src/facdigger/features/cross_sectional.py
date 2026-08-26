"""Point-in-time cross-sectional and market context features."""

from __future__ import annotations

import polars as pl

from facdigger.data.config import (
    DEFAULT_CHANNELS,
    MARKET_CONTEXT_CHANNELS,
    RANK_CHANNELS,
)
from facdigger.data.contracts import DataContractError


def _eligible_feature_panel(
    features: pl.DataFrame,
    universe: pl.DataFrame,
) -> pl.DataFrame:
    required = {"security_id", "trade_date", *DEFAULT_CHANNELS}
    missing = sorted(required - set(features.columns))
    if missing:
        raise DataContractError(
            f"price/volume features missing cross-sectional inputs: {missing}"
        )
    panel = features.join(
        universe.select("security_id", "trade_date", "eligible"),
        on=["security_id", "trade_date"],
        how="left",
        validate="1:1",
    )
    if panel["eligible"].null_count():
        raise DataContractError("feature rows have no matching universe eligibility")
    return panel


def append_cross_sectional_ranks(
    features: pl.DataFrame,
    universe: pl.DataFrame,
) -> pl.DataFrame:
    """Append daily eligible-universe percentile ranks mapped to ``[-1, 1]``."""

    panel = _eligible_feature_panel(features, universe)
    expressions: list[pl.Expr] = []
    observed_expressions: list[pl.Expr] = []
    temporary_columns: list[str] = []
    for channel, rank_channel in zip(DEFAULT_CHANNELS, RANK_CHANNELS, strict=True):
        eligible_value = f"_eligible_{channel}"
        eligible_count = f"_eligible_count_{channel}"
        eligible_rank = f"_eligible_rank_{channel}"
        temporary_columns.extend([eligible_value, eligible_count, eligible_rank])
        panel = panel.with_columns(
            pl.when(pl.col("eligible"))
            .then(pl.col(channel))
            .otherwise(None)
            .alias(eligible_value)
        ).with_columns(
            pl.col(eligible_value).count().over("trade_date").alias(eligible_count),
            pl.col(eligible_value)
            .rank(method="average")
            .over("trade_date")
            .alias(eligible_rank),
        )
        valid = pl.col(eligible_value).is_not_null() & (pl.col(eligible_count) >= 2)
        expressions.append(
            pl.when(valid)
            .then(
                2.0
                * (pl.col(eligible_rank) - 1.0)
                / (pl.col(eligible_count) - 1.0)
                - 1.0
            )
            .otherwise(None)
            .alias(rank_channel)
        )
        observed_expressions.append(valid.alias(f"observed_{rank_channel}"))
    return (
        panel.with_columns(expressions)
        .with_columns(observed_expressions)
        .drop("eligible", *temporary_columns)
        .sort(["security_id", "trade_date"])
    )


def build_market_context_features(
    features: pl.DataFrame,
    universe: pl.DataFrame,
) -> pl.DataFrame:
    """Build one six-channel market state row per trading date."""

    eligible = _eligible_feature_panel(features, universe).filter(pl.col("eligible"))
    if eligible.is_empty():
        raise DataContractError("cannot build market context without eligible feature rows")
    daily_location = eligible.group_by("trade_date").agg(
        pl.col("r_close").median().alias("market_return_median"),
        (
            (pl.col("r_close") > 0).cast(pl.Float64).mean() * 2.0 - 1.0
        ).alias("market_breadth"),
        pl.col("range").median().alias("market_range_median"),
        pl.col("dollar_volume_z20").median().alias("market_volume_activity"),
    )
    dispersion = (
        eligible.join(
            daily_location.select("trade_date", "market_return_median"),
            on="trade_date",
            how="left",
            validate="m:1",
        )
        .with_columns(
            (pl.col("r_close") - pl.col("market_return_median"))
            .abs()
            .alias("_return_deviation")
        )
        .group_by("trade_date")
        .agg(
            pl.col("_return_deviation")
            .median()
            .alias("market_return_dispersion")
        )
    )
    market = (
        daily_location.join(dispersion, on="trade_date", how="left", validate="1:1")
        .sort("trade_date")
        .with_columns(
            pl.col("market_return_median")
            .rolling_std(window_size=20, min_samples=20, ddof=0)
            .alias("market_vol20")
        )
    )
    return market.select(
        "trade_date",
        *MARKET_CONTEXT_CHANNELS,
        *[
            pl.col(channel).is_not_null().alias(f"observed_{channel}")
            for channel in MARKET_CONTEXT_CHANNELS
        ],
    )
