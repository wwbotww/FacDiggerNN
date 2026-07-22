"""Leakage-safe seven-channel daily price/volume features."""

from __future__ import annotations

import polars as pl

from facdigger.data.config import DEFAULT_CHANNELS


def _finite_or_null(column: str) -> pl.Expr:
    value = pl.col(column)
    return pl.when(value.is_finite()).then(value).otherwise(None).alias(column)


def build_price_volume_features(
    bars: pl.DataFrame,
    universe: pl.DataFrame,
) -> pl.DataFrame:
    """Build features on the full security-session grid supplied by the universe table."""

    panel = (
        universe.select("security_id", "symbol", "trade_date")
        .join(
            bars.select(
                "security_id",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "dollar_volume",
                "adj_factor",
            ),
            on=["security_id", "trade_date"],
            how="left",
        )
        .sort(["security_id", "trade_date"])
        .with_columns(
            (pl.col("open") * pl.col("adj_factor")).alias("_adj_open"),
            (pl.col("high") * pl.col("adj_factor")).alias("_adj_high"),
            (pl.col("low") * pl.col("adj_factor")).alias("_adj_low"),
            (pl.col("close") * pl.col("adj_factor")).alias("_adj_close"),
            pl.col("dollar_volume")
            .rolling_median(window_size=20, min_samples=20)
            .over("security_id")
            .alias("_dv_median20"),
            pl.col("dollar_volume")
            .rolling_quantile(0.25, window_size=20, min_samples=20)
            .over("security_id")
            .alias("_dv_q25_20"),
            pl.col("dollar_volume")
            .rolling_quantile(0.75, window_size=20, min_samples=20)
            .over("security_id")
            .alias("_dv_q75_20"),
        )
        .with_columns(
            pl.col("_adj_close").shift(1).over("security_id").alias("_previous_close"),
            pl.col("volume").log1p().alias("_log_volume"),
        )
        .with_columns(
            (pl.col("_adj_close") / pl.col("_previous_close")).log().alias("r_close"),
            (pl.col("_adj_open") / pl.col("_previous_close")).log().alias("r_gap"),
            (pl.col("_adj_close") / pl.col("_adj_open")).log().alias("r_intraday"),
            (pl.col("_adj_high") / pl.col("_adj_low")).log().alias("range"),
            (pl.col("_log_volume") - pl.col("_log_volume").shift(1).over("security_id")).alias(
                "dlog_volume"
            ),
            (
                (pl.col("dollar_volume") - pl.col("_dv_median20"))
                * 1.349
                / (pl.col("_dv_q75_20") - pl.col("_dv_q25_20"))
            ).alias("dollar_volume_z20"),
        )
        .with_columns(
            pl.col("r_close")
            .rolling_std(window_size=20, min_samples=20, ddof=0)
            .over("security_id")
            .alias("vol20")
        )
        .with_columns([_finite_or_null(column) for column in DEFAULT_CHANNELS])
        .with_columns(
            [
                pl.col(column).is_not_null().alias(f"observed_{column}")
                for column in DEFAULT_CHANNELS
            ]
        )
    )
    return panel.select(
        "security_id",
        "symbol",
        "trade_date",
        *DEFAULT_CHANNELS,
        *[f"observed_{column}" for column in DEFAULT_CHANNELS],
    )
