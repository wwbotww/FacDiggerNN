"""Five-session forward excess-return labels with delisting handling."""

from __future__ import annotations

import polars as pl

from facdigger.data.contracts import DataContractError


def _delisting_terms(
    bars: pl.DataFrame,
    delistings: pl.DataFrame | None,
) -> pl.DataFrame | None:
    if delistings is None:
        return None
    last_prices = bars.select(
        "security_id",
        pl.col("trade_date").alias("last_trade_date"),
        (pl.col("close") * pl.col("adj_factor")).alias("last_adjusted_close"),
    )
    terms = delistings.join(
        last_prices,
        on=["security_id", "last_trade_date"],
        how="left",
        validate="1:1",
    ).with_columns(
        pl.coalesce(
            "terminal_value",
            pl.col("last_adjusted_close") * (1 + pl.col("delisting_return")),
        ).alias("_terminal_value")
    )
    missing = terms.filter(pl.col("_terminal_value").is_null())
    if missing.height:
        raise DataContractError(
            f"Cannot derive terminal value for {missing.height} delisted securities"
        )
    return terms.select("security_id", "delist_date", "_terminal_value")


def build_forward_excess_return_labels(
    bars: pl.DataFrame,
    universe: pl.DataFrame,
    delistings: pl.DataFrame | None = None,
    execution_lag: int = 1,
    horizon: int = 5,
) -> pl.DataFrame:
    if execution_lag < 1 or horizon < execution_lag:
        raise ValueError("Require horizon >= execution_lag >= 1")

    calendar = (
        universe.select("trade_date")
        .unique()
        .sort("trade_date")
        .with_columns(
            pl.col("trade_date").shift(-execution_lag).alias("label_start"),
            pl.col("trade_date").shift(-horizon).alias("label_end"),
        )
    )
    adjusted = bars.select(
        "security_id",
        "trade_date",
        (pl.col("open") * pl.col("adj_factor")).alias("_adj_open"),
        (pl.col("close") * pl.col("adj_factor")).alias("_adj_close"),
    )
    entry = adjusted.select(
        "security_id",
        pl.col("trade_date").alias("label_start"),
        pl.col("_adj_open").alias("_entry_value"),
    )
    exit_values = adjusted.select(
        "security_id",
        pl.col("trade_date").alias("label_end"),
        pl.col("_adj_close").alias("_normal_exit"),
    )
    panel = (
        universe.select("security_id", "trade_date", "eligible")
        .join(calendar, on="trade_date", how="left", validate="m:1")
        .join(
            entry,
            on=["security_id", "label_start"],
            how="left",
            validate="m:1",
        )
        .join(
            exit_values,
            on=["security_id", "label_end"],
            how="left",
            validate="m:1",
        )
        .sort(["security_id", "trade_date"])
    )

    terms = _delisting_terms(bars, delistings)
    if terms is None:
        panel = panel.with_columns(
            pl.lit(None, dtype=pl.Date).alias("delist_date"),
            pl.lit(None, dtype=pl.Float64).alias("_terminal_value"),
        )
    else:
        panel = panel.join(terms, on="security_id", how="left", validate="m:1")

    panel = panel.with_columns(
        (
            (pl.col("delist_date") > pl.col("trade_date"))
            & (pl.col("delist_date") <= pl.col("label_end"))
        )
        .fill_null(False)
        .alias("crosses_delisting")
    ).with_columns(
        pl.when(pl.col("crosses_delisting"))
        .then(pl.col("_terminal_value"))
        .otherwise(pl.col("_normal_exit"))
        .alias("_exit_value")
    )
    panel = (
        panel.with_columns(
            (pl.col("_exit_value") / pl.col("_entry_value")).log().alias("raw_return")
        )
        .with_columns(
            pl.when(pl.col("eligible"))
            .then(pl.col("raw_return"))
            .otherwise(None)
            .mean()
            .over("trade_date")
            .alias("benchmark_return")
        )
        .with_columns((pl.col("raw_return") - pl.col("benchmark_return")).alias("target"))
    )
    return panel.select(
        pl.col("trade_date").alias("asof_date"),
        "security_id",
        "label_start",
        "label_end",
        "raw_return",
        "benchmark_return",
        "target",
        "crosses_delisting",
    )
