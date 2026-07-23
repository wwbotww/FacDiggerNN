"""Build the immutable sample index without materializing 3D windows."""

from __future__ import annotations

import polars as pl

from facdigger.data.contracts import DataContractError


def build_sample_index(
    features: pl.DataFrame,
    labels_with_split: pl.DataFrame,
    universe: pl.DataFrame,
    context_length: int,
) -> pl.DataFrame:
    if context_length < 1:
        raise ValueError("context_length must be positive")
    feature_bounds = features.sort(["security_id", "trade_date"]).with_columns(
        pl.col("trade_date").shift(context_length - 1).over("security_id").alias("feature_start"),
        pl.col("trade_date").alias("feature_end"),
    )
    samples = (
        feature_bounds.select("security_id", "trade_date", "feature_start", "feature_end")
        .join(
            labels_with_split,
            left_on=["security_id", "trade_date"],
            right_on=["security_id", "asof_date"],
            how="left",
            validate="1:1",
        )
        .join(
            universe.select("security_id", "trade_date", "symbol", "eligible"),
            on=["security_id", "trade_date"],
            how="left",
            validate="1:1",
        )
        .filter(
            pl.col("feature_start").is_not_null()
            & pl.col("target").is_not_null()
            & pl.col("split").is_not_null()
            & pl.col("eligible")
        )
        .with_columns(
            pl.concat_str(
                "security_id",
                pl.col("trade_date").dt.strftime("%Y-%m-%d"),
                separator="|",
            ).alias("sample_id")
        )
        .select(
            "sample_id",
            "security_id",
            "symbol",
            pl.col("trade_date").alias("asof_date"),
            "feature_start",
            "feature_end",
            "label_start",
            "label_end",
            "split",
            "target",
            "raw_return",
            "benchmark_return",
            "crosses_delisting",
        )
        .sort(["asof_date", "security_id"])
    )
    if samples["sample_id"].n_unique() != samples.height:
        raise DataContractError("sample_index contains duplicate sample_id values")
    temporal_violations = samples.filter(
        (pl.col("feature_end") > pl.col("asof_date"))
        | (pl.col("asof_date") >= pl.col("label_start"))
    )
    if temporal_violations.height:
        raise DataContractError(
            f"sample_index contains {temporal_violations.height} look-ahead violations"
        )
    return samples


def build_inference_index(
    features: pl.DataFrame,
    universe: pl.DataFrame,
    context_length: int,
) -> pl.DataFrame:
    """Build a target-free index for every eligible point-in-time feature window."""

    if context_length < 1:
        raise ValueError("context_length must be positive")
    index = (
        features.sort(["security_id", "trade_date"])
        .with_columns(
            pl.col("trade_date")
            .shift(context_length - 1)
            .over("security_id")
            .alias("feature_start"),
            pl.col("trade_date").alias("feature_end"),
        )
        .select("security_id", "trade_date", "feature_start", "feature_end")
        .join(
            universe.select(
                "security_id",
                "trade_date",
                "symbol",
                "eligible",
                "industry_code",
                "float_market_cap",
            ),
            on=["security_id", "trade_date"],
            how="left",
            validate="1:1",
        )
        .filter(pl.col("feature_start").is_not_null() & pl.col("eligible"))
        .with_columns(
            pl.concat_str(
                "security_id",
                pl.col("trade_date").dt.strftime("%Y-%m-%d"),
                separator="|",
            ).alias("sample_id"),
            pl.when(pl.col("float_market_cap") > 0)
            .then(pl.col("float_market_cap").log())
            .otherwise(None)
            .alias("log_float_market_cap"),
        )
        .select(
            "sample_id",
            "security_id",
            "symbol",
            pl.col("trade_date").alias("asof_date"),
            "feature_start",
            "feature_end",
            "eligible",
            "industry_code",
            "float_market_cap",
            "log_float_market_cap",
        )
        .sort(["asof_date", "security_id"])
    )
    if index["sample_id"].n_unique() != index.height:
        raise DataContractError("inference_index contains duplicate sample_id values")
    if index.filter(pl.col("feature_end") != pl.col("asof_date")).height:
        raise DataContractError("inference_index contains invalid feature window bounds")
    return index
