"""Prediction-table contract and coverage gates shared by every model family."""

from __future__ import annotations

from typing import Any

import polars as pl

from facdigger.data.contracts import DataContractError

PREDICTION_REQUIRED = {
    "security_id",
    "symbol",
    "asof_date",
    "score_raw",
    "score_neutralized",
    "target",
    "split",
    "model_id",
    "checkpoint_hash",
    "dataset_id",
    "eligible",
    "industry_code",
    "log_float_market_cap",
}


def validate_predictions(frame: pl.DataFrame) -> pl.DataFrame:
    missing = sorted(PREDICTION_REQUIRED - set(frame.columns))
    if missing:
        raise DataContractError(f"predictions is missing required columns: {missing}")
    if frame.schema["asof_date"] == pl.String:
        frame = frame.with_columns(pl.col("asof_date").str.to_date(strict=False))
    elif frame.schema["asof_date"] != pl.Date:
        frame = frame.with_columns(pl.col("asof_date").cast(pl.Date, strict=False))
    required_non_null = [
        "security_id",
        "symbol",
        "asof_date",
        "score_raw",
        "target",
        "split",
        "model_id",
        "checkpoint_hash",
        "dataset_id",
        "eligible",
    ]
    nulls = {column: frame[column].null_count() for column in required_non_null}
    bad_nulls = {column: count for column, count in nulls.items() if count}
    if bad_nulls:
        raise DataContractError(f"predictions has nulls in required fields: {bad_nulls}")
    duplicates = (
        frame.group_by("model_id", "security_id", "asof_date").len().filter(pl.col("len") > 1)
    )
    if duplicates.height:
        raise DataContractError("predictions has duplicate model/security/date keys")
    invalid = frame.filter(
        ~pl.col("score_raw").is_finite()
        | ~pl.col("target").is_finite()
        | (pl.col("score_neutralized").is_not_null() & ~pl.col("score_neutralized").is_finite())
    )
    if invalid.height:
        raise DataContractError(f"predictions has {invalid.height} non-finite score/target rows")
    if frame["model_id"].n_unique() != 1 or frame["dataset_id"].n_unique() != 1:
        raise DataContractError("one predictions table must contain one model_id and dataset_id")
    return frame.sort(["asof_date", "security_id"])


def prediction_coverage(
    predictions: pl.DataFrame,
    sample_index: pl.DataFrame,
    *,
    split: str,
    minimum: float,
) -> dict[str, Any]:
    expected = sample_index.filter(pl.col("split") == split).select("security_id", "asof_date")
    actual = predictions.filter(pl.col("split") == split).select("security_id", "asof_date")
    unexpected = actual.join(expected, on=["security_id", "asof_date"], how="anti").height
    missing = expected.join(actual, on=["security_id", "asof_date"], how="anti").height
    expected_rows = expected.height
    ratio = actual.height / expected_rows if expected_rows else 0.0
    report = {
        "split": split,
        "expected_rows": expected_rows,
        "prediction_rows": actual.height,
        "missing_rows": missing,
        "unexpected_rows": unexpected,
        "coverage": ratio,
        "minimum_required": minimum,
    }
    if unexpected or ratio < minimum:
        raise DataContractError(f"prediction coverage gate failed: {report}")
    return report
