"""Chronological split assignment with label purge and session embargo."""

from __future__ import annotations

from datetime import date

import polars as pl

from facdigger.data.config import SplitConfig
from facdigger.data.contracts import DataContractError


def _first_session_after_embargo(
    calendar: list[date], boundary: date, embargo_sessions: int
) -> date | None:
    later = [session for session in calendar if session > boundary]
    if len(later) <= embargo_sessions:
        return None
    return later[embargo_sessions]


def assign_chronological_splits(
    labels: pl.DataFrame,
    calendar: list[date],
    config: SplitConfig,
) -> pl.DataFrame:
    ordered_calendar = sorted(set(calendar))
    valid_start = _first_session_after_embargo(
        ordered_calendar, config.train_end, config.embargo_sessions
    )
    test_start = _first_session_after_embargo(
        ordered_calendar, config.valid_end, config.embargo_sessions
    )
    if valid_start is None or test_start is None:
        raise DataContractError("Calendar does not extend far enough beyond split embargoes")

    split = (
        pl.when(pl.col("label_end") <= pl.lit(config.train_end))
        .then(pl.lit("train"))
        .when(
            (pl.col("asof_date") >= pl.lit(valid_start))
            & (pl.col("label_end") <= pl.lit(config.valid_end))
        )
        .then(pl.lit("valid"))
        .when(
            (pl.col("asof_date") >= pl.lit(test_start))
            & (pl.col("label_end") <= pl.lit(config.test_end))
        )
        .then(pl.lit("test"))
        .otherwise(None)
        .alias("split")
    )
    result = labels.with_columns(split)
    violations = result.filter(
        pl.col("label_start").is_not_null() & (pl.col("asof_date") >= pl.col("label_start"))
    )
    if violations.height:
        raise DataContractError(f"Look-ahead invariant failed for {violations.height} label rows")
    return result
