"""Chronological split assignment with label purge and session embargo."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import polars as pl

from facdigger.data.config import FinanceSelectionConfig, SplitConfig
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


def split_supervised_training_index(
    sample_index: pl.DataFrame,
    *,
    selection_fraction: float,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Create fit/selection partitions wholly inside the official train split."""

    if not 0 < selection_fraction < 0.5:
        raise ValueError("selection_fraction must satisfy 0 < value < 0.5")
    official_train = sample_index.filter(pl.col("split") == "train").sort(
        ["asof_date", "security_id"]
    )
    dates = official_train["asof_date"].unique().sort().to_list()
    if len(dates) < 3:
        raise DataContractError("supervised inner selection requires at least three train dates")
    selection_count = max(1, math.ceil(len(dates) * selection_fraction))
    selection_count = min(selection_count, len(dates) - 2)
    selection_dates = dates[-selection_count:]
    selection_start = selection_dates[0]
    selection = official_train.filter(pl.col("asof_date").is_in(selection_dates))
    fit = official_train.filter(pl.col("label_end") < selection_start)
    if fit.is_empty() or selection.is_empty():
        raise DataContractError("supervised inner selection produced an empty partition")
    purged_rows = official_train.height - fit.height - selection.height
    protocol_index = pl.concat(
        [
            fit.with_columns(pl.lit("train_fit").alias("split")),
            selection.with_columns(pl.lit("inner_selection").alias("split")),
            sample_index.filter(pl.col("split") != "train"),
        ],
        how="vertical",
    ).sort(["asof_date", "security_id"])
    outer_valid = sample_index.filter(pl.col("split") == "valid")
    outer_test = sample_index.filter(pl.col("split") == "test")
    audit = {
        "schema_version": 1,
        "source_split": "train",
        "selection_policy": "chronological_tail_with_label_overlap_purge",
        "selection_fraction": selection_fraction,
        "official_train_rows": official_train.height,
        "fit_rows": fit.height,
        "selection_rows": selection.height,
        "purged_rows": purged_rows,
        "fit_min_asof_date": fit["asof_date"].min(),
        "fit_max_asof_date": fit["asof_date"].max(),
        "fit_max_label_end": fit["label_end"].max(),
        "selection_min_asof_date": selection_start,
        "selection_max_asof_date": selection["asof_date"].max(),
        "outer_validation_rows_used_for_training": 0,
        "outer_validation_rows_used_for_checkpoint_selection": 0,
        "outer_test_rows_used": 0,
        "outer_validation_rows": outer_valid.height,
        "outer_test_rows": outer_test.height,
    }
    if audit["fit_max_label_end"] >= audit["selection_min_asof_date"]:
        raise DataContractError("fit labels overlap inner selection dates")
    return protocol_index, audit


def build_finance_selection_indices(
    sample_index: pl.DataFrame,
    calendar: list[date],
    config: FinanceSelectionConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Plan isolated SSL, encoder selection and supervised selection on Train.

    The supervised support is unchanged. Probe date selection always starts
    from row-purged candidates so a same-date security cannot cross a boundary.
    All returned plan values are JSON-native for immutable snapshot comparison.
    """
    required = {"label_end", "label_end_1", "label_end_5", "label_end_20"}
    if not required.issubset(sample_index.columns):
        raise DataContractError("Finance selection requires all 1/5/20 label_end columns")
    invalid = sample_index.filter(
        pl.any_horizontal(
            *[pl.col(column).is_null() for column in required],
            *[pl.col(f"label_end_{horizon}") > pl.col("label_end") for horizon in (1, 5, 20)],
        )
    )
    if invalid.height:
        raise DataContractError("Finance label_end does not bound every supervised horizon")
    supervised, audit = split_supervised_training_index(
        sample_index, selection_fraction=config.supervised_selection_fraction,
    )
    fit = supervised.filter(pl.col("split") == "train_fit")
    fit_dates = fit["asof_date"].unique().sort().to_list()
    if len(fit_dates) < config.probe_selection_dates:
        raise DataContractError("Finance selection has too few purged dates for probe selection")
    selection_dates = fit_dates[-config.probe_selection_dates:]
    p_start = selection_dates[0]
    probe_selection = fit.filter(pl.col("asof_date").is_in(selection_dates)).with_columns(
        pl.lit("probe_selection").alias("split")
    )
    candidates = sample_index.filter(
        (pl.col("split") == "train") & (pl.col("label_end") < pl.lit(p_start))
    )
    candidate_dates = candidates["asof_date"].unique().sort().to_list()
    if len(candidate_dates) < config.probe_fit_dates:
        raise DataContractError("Finance selection has too few purged dates for probe fit")
    probe_fit = candidates.filter(
        pl.col("asof_date").is_in(candidate_dates[-config.probe_fit_dates:])
    ).with_columns(pl.lit("probe_fit").alias("split"))
    earlier_sessions = sorted({session for session in calendar if session < p_start})
    if not earlier_sessions or p_start not in calendar:
        raise DataContractError("Finance selection has no valid market calendar before probe")
    a_end = earlier_sessions[-1]
    s_start = audit["selection_min_asof_date"]
    if not a_end < p_start < s_start:
        raise DataContractError("Finance selection boundaries are not chronological")
    probe = pl.concat([probe_fit, probe_selection], how="vertical").sort(
        ["asof_date", "security_id"]
    )
    plan = {
        "policy": "isolated_pretraining_probe_and_supervised_selection",
        "parameters": config.model_dump(mode="json"),
        "a_end": a_end.isoformat(),
        "p_start": p_start.isoformat(),
        "s_start": s_start.isoformat(),
        "scaler_fit_end": a_end.isoformat(),
        "supervised": {
            key: value.isoformat() if isinstance(value, date) else value
            for key, value in audit.items()
        },
        "probe": {
            "fit_dates": probe_fit["asof_date"].n_unique(),
            "selection_dates": probe_selection["asof_date"].n_unique(),
            "fit_rows": probe_fit.height,
            "selection_rows": probe_selection.height,
            "fit_min_asof_date": probe_fit["asof_date"].min().isoformat(),
            "fit_max_asof_date": probe_fit["asof_date"].max().isoformat(),
            "fit_max_label_end": probe_fit["label_end"].max().isoformat(),
            "selection_min_asof_date": p_start.isoformat(),
            "selection_max_asof_date": probe_selection["asof_date"].max().isoformat(),
            "selection_max_label_end": probe_selection["label_end"].max().isoformat(),
            "outer_validation_rows_used": 0,
            "outer_test_rows_used": 0,
        },
    }
    return supervised, probe, plan
