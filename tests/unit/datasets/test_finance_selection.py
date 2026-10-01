from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest
from pydantic import ValidationError

from facdigger.data.config import FinanceSelectionConfig
from facdigger.data.contracts import DataContractError
from facdigger.datasets.splits import (
    build_finance_selection_indices,
    split_supervised_training_index,
)


def selection_rows():
    calendar = [date(2020, 1, 1) + timedelta(days=i) for i in range(100)]
    rows = []
    for index in range(10, 80):
        for security in ["a", "b"]:
            rows.append(
                {
                    "sample_id": f"{security}|{calendar[index]}",
                    "security_id": security,
                    "asof_date": calendar[index],
                    "label_end": calendar[index + 3],
                    "label_end_1": calendar[index + 1],
                    "label_end_5": calendar[index + 2],
                    "label_end_20": calendar[index + 3],
                    "split": "train",
                }
            )
    return pl.DataFrame(rows), calendar


def test_finance_plan_preserves_supervised_support_and_purges_rows_before_selecting_dates():
    rows, calendar = selection_rows()
    config = FinanceSelectionConfig(
        supervised_selection_fraction=0.2,
        probe_fit_dates=5,
        probe_selection_dates=4,
    )
    _, _, initial = build_finance_selection_indices(rows, calendar, config)
    p_start = date.fromisoformat(initial["p_start"])
    s_start = date.fromisoformat(initial["s_start"])
    rows = rows.with_columns(
        pl.when((pl.col("security_id") == "b") & (pl.col("asof_date") == p_start))
        .then(pl.lit(s_start))
        .when((pl.col("security_id") == "b") & (pl.col("asof_date") == p_start - timedelta(days=4)))
        .then(pl.lit(p_start))
        .otherwise(pl.col("label_end"))
        .alias("label_end")
    )
    supervised, probe, plan = build_finance_selection_indices(rows, calendar, config)
    expected, _ = split_supervised_training_index(rows, selection_fraction=0.2)
    assert supervised.equals(expected)
    assert plan["p_start"] == initial["p_start"]
    assert plan["a_end"] == (p_start - timedelta(days=1)).isoformat()
    assert probe.filter(pl.col("split") == "probe_fit")["label_end"].max() < p_start
    selected = probe.filter(pl.col("split") == "probe_selection")
    assert selected["label_end"].max() < s_start
    assert selected.filter(
        (pl.col("security_id") == "b") & (pl.col("asof_date") == p_start)
    ).is_empty()
    assert plan["probe"]["fit_dates"] == 5
    assert plan["probe"]["selection_dates"] == 4
    assert plan["scaler_fit_end"] == plan["a_end"]


@pytest.mark.parametrize("field,value", [("probe_fit_dates", 80), ("probe_selection_dates", 80)])
def test_finance_plan_never_shortens_requested_probe_dates(field, value):
    rows, calendar = selection_rows()
    config = FinanceSelectionConfig(**{field: value})
    with pytest.raises(DataContractError, match="dates"):
        build_finance_selection_indices(rows, calendar, config)


def test_finance_plan_rejects_label_horizon_beyond_purged_boundary():
    rows, calendar = selection_rows()
    rows = rows.with_columns((pl.col("label_end") + timedelta(days=1)).alias("label_end_20"))
    with pytest.raises(DataContractError, match="label_end"):
        build_finance_selection_indices(
            rows,
            calendar,
            FinanceSelectionConfig(
                probe_fit_dates=5,
                probe_selection_dates=4,
            ),
        )


@pytest.mark.parametrize(
    "field,value",
    [("probe_fit_dates", True), ("probe_selection_dates", 2.5), ("future_horizon", 20)],
)
def test_finance_selection_config_rejects_invalid_protocol(field, value):
    with pytest.raises(ValidationError):
        FinanceSelectionConfig(**{field: value})


def test_finance_experiment_settings_assert_snapshot_values_without_changing_plan():
    config = FinanceSelectionConfig()
    config.validate_expectations(supervised_selection_fraction=0.15)
    config.validate_expectations(probe_fit_dates=60, probe_selection_dates=20, future_horizon=5)
    with pytest.raises(DataContractError, match="probe_fit_dates"):
        config.validate_expectations(probe_fit_dates=61)
    with pytest.raises(DataContractError, match="Unknown"):
        config.validate_expectations(selection_fraction=0.15)
    assert config.probe_fit_dates == 60
