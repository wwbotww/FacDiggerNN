from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from facdigger.data.contracts import DataContractError
from facdigger.evaluation.contracts import prediction_coverage, validate_predictions
from facdigger.evaluation.metrics import (
    aggregate_ic,
    daily_information_coefficients,
    daily_quantile_portfolio,
)
from facdigger.evaluation.neutralization import neutralize_predictions


def metric_frame() -> pl.DataFrame:
    records = []
    for day_index in range(2):
        asof_date = date(2025, 1, 2) + timedelta(days=day_index)
        for security_index in range(10):
            score = float(security_index if day_index == 0 else 9 - security_index)
            records.append(
                {
                    "security_id": f"sec-{security_index}",
                    "symbol": f"S{security_index}",
                    "asof_date": asof_date,
                    "score_raw": score,
                    "score_neutralized": None,
                    "target": float(security_index),
                    "split": "valid",
                    "model_id": "model-1",
                    "checkpoint_hash": "abc",
                    "dataset_id": "dataset-1",
                    "eligible": True,
                    "industry_code": "A" if security_index < 5 else "B",
                    "log_float_market_cap": float(security_index + 10),
                }
            )
    return pl.DataFrame(records, schema_overrides={"score_neutralized": pl.Float64})


def test_ic_rank_ic_quantiles_turnover_and_cost_are_hand_checkable() -> None:
    frame = validate_predictions(metric_frame())
    daily_ic = daily_information_coefficients(frame)
    assert daily_ic["ic"].to_list() == pytest.approx([1.0, -1.0])
    assert daily_ic["rank_ic"].to_list() == pytest.approx([1.0, -1.0])
    aggregate = aggregate_ic(daily_ic)
    assert aggregate["rank_ic"]["mean"] == pytest.approx(0.0)

    portfolio = daily_quantile_portfolio(frame, costs_bps=[0, 10])
    assert portfolio["gross_q_high_minus_low"].to_list() == pytest.approx([8.0, -8.0])
    assert portfolio["turnover"][0] is None
    assert portfolio["turnover"][1] == pytest.approx(2.0)
    assert portfolio["net_10bps"][1] == pytest.approx(-8.002)


def test_prediction_contract_and_coverage_reject_missing_rows() -> None:
    frame = validate_predictions(metric_frame())
    expected = frame.select("security_id", "asof_date", "split")
    with pytest.raises(DataContractError, match="coverage gate"):
        prediction_coverage(frame.head(10), expected, split="valid", minimum=0.9)

    duplicate = pl.concat([frame, frame.head(1)])
    with pytest.raises(DataContractError, match="duplicate"):
        validate_predictions(duplicate)


def test_neutralization_is_orthogonal_to_point_in_time_exposures() -> None:
    frame = metric_frame().filter(pl.col("asof_date") == date(2025, 1, 2))
    result, audit = neutralize_predictions(frame)
    assert audit["available_rows"] == 10
    residuals = result["score_neutralized"]
    assert residuals.sum() == pytest.approx(0.0, abs=1e-10)
    cap = result["log_float_market_cap"].to_numpy()
    assert np.dot(residuals.to_numpy(), cap - cap.mean()) == pytest.approx(0.0, abs=1e-10)


def test_neutralization_stays_null_when_exposures_are_missing() -> None:
    frame = metric_frame().with_columns(
        pl.lit(None, dtype=pl.String).alias("industry_code"),
        pl.lit(None, dtype=pl.Float64).alias("log_float_market_cap"),
    )
    result, audit = neutralize_predictions(frame)
    assert result["score_neutralized"].null_count() == result.height
    assert audit["available_rows"] == 0
