from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl
import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/research/diagnose_historical_results.py"
spec = importlib.util.spec_from_file_location("historical_diagnosis", SCRIPT)
diagnosis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnosis)


def predictions():
    return pl.DataFrame(
        {
            "asof_date": [date(2021, 1, 4)] * 3,
            "security_id": ["a", "b", "c"],
            "split": ["valid"] * 3,
            "score_raw": [1.0, 1.0, 3.0],
            "target": [1.0, 2.0, 2.0],
        }
    )


def test_recalculated_ic_uses_average_ties_and_ignores_row_order():
    frame = predictions()
    daily = diagnosis.daily_ic(frame.reverse())
    assert daily["rank_ic"][0] == pytest.approx(0.5)
    assert daily["n"][0] == 3


@pytest.mark.parametrize(
    "change,error",
    [
        ("test", "validation predictions only"),
        ("duplicate", "Duplicate prediction keys"),
        ("nan", "Non-finite"),
    ],
)
def test_invalid_or_holdout_predictions_fail_closed(tmp_path, change, error):
    frame = predictions()
    if change == "test":
        frame = frame.with_columns(pl.lit("test").alias("split"))
    elif change == "duplicate":
        frame = pl.concat([frame, frame.head(1)])
    else:
        frame = frame.with_columns(pl.lit(float("nan")).alias("score_raw"))
    path = tmp_path / "predictions.parquet"
    frame.write_parquet(path)
    with pytest.raises(ValueError, match=error):
        diagnosis.read_predictions(path)


def test_constant_scores_cannot_silently_drop_unscorable_dates():
    with pytest.raises(ValueError, match="Undefined daily IC"):
        diagnosis.daily_ic(predictions().with_columns(pl.lit(1.0).alias("score_raw")))


def test_fold_and_date_weighting_are_distinct_and_null_is_explicit():
    groups = [[0.005] * 200, [0.005] * 200, [-0.006] * 400]
    result = diagnosis.paired_inference(groups, null_mean=0.001)
    assert result["equal_fold_mean"] == pytest.approx(0.004 / 3)
    assert result["date_weighted_mean"] == pytest.approx(-0.0005)
    for hac in result["hac_sensitivity_exploratory"].values():
        assert hac["null_mean"] == 0.001
        assert hac["t_stat"] < 0
    assert len(result["nonoverlapping_means_all_offsets"]) == 5
