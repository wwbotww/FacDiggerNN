from __future__ import annotations

import math

import pytest

from facdigger.research.statistics import (
    newey_west_mean_inference,
    non_overlapping_mean_inference,
    panel_mean_inference,
)


def test_newey_west_mean_matches_iid_standard_error_at_zero_lags() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    result = newey_west_mean_inference(values, lags=0)
    expected = math.sqrt(sum((value - 2.5) ** 2 for value in values) / 4 / 4)

    assert result["mean"] == 2.5
    assert result["standard_error"] == pytest.approx(expected)
    assert result["t_stat"] == pytest.approx(2.5 / expected)


def test_non_overlapping_inference_uses_fixed_offset_without_cherry_picking() -> None:
    result = non_overlapping_mean_inference(
        [1.0, 100.0, 2.0, 200.0, 3.0, 300.0], stride=2, offset=0
    )

    assert result["n"] == 3
    assert result["mean"] == 2.0
    assert result["offset"] == 0


def test_panel_inference_restarts_non_overlapping_offset_for_each_fold() -> None:
    result = panel_mean_inference(
        [[1.0, 100.0, 2.0], [3.0, 200.0, 4.0]],
        hac_lags=1,
        stride=2,
        offset=0,
    )

    assert result["non_overlapping"]["n"] == 4
    assert result["non_overlapping"]["mean"] == pytest.approx(2.5)
