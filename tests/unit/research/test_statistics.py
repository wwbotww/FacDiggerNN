from __future__ import annotations

import math

import pytest

from facdigger.research.statistics import (
    holm_step_down,
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
    assert result["p_value_one_sided"] < 0.05
    assert result["estimable"] is True


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
    assert result["non_overlapping"]["fold_counts"] == [2, 2]


def test_zero_variance_hac_is_not_estimable() -> None:
    result = newey_west_mean_inference([0.1, 0.1, 0.1], lags=1)

    assert result["estimable"] is False
    assert result["p_value_one_sided"] is None


def test_holm_step_down_controls_the_three_attribution_tests() -> None:
    result = holm_step_down({"architecture": 0.01, "external": 0.02, "financial": 0.03}, alpha=0.05)

    assert result["architecture"]["rejected"] is True
    assert result["architecture"]["critical_alpha"] == pytest.approx(0.05 / 3)
    assert result["external"]["rejected"] is True
    assert result["financial"]["rejected"] is True
    assert result["financial"]["adjusted_p_value"] == pytest.approx(0.04)


def test_holm_step_down_stops_after_first_failure_and_invalid_p_fails_closed() -> None:
    result = holm_step_down({"architecture": 0.01, "external": 0.04, "financial": None}, alpha=0.05)

    assert result["architecture"]["rejected"] is True
    assert result["external"]["rejected"] is False
    assert result["financial"]["rejected"] is False
