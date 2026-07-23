"""Deterministic HAC and non-overlapping inference for daily factor series."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def newey_west_mean_inference(values: list[float], lags: int) -> dict[str, Any]:
    if lags < 0:
        raise ValueError("lags cannot be negative")
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    count = len(array)
    if count == 0:
        return {
            "n": 0,
            "mean": None,
            "standard_error": None,
            "t_stat": None,
            "lags": lags,
        }
    mean = float(array.mean())
    if count == 1:
        return {
            "n": 1,
            "mean": mean,
            "standard_error": None,
            "t_stat": None,
            "lags": 0,
        }
    centered = array - mean
    effective_lags = min(lags, count - 1)
    long_run_variance = float(np.dot(centered, centered) / count)
    for lag in range(1, effective_lags + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        weight = 1.0 - lag / (effective_lags + 1)
        long_run_variance += 2.0 * weight * covariance
    variance_of_mean = max(long_run_variance, 0.0) / count
    standard_error = math.sqrt(variance_of_mean)
    return {
        "n": count,
        "mean": mean,
        "standard_error": standard_error,
        "t_stat": mean / standard_error if standard_error > 0 else None,
        "lags": effective_lags,
    }


def non_overlapping_mean_inference(
    values: list[float], *, stride: int, offset: int
) -> dict[str, Any]:
    if stride < 1:
        raise ValueError("stride must be positive")
    if not 0 <= offset < stride:
        raise ValueError("offset must satisfy 0 <= offset < stride")
    selected = np.asarray(values, dtype=np.float64)[offset::stride]
    selected = selected[np.isfinite(selected)]
    count = len(selected)
    if count == 0:
        return {
            "n": 0,
            "mean": None,
            "standard_error": None,
            "t_stat": None,
            "stride": stride,
            "offset": offset,
        }
    mean = float(selected.mean())
    standard_error = (
        float(selected.std(ddof=1) / math.sqrt(count)) if count > 1 else None
    )
    return {
        "n": count,
        "mean": mean,
        "standard_error": standard_error,
        "t_stat": (
            mean / standard_error if standard_error is not None and standard_error > 0 else None
        ),
        "stride": stride,
        "offset": offset,
    }


def series_inference(
    values: list[float], *, hac_lags: int, stride: int, offset: int
) -> dict[str, Any]:
    return {
        "hac": newey_west_mean_inference(values, hac_lags),
        "non_overlapping": non_overlapping_mean_inference(
            values, stride=stride, offset=offset
        ),
    }


def panel_mean_inference(
    groups: list[list[float]], *, hac_lags: int, stride: int, offset: int
) -> dict[str, Any]:
    """Infer a pooled mean while never creating autocovariance across fold boundaries."""

    if hac_lags < 0:
        raise ValueError("hac_lags cannot be negative")
    if stride < 1 or not 0 <= offset < stride:
        raise ValueError("invalid non-overlapping stride or offset")
    arrays = [
        values
        for group in groups
        if len(values := np.asarray(group, dtype=np.float64)[np.isfinite(group)]) > 0
    ]
    if not arrays:
        return {
            "hac": newey_west_mean_inference([], hac_lags),
            "non_overlapping": non_overlapping_mean_inference(
                [], stride=stride, offset=offset
            ),
        }
    combined = np.concatenate(arrays)
    count = len(combined)
    mean = float(combined.mean())
    if count == 1:
        hac = {
            "n": 1,
            "mean": mean,
            "standard_error": None,
            "t_stat": None,
            "lags": 0,
        }
    else:
        centered = [array - mean for array in arrays]
        effective_lags = min(hac_lags, max(len(array) for array in arrays) - 1)
        long_run_variance = sum(float(np.dot(value, value)) for value in centered) / count
        for lag in range(1, effective_lags + 1):
            covariance_sum = sum(
                float(np.dot(value[lag:], value[:-lag]))
                for value in centered
                if len(value) > lag
            )
            weight = 1.0 - lag / (effective_lags + 1)
            long_run_variance += 2.0 * weight * covariance_sum / count
        standard_error = math.sqrt(max(long_run_variance, 0.0) / count)
        hac = {
            "n": count,
            "mean": mean,
            "standard_error": standard_error,
            "t_stat": mean / standard_error if standard_error > 0 else None,
            "lags": effective_lags,
        }
    selected = [array[offset::stride] for array in arrays if len(array) > offset]
    non_overlapping_values = np.concatenate(selected) if selected else np.asarray([])
    non_overlapping = non_overlapping_mean_inference(
        non_overlapping_values.tolist(), stride=1, offset=0
    )
    non_overlapping.update({"stride": stride, "offset": offset})
    return {"hac": hac, "non_overlapping": non_overlapping}
