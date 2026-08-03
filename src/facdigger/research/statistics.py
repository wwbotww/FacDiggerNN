"""Deterministic HAC and non-overlapping inference for daily factor series."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np


def _as_finite_array(values: list[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite observations")
    return array


def _inference_payload(
    *,
    count: int,
    mean: float | None,
    standard_error: float | None,
    lags: int,
    long_run_variance: float | None,
    null_mean: float,
    alpha: float,
) -> dict[str, Any]:
    variance_floor = np.finfo(np.float64).eps * max(abs(mean or 0.0), 1e-12) ** 2
    estimable = bool(
        count >= 2
        and mean is not None
        and standard_error is not None
        and math.isfinite(standard_error)
        and standard_error > 0
        and long_run_variance is not None
        and math.isfinite(long_run_variance)
        and long_run_variance > variance_floor
    )
    t_stat = (mean - null_mean) / standard_error if estimable else None
    p_value = 0.5 * math.erfc(t_stat / math.sqrt(2.0)) if t_stat is not None else None
    critical_value = NormalDist().inv_cdf(1.0 - alpha)
    lower_bound = (
        mean - critical_value * standard_error
        if estimable and mean is not None and standard_error is not None
        else None
    )
    return {
        "n": count,
        "mean": mean,
        "standard_error": standard_error if estimable else None,
        "t_stat": t_stat,
        "lags": lags,
        "long_run_variance": long_run_variance,
        "estimable": estimable,
        "alternative": "greater",
        "null_mean": null_mean,
        "p_value_one_sided": p_value,
        "confidence_level": 1.0 - alpha,
        "lower_bound": lower_bound,
    }


def newey_west_mean_inference(
    values: list[float], lags: int, *, null_mean: float = 0.0, alpha: float = 0.05
) -> dict[str, Any]:
    if lags < 0:
        raise ValueError("lags cannot be negative")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must satisfy 0 < alpha < 0.5")
    array = _as_finite_array(values, name="values")
    count = len(array)
    if count == 0:
        return _inference_payload(
            count=0,
            mean=None,
            standard_error=None,
            lags=lags,
            long_run_variance=None,
            null_mean=null_mean,
            alpha=alpha,
        )
    mean = float(array.mean())
    if count == 1:
        return _inference_payload(
            count=1,
            mean=mean,
            standard_error=None,
            lags=0,
            long_run_variance=None,
            null_mean=null_mean,
            alpha=alpha,
        )
    centered = array - mean
    effective_lags = min(lags, count - 1)
    long_run_variance = float(np.dot(centered, centered) / count)
    for lag in range(1, effective_lags + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        weight = 1.0 - lag / (effective_lags + 1)
        long_run_variance += 2.0 * weight * covariance
    standard_error = (
        math.sqrt(long_run_variance / count)
        if math.isfinite(long_run_variance)
        and long_run_variance > np.finfo(np.float64).eps * max(abs(mean), 1e-12) ** 2
        else None
    )
    return _inference_payload(
        count=count,
        mean=mean,
        standard_error=standard_error,
        lags=effective_lags,
        long_run_variance=long_run_variance,
        null_mean=null_mean,
        alpha=alpha,
    )


def non_overlapping_mean_inference(
    values: list[float], *, stride: int, offset: int
) -> dict[str, Any]:
    if stride < 1:
        raise ValueError("stride must be positive")
    if not 0 <= offset < stride:
        raise ValueError("offset must satisfy 0 <= offset < stride")
    selected = _as_finite_array(values, name="values")[offset::stride]
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
    standard_error = float(selected.std(ddof=1) / math.sqrt(count)) if count > 1 else None
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
    values: list[float],
    *,
    hac_lags: int,
    stride: int,
    offset: int,
    null_mean: float = 0.0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    return {
        "hac": newey_west_mean_inference(values, hac_lags, null_mean=null_mean, alpha=alpha),
        "non_overlapping": non_overlapping_mean_inference(values, stride=stride, offset=offset),
    }


def panel_mean_inference(
    groups: list[list[float]],
    *,
    hac_lags: int,
    stride: int,
    offset: int,
    null_mean: float = 0.0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Infer a pooled mean without creating autocovariance across fold boundaries."""

    if hac_lags < 0:
        raise ValueError("hac_lags cannot be negative")
    if stride < 1 or not 0 <= offset < stride:
        raise ValueError("invalid non-overlapping stride or offset")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must satisfy 0 < alpha < 0.5")
    arrays = [
        _as_finite_array(group, name=f"groups[{index}]") for index, group in enumerate(groups)
    ]
    arrays = [array for array in arrays if len(array) > 0]
    if not arrays:
        return {
            "hac": newey_west_mean_inference([], hac_lags, null_mean=null_mean, alpha=alpha),
            "non_overlapping": {
                **non_overlapping_mean_inference([], stride=stride, offset=offset),
                "fold_counts": [],
            },
        }
    combined = np.concatenate(arrays)
    count = len(combined)
    mean = float(combined.mean())
    effective_lags = min(hac_lags, max(len(array) for array in arrays) - 1)
    if count == 1:
        hac = _inference_payload(
            count=1,
            mean=mean,
            standard_error=None,
            lags=0,
            long_run_variance=None,
            null_mean=null_mean,
            alpha=alpha,
        )
    else:
        centered = [array - mean for array in arrays]
        long_run_variance = sum(float(np.dot(value, value)) for value in centered) / count
        for lag in range(1, effective_lags + 1):
            covariance_sum = sum(
                float(np.dot(value[lag:], value[:-lag])) for value in centered if len(value) > lag
            )
            weight = 1.0 - lag / (effective_lags + 1)
            long_run_variance += 2.0 * weight * covariance_sum / count
        standard_error = (
            math.sqrt(long_run_variance / count)
            if math.isfinite(long_run_variance)
            and long_run_variance > np.finfo(np.float64).eps * max(abs(mean), 1e-12) ** 2
            else None
        )
        hac = _inference_payload(
            count=count,
            mean=mean,
            standard_error=standard_error,
            lags=effective_lags,
            long_run_variance=long_run_variance,
            null_mean=null_mean,
            alpha=alpha,
        )
    selected = [array[offset::stride] for array in arrays]
    fold_counts = [len(array) for array in selected]
    non_overlapping_values = np.concatenate(selected) if selected else np.asarray([])
    non_overlapping = non_overlapping_mean_inference(
        non_overlapping_values.tolist(), stride=1, offset=0
    )
    non_overlapping.update({"stride": stride, "offset": offset, "fold_counts": fold_counts})
    return {"hac": hac, "non_overlapping": non_overlapping}


def holm_step_down(p_values: dict[str, float | None], *, alpha: float) -> dict[str, Any]:
    """Apply Holm's family-wise error correction with fail-closed invalid p-values."""

    if not 0 < alpha < 0.5:
        raise ValueError("alpha must satisfy 0 < alpha < 0.5")
    valid = sorted(
        (
            (key, float(value))
            for key, value in p_values.items()
            if value is not None and math.isfinite(value) and 0 <= value <= 1
        ),
        key=lambda item: (item[1], item[0]),
    )
    family_size = len(p_values)
    rejected_so_far = True
    running_adjusted = 0.0
    output: dict[str, Any] = {}
    for rank, (key, value) in enumerate(valid, start=1):
        multiplier = family_size - rank + 1
        critical_alpha = alpha / multiplier
        rejected = rejected_so_far and value <= critical_alpha
        rejected_so_far = rejected
        running_adjusted = max(running_adjusted, min(1.0, multiplier * value))
        output[key] = {
            "raw_p_value": value,
            "adjusted_p_value": running_adjusted,
            "rank": rank,
            "critical_alpha": critical_alpha,
            "rejected": rejected,
        }
    for key in p_values.keys() - output.keys():
        output[key] = {
            "raw_p_value": p_values[key],
            "adjusted_p_value": None,
            "rank": None,
            "critical_alpha": None,
            "rejected": False,
        }
    return output
