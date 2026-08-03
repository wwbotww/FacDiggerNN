"""Shared cross-sectional ranking targets, losses and selection metrics."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from pydantic import Field

from facdigger.data.config import StrictModel

RANKING_OBJECTIVE = "cross_sectional_rank_correlation_surrogate_v1"
TARGET_TRANSFORM = "full_date_average_rank_percentile_minus_one_to_one"


class CrossSectionalRankingConfig(StrictModel):
    name: Literal["cross_sectional_rank_correlation_surrogate_v1"] = (
        "cross_sectional_rank_correlation_surrogate_v1"
    )
    epsilon: float = Field(default=1e-6, gt=0)
    minimum_cross_section_size: int = Field(default=2, ge=2)
    minimum_selection_dates: int = Field(default=2, ge=1)
    minimum_selection_coverage: float = Field(default=0.99, gt=0, le=1)


def _group_indices(asof_dates: Sequence[Any]) -> list[np.ndarray]:
    grouped: OrderedDict[Any, list[int]] = OrderedDict()
    for index, asof_date in enumerate(asof_dates):
        grouped.setdefault(asof_date, []).append(index)
    return [np.asarray(indices, dtype=np.int64) for indices in grouped.values()]


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic one-based average ranks while preserving input order."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("rank values must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError("rank values must be finite")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def cross_sectional_rank_targets(
    targets: Sequence[float] | np.ndarray,
    asof_dates: Sequence[Any],
    *,
    minimum_cross_section_size: int,
) -> np.ndarray:
    """Rank raw targets inside each complete date and map ranks to ``[-1, 1]``."""

    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(asof_dates):
        raise ValueError("targets and asof_dates must be aligned one-dimensional arrays")
    if not np.isfinite(values).all():
        raise ValueError("cross-sectional targets must be finite")
    if minimum_cross_section_size < 2:
        raise ValueError("minimum_cross_section_size must be at least two")
    result = np.empty(len(values), dtype=np.float32)
    for indices in _group_indices(asof_dates):
        if len(indices) < minimum_cross_section_size:
            raise ValueError(
                "date cross-section is smaller than minimum_cross_section_size: "
                f"{len(indices)} < {minimum_cross_section_size}"
            )
        ranks = average_ranks(values[indices])
        result[indices] = (2.0 * (ranks - 1.0) / (len(indices) - 1.0) - 1.0).astype(
            np.float32
        )
    return result


def cross_sectional_rank_correlation_loss(
    scores: Any,
    target_ranks: Any,
    *,
    epsilon: float,
) -> Any:
    """Differentiate Pearson correlation against fixed target ranks in Float32."""

    import torch

    if epsilon <= 0:
        raise ValueError("ranking epsilon must be positive")
    score = scores.float().reshape(-1)
    target = target_ranks.float().reshape(-1)
    if score.shape != target.shape or score.numel() < 2:
        raise ValueError("rank correlation loss requires aligned vectors with at least two rows")
    if not torch.isfinite(score).all() or not torch.isfinite(target).all():
        raise FloatingPointError("rank correlation loss received non-finite values")
    score_centered = score - score.mean()
    target_centered = target - target.mean()
    target_variance = target_centered.square().mean()
    if float(target_variance.detach().cpu()) <= 0:
        raise ValueError("rank correlation target has zero variance")
    covariance = (score_centered * target_centered).mean()
    denominator = torch.sqrt(score_centered.square().mean() + epsilon) * torch.sqrt(
        target_variance + epsilon
    )
    loss = 1.0 - covariance / denominator
    if not torch.isfinite(loss):
        raise FloatingPointError("rank correlation loss became non-finite")
    return loss


def grouped_rank_ic_audit(
    scores: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
    asof_dates: Sequence[Any],
    *,
    minimum_cross_section_size: int,
    minimum_dates: int,
    minimum_coverage: float,
) -> dict[str, Any]:
    """Compute date-equal true Spearman Rank IC and enforce selection coverage."""

    score_values = np.asarray(scores, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.float64)
    if score_values.ndim != 1 or target_values.ndim != 1:
        raise ValueError("selection scores and targets must be one-dimensional")
    if len(score_values) != len(target_values) or len(score_values) != len(asof_dates):
        raise ValueError("selection scores, targets and dates must be aligned")
    if not np.isfinite(score_values).all() or not np.isfinite(target_values).all():
        raise FloatingPointError("selection scores and targets must be finite")
    daily: list[float] = []
    skipped = 0
    for indices in _group_indices(asof_dates):
        if len(indices) < minimum_cross_section_size:
            skipped += 1
            continue
        score_ranks = average_ranks(score_values[indices])
        target_ranks = average_ranks(target_values[indices])
        if np.std(score_ranks) == 0 or np.std(target_ranks) == 0:
            skipped += 1
            continue
        rank_ic = float(np.corrcoef(score_ranks, target_ranks)[0, 1])
        if not math.isfinite(rank_ic):
            skipped += 1
            continue
        daily.append(rank_ic)
    total_dates = len(daily) + skipped
    coverage = len(daily) / total_dates if total_dates else 0.0
    if len(daily) < minimum_dates:
        raise ValueError(
            f"selection has too few valid dates: {len(daily)} < {minimum_dates}"
        )
    if coverage < minimum_coverage:
        raise ValueError(
            f"selection Rank IC coverage is below threshold: {coverage:.6f} < "
            f"{minimum_coverage:.6f}"
        )
    return {
        "mean_rank_ic": float(np.mean(daily)),
        "valid_dates": len(daily),
        "skipped_dates": skipped,
        "date_coverage": coverage,
        "score_std": float(np.std(score_values)),
    }


def contiguous_group_sizes(asof_dates: Sequence[Any]) -> np.ndarray:
    """Return LightGBM group sizes and reject non-contiguous date rows."""

    sizes: list[int] = []
    seen: set[Any] = set()
    current: Any = object()
    for asof_date in asof_dates:
        if not sizes or asof_date != current:
            if asof_date in seen:
                raise ValueError("LightGBM date groups must be contiguous")
            seen.add(asof_date)
            sizes.append(1)
            current = asof_date
        else:
            sizes[-1] += 1
    return np.asarray(sizes, dtype=np.int32)


def lightgbm_relevance_grades(target_ranks: np.ndarray, *, bins: int) -> np.ndarray:
    """Convert ``[-1, 1]`` percentile ranks to deterministic integer grades."""

    if bins < 2:
        raise ValueError("LightGBM relevance bins must be at least two")
    ranks = np.asarray(target_ranks, dtype=np.float64)
    if not np.isfinite(ranks).all() or np.any(ranks < -1.000001) or np.any(ranks > 1.000001):
        raise ValueError("LightGBM target ranks must be finite and inside [-1, 1]")
    grades = np.floor((ranks + 1.0) * 0.5 * bins).astype(np.int32)
    return np.clip(grades, 0, bins - 1)


def mean_grouped_spearman(
    scores: np.ndarray, target_ranks: np.ndarray, group_sizes: np.ndarray
) -> float:
    """Compute a date-equal Spearman mean for LightGBM custom evaluation."""

    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(target_ranks, dtype=np.float64)
    sizes = np.asarray(group_sizes, dtype=np.int64)
    if sizes.ndim != 1 or np.any(sizes < 2) or int(sizes.sum()) != len(scores):
        raise ValueError("invalid grouped Spearman group sizes")
    if len(scores) != len(targets):
        raise ValueError("grouped Spearman scores and targets are misaligned")
    daily: list[float] = []
    start = 0
    for size in sizes:
        stop = start + int(size)
        score_ranks = average_ranks(scores[start:stop])
        fixed_target_ranks = targets[start:stop]
        if np.std(score_ranks) == 0 or np.std(fixed_target_ranks) == 0:
            raise ValueError("grouped Spearman contains a zero-variance date")
        daily.append(float(np.corrcoef(score_ranks, fixed_target_ranks)[0, 1]))
        start = stop
    result = float(np.mean(daily))
    if not math.isfinite(result):
        raise FloatingPointError("grouped Spearman became non-finite")
    return result
