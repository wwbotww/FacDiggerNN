"""Shared cross-sectional ranking targets, losses and selection metrics."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import Field

from facdigger.data.config import StrictModel

RANKING_OBJECTIVE = "cross_sectional_rank_correlation_surrogate_v2_full_date"
TARGET_TRANSFORM = "full_date_average_rank_percentile_minus_one_to_one"


class CrossSectionalRankingConfig(StrictModel):
    name: Literal["cross_sectional_rank_correlation_surrogate_v2_full_date"] = (
        "cross_sectional_rank_correlation_surrogate_v2_full_date"
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


@dataclass(frozen=True)
class FullDateRankCorrelationStatistics:
    """Detached Float32 moments for one complete-date Pearson objective.

    The statistics are sufficient to compute the analytic gradient of the
    regularized loss used by :func:`cross_sectional_rank_correlation_loss`:

    ``1 - cov(score, target) / sqrt((var(score) + eps) * (var(target) + eps))``.

    Keeping only these scalar moments lets a trainer discard first-pass model
    activations and recompute one microbatch at a time during the gradient pass.
    """

    count: int
    microbatch_count: int
    epsilon: float
    score_mean: float
    target_mean: float
    score_centered_mean: float
    target_centered_mean: float
    score_variance: float
    target_variance: float
    covariance: float
    score_min: float
    score_max: float
    target_min: float
    target_max: float
    correlation: float
    loss: float

    def gradient(self, scores: Any, target_ranks: Any) -> Any:
        """Return ``d(loss) / d(scores)`` for one replayed microbatch.

        The returned tensor has the score tensor's shape and is always
        Float32. The caller must replay every row used to build these
        statistics and should use :class:`FullDateRankCorrelationReplay` to
        verify that contract before applying an optimizer step.
        """

        import torch

        score = scores.detach().float().reshape(-1)
        target = target_ranks.detach().float().reshape(-1)
        if score.shape != target.shape or score.numel() == 0:
            raise ValueError(
                "full-date rank correlation gradient requires aligned non-empty vectors"
            )
        if not torch.isfinite(score).all() or not torch.isfinite(target).all():
            raise FloatingPointError(
                "full-date rank correlation gradient received non-finite values"
            )

        score_mean = score.new_tensor(self.score_mean)
        target_mean = score.new_tensor(self.target_mean)
        score_variance = score.new_tensor(self.score_variance)
        target_variance = score.new_tensor(self.target_variance)
        covariance = score.new_tensor(self.covariance)
        epsilon = score.new_tensor(self.epsilon)
        denominator = torch.sqrt(score_variance + epsilon) * torch.sqrt(
            target_variance + epsilon
        )
        gradient = (
            covariance
            * (score - score_mean - self.score_centered_mean)
            / (score_variance + epsilon)
            - (target - target_mean - self.target_centered_mean)
        ) / (self.count * denominator)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(
                "full-date rank correlation analytic gradient became non-finite"
            )
        return gradient.reshape(scores.shape)

    def audit(self) -> dict[str, Any]:
        """Return JSON-serializable evidence for the first-pass objective."""

        return {
            "schema_version": 1,
            "objective": RANKING_OBJECTIVE,
            "method": "exact_two_pass_full_date_pearson",
            "accumulation_dtype": "float32",
            "rows": self.count,
            "first_pass_microbatches": self.microbatch_count,
            "epsilon": self.epsilon,
            "score_mean": self.score_mean,
            "target_mean": self.target_mean,
            "score_centered_mean": self.score_centered_mean,
            "target_centered_mean": self.target_centered_mean,
            "score_variance": self.score_variance,
            "target_variance": self.target_variance,
            "covariance": self.covariance,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "target_min": self.target_min,
            "target_max": self.target_max,
            "regularized_correlation": self.correlation,
            "loss": self.loss,
        }


class FullDateRankCorrelationAccumulator:
    """Collect detached microbatch outputs for exact complete-date moments.

    Only the Float32 score and target vectors for the current date are retained
    on the first input's device. This is bounded by the cross-section size (not
    by model activation size), keeps no computation graph, and lets
    :meth:`finalize` use the exact same centered reductions as the single-graph
    reference loss.
    """

    def __init__(self, *, epsilon: float) -> None:
        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ValueError("ranking epsilon must be finite and positive")
        self.epsilon = float(epsilon)
        self.count = 0
        self.microbatch_count = 0
        self._device: Any | None = None
        self._score_chunks: list[Any] = []
        self._target_chunks: list[Any] = []
        self._finalized = False

    def update(self, scores: Any, target_ranks: Any) -> None:
        """Add one no-graph microbatch from the same complete date."""

        import torch

        if self._finalized:
            raise RuntimeError("full-date rank correlation statistics are already finalized")
        with torch.no_grad():
            score = scores.detach().float().reshape(-1)
            target = target_ranks.detach().float().reshape(-1)
            if score.shape != target.shape or score.numel() == 0:
                raise ValueError(
                    "full-date statistics require aligned non-empty score/target vectors"
                )
            if self._device is not None and score.device != self._device:
                raise ValueError("all full-date statistic microbatches must share one device")
            if target.device != score.device:
                raise ValueError("full-date scores and targets must share one device")

            if self.count == 0:
                self._device = score.device
            self._score_chunks.append(score.clone())
            self._target_chunks.append(target.clone())
            self.count += score.numel()
            self.microbatch_count += 1

    def finalize(self) -> FullDateRankCorrelationStatistics:
        """Validate the complete date and freeze its scalar objective state."""

        import torch

        if self.count < 2 or not self._score_chunks:
            raise ValueError(
                "full-date rank correlation requires at least two accumulated rows"
            )
        self._finalized = True
        score = torch.cat(self._score_chunks)
        target = torch.cat(self._target_chunks)
        self._score_chunks.clear()
        self._target_chunks.clear()
        if not torch.isfinite(score).all() or not torch.isfinite(target).all():
            raise FloatingPointError(
                "full-date rank correlation statistics received non-finite values"
            )

        score_mean_tensor = score.mean()
        target_mean_tensor = target.mean()
        score_centered = score - score_mean_tensor
        target_centered = target - target_mean_tensor
        score_variance_tensor = score_centered.square().mean()
        target_variance_tensor = target_centered.square().mean()
        covariance_tensor = (score_centered * target_centered).mean()
        score_mean = float(score_mean_tensor.detach().cpu())
        target_mean = float(target_mean_tensor.detach().cpu())
        score_centered_mean = float(score_centered.mean().detach().cpu())
        target_centered_mean = float(target_centered.mean().detach().cpu())
        score_variance = float(score_variance_tensor.detach().cpu())
        target_variance = float(target_variance_tensor.detach().cpu())
        covariance = float(covariance_tensor.detach().cpu())
        scalar_values = {
            "score_mean": score_mean,
            "target_mean": target_mean,
            "score_centered_mean": score_centered_mean,
            "target_centered_mean": target_centered_mean,
            "score_variance": score_variance,
            "target_variance": target_variance,
            "covariance": covariance,
        }
        if not all(math.isfinite(value) for value in scalar_values.values()):
            raise FloatingPointError(
                "full-date rank correlation statistics became non-finite"
            )
        if score_variance < 0:
            raise FloatingPointError(
                "full-date rank correlation score variance became negative"
            )
        if target_variance <= 0:
            raise ValueError("full-date rank correlation target has zero variance")

        denominator = torch.sqrt(score_variance_tensor + self.epsilon) * torch.sqrt(
            target_variance_tensor + self.epsilon
        )
        correlation = float((covariance_tensor / denominator).detach().cpu())
        loss = float((1.0 - covariance_tensor / denominator).detach().cpu())
        if not math.isfinite(correlation) or not math.isfinite(loss):
            raise FloatingPointError("full-date rank correlation loss became non-finite")
        return FullDateRankCorrelationStatistics(
            count=self.count,
            microbatch_count=self.microbatch_count,
            epsilon=self.epsilon,
            score_mean=score_mean,
            target_mean=target_mean,
            score_centered_mean=score_centered_mean,
            target_centered_mean=target_centered_mean,
            score_variance=score_variance,
            target_variance=target_variance,
            covariance=covariance,
            score_min=float(score.min().detach().cpu()),
            score_max=float(score.max().detach().cpu()),
            target_min=float(target.min().detach().cpu()),
            target_max=float(target.max().detach().cpu()),
            correlation=correlation,
            loss=loss,
        )


class FullDateRankCorrelationReplay:
    """Validate a gradient replay before its accumulated gradients are stepped.

    A trainer should call :meth:`gradient` for every second-pass microbatch,
    backpropagate those returned gradients, then call :meth:`finalize` before
    ``optimizer.step()``. A mismatch means stochastic model state or row
    membership changed between passes, so the accumulated gradients must be
    discarded.
    """

    def __init__(
        self,
        statistics: FullDateRankCorrelationStatistics,
        *,
        relative_tolerance: float = 1e-5,
        absolute_tolerance: float = 1e-6,
    ) -> None:
        if relative_tolerance < 0 or absolute_tolerance < 0:
            raise ValueError("replay tolerances must be non-negative")
        self.statistics = statistics
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance
        self._accumulator = FullDateRankCorrelationAccumulator(
            epsilon=statistics.epsilon
        )
        self._finalized = False

    def gradient(self, scores: Any, target_ranks: Any) -> Any:
        """Record a replayed microbatch and return its analytic gradient."""

        if self._finalized:
            raise RuntimeError("full-date rank correlation replay is already finalized")
        self._accumulator.update(scores, target_ranks)
        return self.statistics.gradient(scores, target_ranks)

    def finalize(self) -> dict[str, Any]:
        """Fail closed if the second pass did not reproduce the first pass."""

        if self._finalized:
            raise RuntimeError("full-date rank correlation replay is already finalized")
        self._finalized = True
        replay = self._accumulator.finalize()
        if replay.count != self.statistics.count:
            raise RuntimeError(
                "full-date rank correlation replay row count differs from first pass: "
                f"{replay.count} != {self.statistics.count}"
            )
        checked_fields = (
            "score_mean",
            "target_mean",
            "score_centered_mean",
            "target_centered_mean",
            "score_variance",
            "target_variance",
            "covariance",
            "score_min",
            "score_max",
            "target_min",
            "target_max",
        )
        mismatches = [
            field
            for field in checked_fields
            if not math.isclose(
                getattr(replay, field),
                getattr(self.statistics, field),
                rel_tol=self.relative_tolerance,
                abs_tol=self.absolute_tolerance,
            )
        ]
        if mismatches:
            raise RuntimeError(
                "full-date rank correlation replay differs from first pass for: "
                + ", ".join(mismatches)
            )
        return {
            "schema_version": 1,
            "verified": True,
            "rows": replay.count,
            "second_pass_microbatches": replay.microbatch_count,
            "relative_tolerance": self.relative_tolerance,
            "absolute_tolerance": self.absolute_tolerance,
        }


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
