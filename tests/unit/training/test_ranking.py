from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from facdigger.training.ranking import (  # noqa: E402
    RANKING_OBJECTIVE,
    CrossSectionalRankingConfig,
    FullDateRankCorrelationAccumulator,
    FullDateRankCorrelationReplay,
    contiguous_group_sizes,
    cross_sectional_rank_correlation_loss,
    cross_sectional_rank_targets,
    grouped_rank_ic_audit,
    lightgbm_relevance_grades,
)


def test_ranking_protocol_rejects_legacy_microbatch_objective() -> None:
    assert RANKING_OBJECTIVE == "cross_sectional_rank_correlation_surrogate_v2_full_date"
    assert CrossSectionalRankingConfig().name == RANKING_OBJECTIVE

    with pytest.raises(ValueError, match="literal_error"):
        CrossSectionalRankingConfig(
            name="cross_sectional_rank_correlation_surrogate_v1"
        )


def test_cross_sectional_targets_use_full_date_average_ties() -> None:
    targets = np.asarray([3.0, 1.0, 1.0, 4.0, 2.0, 8.0], dtype=np.float64)
    dates = ["d1", "d1", "d1", "d2", "d2", "d2"]

    ranks = cross_sectional_rank_targets(
        targets, dates, minimum_cross_section_size=3
    )

    np.testing.assert_allclose(ranks[:3], [1.0, -0.5, -0.5])
    np.testing.assert_allclose(ranks[3:], [0.0, -1.0, 1.0])


def test_rank_correlation_loss_has_expected_order_and_finite_gradient() -> None:
    target = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    perfect = torch.tensor([-2.0, -1.0, 1.0, 2.0], requires_grad=True)
    reverse = -perfect.detach()

    perfect_loss = cross_sectional_rank_correlation_loss(
        perfect, target, epsilon=1e-6
    )
    reverse_loss = cross_sectional_rank_correlation_loss(
        reverse, target, epsilon=1e-6
    )
    shifted_scaled_loss = cross_sectional_rank_correlation_loss(
        perfect * 7.0 + 11.0, target, epsilon=1e-6
    )
    perfect_loss.backward()

    assert float(perfect_loss.detach()) == pytest.approx(0.0, abs=2e-6)
    assert float(reverse_loss) == pytest.approx(2.0, abs=2e-6)
    assert float(shifted_scaled_loss.detach()) == pytest.approx(
        float(perfect_loss.detach()), abs=2e-6
    )
    assert perfect.grad is not None
    assert torch.isfinite(perfect.grad).all()


def test_two_pass_full_date_gradient_matches_single_graph_autograd() -> None:
    generator = torch.Generator().manual_seed(73)
    scores = torch.randn(19, generator=generator, dtype=torch.float32)
    targets = torch.linspace(-1.0, 1.0, 19, dtype=torch.float32)
    epsilon = 1e-5

    reference_scores = scores.clone().requires_grad_(True)
    reference_loss = cross_sectional_rank_correlation_loss(
        reference_scores, targets, epsilon=epsilon
    )
    reference_loss.backward()

    accumulator = FullDateRankCorrelationAccumulator(epsilon=epsilon)
    for start, stop in ((0, 3), (3, 11), (11, 19)):
        accumulator.update(scores[start:stop], targets[start:stop])
    statistics = accumulator.finalize()

    replay_scores = scores.clone().requires_grad_(True)
    replay = FullDateRankCorrelationReplay(statistics)
    for start, stop in ((0, 5), (5, 7), (7, 19)):
        score_chunk = replay_scores[start:stop]
        gradient = replay.gradient(score_chunk, targets[start:stop])
        assert gradient.dtype == torch.float32
        score_chunk.backward(gradient)
    replay_audit = replay.finalize()

    assert statistics.loss == pytest.approx(float(reference_loss.detach()), abs=2e-6)
    torch.testing.assert_close(
        replay_scores.grad,
        reference_scores.grad,
        rtol=2e-5,
        atol=2e-6,
    )
    assert replay_audit["verified"] is True
    assert replay_audit["rows"] == 19
    assert replay_audit["second_pass_microbatches"] == 3
    assert statistics.audit()["accumulation_dtype"] == "float32"
    assert statistics.audit()["first_pass_microbatches"] == 3


def test_two_pass_gradient_matches_float32_autograd_with_large_score_offset() -> None:
    generator = torch.Generator().manual_seed(22)
    scores = 1000.0 + torch.randn(24, generator=generator) * 1e-3
    targets = torch.randn(24, generator=generator)
    reference_scores = scores.clone().requires_grad_(True)
    reference_loss = cross_sectional_rank_correlation_loss(
        reference_scores, targets, epsilon=1e-6
    )
    reference_loss.backward()
    accumulator = FullDateRankCorrelationAccumulator(epsilon=1e-6)
    for score_chunk, target_chunk in zip(scores.split(5), targets.split(5), strict=True):
        accumulator.update(score_chunk, target_chunk)

    statistics = accumulator.finalize()
    analytic_gradient = statistics.gradient(scores, targets)

    torch.testing.assert_close(
        analytic_gradient,
        reference_scores.grad,
        rtol=2e-5,
        atol=2e-6,
    )
    assert abs(statistics.score_centered_mean) > 0


def test_two_pass_full_date_gradient_preserves_score_shape() -> None:
    scores = torch.tensor([[-0.7], [0.2], [1.3]], dtype=torch.float16)
    targets = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    accumulator = FullDateRankCorrelationAccumulator(epsilon=1e-6)
    accumulator.update(scores, targets)

    gradient = accumulator.finalize().gradient(scores, targets)

    assert gradient.shape == scores.shape
    assert gradient.dtype == torch.float32


def test_full_date_statistics_fail_closed_on_zero_target_variance() -> None:
    scores = torch.tensor([-1.0, 0.0, 1.0])
    targets = torch.ones(3)
    accumulator = FullDateRankCorrelationAccumulator(epsilon=1e-6)
    accumulator.update(scores, targets)

    with pytest.raises(ValueError, match="target has zero variance"):
        accumulator.finalize()


def test_full_date_gradient_allows_constant_scores_like_reference_loss() -> None:
    scores = torch.ones(3)
    targets = torch.tensor([-1.0, 0.0, 1.0])
    reference_scores = scores.clone().requires_grad_(True)
    reference_loss = cross_sectional_rank_correlation_loss(
        reference_scores, targets, epsilon=1e-6
    )
    reference_loss.backward()
    accumulator = FullDateRankCorrelationAccumulator(epsilon=1e-6)
    accumulator.update(scores, targets)
    statistics = accumulator.finalize()

    gradient = statistics.gradient(scores, targets)

    assert statistics.loss == pytest.approx(float(reference_loss.detach()))
    torch.testing.assert_close(gradient, reference_scores.grad)


def test_full_date_statistics_fail_closed_on_non_finite_input() -> None:
    accumulator = FullDateRankCorrelationAccumulator(epsilon=1e-6)
    accumulator.update(torch.tensor([0.0, float("nan")]), torch.tensor([-1.0, 1.0]))

    with pytest.raises(FloatingPointError, match="non-finite"):
        accumulator.finalize()


def test_full_date_gradient_replay_rejects_changed_scores() -> None:
    scores = torch.tensor([-1.0, -0.2, 0.4, 1.1])
    targets = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    accumulator = FullDateRankCorrelationAccumulator(epsilon=1e-6)
    accumulator.update(scores, targets)
    statistics = accumulator.finalize()
    replay = FullDateRankCorrelationReplay(statistics)

    replay.gradient(scores + torch.tensor([0.0, 0.0, 0.0, 0.2]), targets)

    with pytest.raises(RuntimeError, match="differs from first pass"):
        replay.finalize()


def test_selection_rank_ic_is_date_equal_and_fails_closed_on_missing_dates() -> None:
    dates = ["d1"] * 3 + ["d2"] * 3
    targets = np.asarray([1, 2, 3, 1, 2, 3], dtype=np.float64)
    scores = np.asarray([1, 2, 3, 3, 2, 1], dtype=np.float64)

    audit = grouped_rank_ic_audit(
        scores,
        targets,
        dates,
        minimum_cross_section_size=3,
        minimum_dates=2,
        minimum_coverage=1.0,
    )
    assert audit["mean_rank_ic"] == pytest.approx(0.0)
    assert audit["valid_dates"] == 2

    with pytest.raises(ValueError, match="coverage"):
        grouped_rank_ic_audit(
            np.asarray([1, 2, 3, 1, 1, 1], dtype=np.float64),
            targets,
            dates,
            minimum_cross_section_size=3,
            minimum_dates=1,
            minimum_coverage=1.0,
        )


def test_lightgbm_rank_artifacts_require_contiguous_groups() -> None:
    sizes = contiguous_group_sizes(["d1", "d1", "d2", "d2", "d2"])
    np.testing.assert_array_equal(sizes, [2, 3])
    np.testing.assert_array_equal(
        lightgbm_relevance_grades(
            np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0]), bins=5
        ),
        [0, 1, 2, 3, 4],
    )
    with pytest.raises(ValueError, match="contiguous"):
        contiguous_group_sizes(["d1", "d2", "d1"])
