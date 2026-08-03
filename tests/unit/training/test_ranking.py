from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from facdigger.training.ranking import (  # noqa: E402
    contiguous_group_sizes,
    cross_sectional_rank_correlation_loss,
    cross_sectional_rank_targets,
    grouped_rank_ic_audit,
    lightgbm_relevance_grades,
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
