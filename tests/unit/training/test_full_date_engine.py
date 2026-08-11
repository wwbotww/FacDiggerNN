from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from facdigger.training.e1_config import E1TrainingConfig  # noqa: E402
from facdigger.training.e1_engine import (  # noqa: E402
    _backward_complete_date,
    _dates_in_current_optimizer_step,
    _iter_device_microbatches,
)
from facdigger.training.e2_config import E2FineTuneConfig  # noqa: E402
from facdigger.training.ranking import (  # noqa: E402
    cross_sectional_rank_correlation_loss,
)


class _LinearAlpha(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 1)

    def forward(self, values, observed_mask):
        del observed_mask
        score = self.projection(values[:, 0, :]).squeeze(-1)
        return SimpleNamespace(score=score)


class _StochasticAlpha(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.normalization = torch.nn.BatchNorm1d(3)
        self.dropout = torch.nn.Dropout(0.35)
        self.projection = torch.nn.Linear(3, 1)

    def forward(self, values, observed_mask):
        del observed_mask
        hidden = self.normalization(values[:, 0, :])
        hidden = self.dropout(hidden)
        return SimpleNamespace(score=self.projection(hidden).squeeze(-1))


def _batch(rows: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    generator = torch.Generator().manual_seed(101)
    values = torch.randn(rows, 1, 3, generator=generator)
    batch = {
        "values": values,
        "observed_mask": torch.ones_like(values, dtype=torch.bool),
        "sample_index": torch.arange(rows),
    }
    target_ranks = torch.linspace(-1.0, 1.0, rows)
    return batch, target_ranks


def _disabled_scaler() -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=False)


def test_cross_section_minimum_is_independent_of_physical_microbatch_size() -> None:
    objective = {"minimum_cross_section_size": 128}

    assert E1TrainingConfig(batch_size=16, objective=objective).batch_size == 16
    assert E2FineTuneConfig(batch_size=16, objective=objective).batch_size == 16


def test_final_optimizer_group_uses_its_actual_date_count() -> None:
    group_sizes = [
        _dates_in_current_optimizer_step(
            date_index, total_dates=5, configured_dates=2
        )
        for date_index in range(1, 6)
    ]

    assert group_sizes == [2, 2, 2, 2, 1]


def test_device_microbatches_balance_the_tail_without_exceeding_limit() -> None:
    batch, _ = _batch(65)

    microbatches = _iter_device_microbatches(batch, batch_size=64)

    assert [len(item["sample_index"]) for item in microbatches] == [33, 32]
    torch.testing.assert_close(
        torch.cat([item["sample_index"] for item in microbatches]),
        batch["sample_index"],
        rtol=0,
        atol=0,
    )


def test_memory_bounded_full_date_step_matches_single_graph_parameter_update() -> None:
    batch, target_ranks = _batch(13)
    reference = _LinearAlpha()
    replayed = copy.deepcopy(reference)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
    replayed_optimizer = torch.optim.SGD(replayed.parameters(), lr=0.03)

    reference_score = reference(batch["values"], batch["observed_mask"]).score
    reference_loss = cross_sectional_rank_correlation_loss(
        reference_score, target_ranks, epsilon=1e-6
    )
    reference_loss.backward()
    reference_optimizer.step()

    audit = _backward_complete_date(
        replayed,
        batch,
        device="cpu",
        target_rank_lookup=target_ranks,
        epsilon=1e-6,
        amp_enabled=False,
        scaler=_disabled_scaler(),
        physical_microbatch_size=4,
        dates_in_optimizer_step=1,
    )
    replayed_optimizer.step()

    assert audit["rows"] == 13
    assert audit["first_pass_microbatches"] == 4
    assert audit["second_pass_microbatches"] == 4
    assert audit["loss"] == pytest.approx(float(reference_loss.detach()), abs=1e-7)
    for reference_parameter, replayed_parameter in zip(
        reference.parameters(), replayed.parameters(), strict=True
    ):
        torch.testing.assert_close(
            replayed_parameter, reference_parameter, rtol=2e-6, atol=2e-7
        )


def test_two_pass_replays_dropout_and_updates_batchnorm_only_once() -> None:
    batch, target_ranks = _batch(11)
    base = _StochasticAlpha()
    reference = copy.deepcopy(base).train()
    replayed = copy.deepcopy(base).train()
    microbatch_size = 4

    torch.manual_seed(773)
    reference_scores = []
    for start in range(0, len(target_ranks), microbatch_size):
        stop = start + microbatch_size
        reference_scores.append(
            reference(
                batch["values"][start:stop],
                batch["observed_mask"][start:stop],
            ).score
        )
    reference_loss = cross_sectional_rank_correlation_loss(
        torch.cat(reference_scores), target_ranks, epsilon=1e-6
    )
    reference_loss.backward()

    torch.manual_seed(773)
    _backward_complete_date(
        replayed,
        batch,
        device="cpu",
        target_rank_lookup=target_ranks,
        epsilon=1e-6,
        amp_enabled=False,
        scaler=_disabled_scaler(),
        physical_microbatch_size=microbatch_size,
        dates_in_optimizer_step=1,
    )

    assert int(reference.normalization.num_batches_tracked) == 3
    assert int(replayed.normalization.num_batches_tracked) == 3
    torch.testing.assert_close(
        replayed.normalization.running_mean,
        reference.normalization.running_mean,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        replayed.normalization.running_var,
        reference.normalization.running_var,
        rtol=0,
        atol=0,
    )
    for (reference_name, reference_parameter), (
        replayed_name,
        replayed_parameter,
    ) in zip(reference.named_parameters(), replayed.named_parameters(), strict=True):
        assert replayed_name == reference_name
        torch.testing.assert_close(
            replayed_parameter.grad,
            reference_parameter.grad,
            rtol=3e-5,
            atol=3e-6,
        )
