import pytest
import torch
from torch import nn

from facdigger.models.finance_statistics import FinanceStatisticsRanker, statistics_dropout_identity


def test_dropout_intervention_preserves_initial_weights_and_eval_but_changes_training():
    torch.manual_seed(17)
    control = FinanceStatisticsRanker("statistics_mlp", input_dim=520, horizons=(1, 5, 20))
    torch.manual_seed(17)
    intervention = FinanceStatisticsRanker(
        "statistics_mlp", input_dim=520, horizons=(1, 5, 20), dropout=0.3
    )
    # Saved legacy MLP keys remain loadable without migrations or extra parameters.
    assert set(control.state_dict()) == {
        "head.0.weight",
        "head.0.bias",
        "head.3.weight",
        "head.3.bias",
        "head.6.weight",
        "head.6.bias",
    }
    for name, value in control.state_dict().items():
        assert torch.equal(value, intervention.state_dict()[name])
    intervention.load_state_dict(control.state_dict(), strict=True)
    local, market = torch.randn(50, 400), torch.randn(1, 120)
    control.eval()
    intervention.eval()
    assert torch.equal(
        control.score_date(local, market).scores, intervention.score_date(local, market).scores
    )
    control.train()
    intervention.train()
    torch.manual_seed(42)
    baseline = control.score_date(local, market).scores
    torch.manual_seed(42)
    changed = intervention.score_date(local, market).scores
    assert not torch.equal(baseline, changed)
    assert [m.p for m in intervention.modules() if isinstance(m, nn.Dropout)] == [0.3, 0.3]
    assert statistics_dropout_identity("statistics_mlp", 0.1) == {}


@pytest.mark.parametrize("probability", [-0.1, 1, float("nan"), float("inf")])
def test_statistics_dropout_rejects_invalid_probabilities(probability):
    with pytest.raises(ValueError, match="finite and in"):
        FinanceStatisticsRanker("statistics_mlp", input_dim=3, horizons=(5,), dropout=probability)


@pytest.mark.parametrize("kind", [None, "finance", "statistics_linear"])
def test_statistics_dropout_cannot_change_other_models(kind):
    with pytest.raises(ValueError, match="requires statistics_mlp"):
        statistics_dropout_identity(kind, 0.3)


def test_conditional_linear_pairs_head_rng_and_initial_scores_and_learns_state():
    torch.manual_seed(42)
    base = FinanceStatisticsRanker("statistics_linear", input_dim=520, horizons=(1, 5, 20))
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    model = FinanceStatisticsRanker(
        "statistics_conditional_linear",
        input_dim=520,
        horizons=(1, 5, 20),
        local_dim=364,
        market_channels=6,
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in model.parameters()) == 8115
    assert set(base.state_dict()) == {"head.weight", "head.bias"}
    assert not torch.count_nonzero(model.interaction)
    for key, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])
    local, market = torch.randn(7, 364), torch.randn(1, 156)
    assert torch.equal(
        model.score_date(local, market).scores, base.score_date(local, market).scores
    )
    model.score_date(local, market).scores.square().sum().backward()
    assert model.interaction.grad.abs().sum() > 0
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.interaction[0, 0, 0] = 1
    market[0, -6] = 1
    scores = model.score_date(local, market).scores
    market[0, -6] = -1
    assert torch.equal(model.score_date(local, market).scores, -scores)
    order = torch.randperm(len(local))
    whole = model.score_date(local, market).scores
    assert torch.equal(model.score_date(local[order], market).scores, whole[order])
    parts = torch.cat([model.score_date(part, market).scores for part in local.split(2)])
    torch.testing.assert_close(whole, parts)


def test_conditional_linear_requires_explicit_dimensions():
    with pytest.raises(ValueError, match="requires local"):
        FinanceStatisticsRanker("statistics_conditional_linear", input_dim=3, horizons=(5,))
