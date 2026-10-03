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
        "head.0.weight", "head.0.bias", "head.3.weight", "head.3.bias",
        "head.6.weight", "head.6.bias",
    }
    for name, value in control.state_dict().items():
        assert torch.equal(value, intervention.state_dict()[name])
    intervention.load_state_dict(control.state_dict(), strict=True)
    local, market = torch.randn(50, 400), torch.randn(1, 120)
    control.eval()
    intervention.eval()
    assert torch.equal(control.score_date(local, market).scores,
                       intervention.score_date(local, market).scores)
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
        FinanceStatisticsRanker(
            "statistics_mlp", input_dim=3, horizons=(5,), dropout=probability
        )


@pytest.mark.parametrize("kind", [None, "finance", "statistics_linear"])
def test_statistics_dropout_cannot_change_other_models(kind):
    with pytest.raises(ValueError, match="requires statistics_mlp"):
        statistics_dropout_identity(kind, 0.3)
