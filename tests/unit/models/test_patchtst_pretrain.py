from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from facdigger.models.patchtst_pretrain import FinancialPatchTSTPretrainer  # noqa: E402


def _model(loss: str = "huber") -> FinancialPatchTSTPretrainer:
    config = SimpleNamespace(
        patch_length=4,
        patch_stride=4,
        d_model=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        ffn_dim=16,
        dropout=0.0,
        attention_dropout=0.0,
        positional_dropout=0.0,
        path_dropout=0.0,
        ff_dropout=0.0,
        norm_type="layernorm",
        pre_norm=False,
        scaling="mean",
    )
    return FinancialPatchTSTPretrainer(
        context_length=12,
        num_input_channels=2,
        model_config=config,
        mask_ratio=0.5,
        loss=loss,
        huber_delta=1.0,
    )


def test_masked_reconstruction_loss_uses_only_observed_masked_elements() -> None:
    model = _model()
    values = torch.randn(2, 12, 2)
    observed = torch.ones_like(values, dtype=torch.bool)
    observed[:, 1, 0] = False
    values[:, 1, 0] = 0.0

    torch.manual_seed(19)
    output = model(values, observed)
    torch.manual_seed(19)
    internal = model.pretrainer.model(
        past_values=values, past_observed_mask=observed, return_dict=True
    )
    prediction = model.pretrainer.head(internal.last_hidden_state)
    observed_elements = observed.transpose(1, 2).unfold(-1, 4, 4)
    valid = internal.mask.bool().unsqueeze(-1) & observed_elements
    difference = prediction - internal.patch_input
    absolute = difference.abs()
    expected_elements = torch.where(
        absolute <= 1.0,
        0.5 * difference.square(),
        absolute - 0.5,
    )
    expected = expected_elements.masked_select(valid).mean()

    torch.testing.assert_close(output.loss, expected)
    assert output.valid_element_count == int(valid.sum())
    assert output.valid_element_count < int(internal.mask.sum()) * 4
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
