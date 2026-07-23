from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
PatchTSTAlphaModel = pytest.importorskip("facdigger.models.patchtst_alpha").PatchTSTAlphaModel


def test_patchtst_alpha_shape_masks_and_backward() -> None:
    model = PatchTSTAlphaModel(
        context_length=12,
        num_input_channels=3,
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
        alpha_hidden_dim=8,
        alpha_dropout=0.0,
    )
    values = torch.randn(2, 12, 3)
    observed = torch.ones_like(values, dtype=torch.bool)
    observed[0, :4, 1] = False
    values[~observed] = 0.0

    output = model(values, observed)

    assert output.score.shape == (2,)
    assert output.encoder.hidden.shape == (2, 3, 3, 8)
    assert output.encoder.patch_mask.shape == (2, 3, 3)
    assert output.encoder.channel_mask.shape == (2, 3)
    assert not output.encoder.patch_mask[0, 1, 0]
    output.score.sum().backward()
    assert model.alpha_head.projection[-1].weight.grad is not None


def test_patchtst_alpha_can_overfit_one_tiny_batch() -> None:
    torch.manual_seed(9)
    model = PatchTSTAlphaModel(
        context_length=8,
        num_input_channels=2,
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
        alpha_hidden_dim=16,
        alpha_dropout=0.0,
    )
    values = torch.randn(6, 8, 2)
    observed = torch.ones_like(values, dtype=torch.bool)
    target = torch.tensor([-1.5, -0.9, -0.3, 0.3, 0.9, 1.5])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.HuberLoss()

    with torch.no_grad():
        initial = float(loss_fn(model(values, observed).score, target))
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(values, observed).score, target)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = float(loss_fn(model(values, observed).score, target))

    assert final < initial * 0.25
