from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from facdigger.models.patchtst_alpha import PatchTSTAlphaModel  # noqa: E402
from facdigger.models.patchtst_transfer import module_fingerprint  # noqa: E402
from facdigger.training.e2_config import E2ExperimentConfig  # noqa: E402
from facdigger.training.e2_engine import configure_finetune_stage  # noqa: E402


def _model() -> PatchTSTAlphaModel:
    return PatchTSTAlphaModel(
        context_length=8,
        num_input_channels=2,
        patch_length=4,
        patch_stride=4,
        d_model=8,
        num_attention_heads=2,
        num_hidden_layers=2,
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


def _config() -> E2ExperimentConfig:
    return E2ExperimentConfig.model_validate(
        {
            "channels": ["x0", "x1"],
            "model": {
                "patch_length": 4,
                "patch_stride": 4,
                "d_model": 8,
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "ffn_dim": 16,
                "dropout": 0.0,
                "norm_type": "layernorm",
                "alpha_hidden_dim": 8,
                "alpha_dropout": 0.0,
            },
            "training": {
                "max_epochs": 3,
                "minimum_epochs": 2,
                "head_only_epochs": 1,
                "unfreeze_last_n_blocks": 1,
            },
        }
    )


def test_finetune_policy_freezes_encoder_then_unfreezes_only_last_block() -> None:
    model = _model()
    config = _config()

    ft0 = configure_finetune_stage(model, config, epoch=1)
    assert ft0["name"] == "ft0_head_only"
    assert ft0["trainable_encoder_parameters"] == 0
    assert not model.backbone.training
    assert model.alpha_head.training

    ft1 = configure_finetune_stage(model, config, epoch=2)
    assert ft1["name"] == "ft1_last_blocks"
    assert ft1["unfrozen_blocks"] == [1]
    assert not any(
        parameter.requires_grad for parameter in model.backbone.encoder.layers[0].parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.backbone.encoder.layers[1].parameters()
    )
    assert not model.backbone.encoder.layers[0].training
    assert model.backbone.encoder.layers[1].training


def test_module_fingerprint_covers_scalar_buffers_and_parameter_changes() -> None:
    model = _model()
    before = module_fingerprint(model.backbone)
    with torch.no_grad():
        next(model.backbone.parameters()).add_(1.0)
    after = module_fingerprint(model.backbone)

    assert before != after
