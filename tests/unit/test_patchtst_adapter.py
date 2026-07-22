from dataclasses import dataclass

import pytest

from facdigger.models.patchtst_adapter import (
    CanonicalPatchTSTConfig,
    WeightLoadError,
    audit_matching_encoder_weights,
    load_matching_encoder_weights,
)


@dataclass
class FakeTensor:
    shape: tuple[int, ...]
    value: int = 0

    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


class FakeModule:
    def __init__(self, state):
        self._state = state
        self.loaded = None

    def state_dict(self):
        return self._state

    def load_state_dict(self, state, strict):
        self.loaded = (state, strict)


def source_checkpoint_config() -> dict:
    return {
        "num_input_channels": 7,
        "context_length": 512,
        "patch_length": 12,
        "stride": 12,
        "encoder_layers": 6,
        "d_model": 128,
        "encoder_attention_heads": 16,
        "encoder_ffn_dim": 512,
        "shared_embedding": True,
        "channel_attention": False,
        "dropout": 0.3,
        "attention_dropout": 0.0,
        "positional_dropout": 0.0,
        "dropout_path": 0.0,
        "ff_dropout": 0.0,
        "head_dropout": 0.2,
        "norm": "BatchNorm",
        "pre_norm": False,
        "positional_encoding": "sincos",
        "use_cls_token": False,
        "shared_projection": True,
        "scaling": "mean",
        "mask_input": True,
        "mask_type": "random",
        "mask_ratio": 0.2,
        "mask_value": 0,
        "prediction_length": 24,
    }


def test_legacy_checkpoint_config_is_normalized_without_using_current_defaults() -> None:
    config = CanonicalPatchTSTConfig.from_checkpoint_dict(source_checkpoint_config())

    assert config.num_hidden_layers == 6
    assert config.num_attention_heads == 16
    assert config.patch_stride == 12
    assert config.norm_type == "batchnorm"
    assert config.pre_norm is False
    assert config.scaling == "mean"
    assert config.random_mask_ratio == 0.2


def test_legacy_checkpoint_state_keys_are_normalized() -> None:
    from facdigger.models.patchtst_adapter import canonical_state_key

    assert (
        canonical_state_key("model.encoder.w_p.weight") == "encoder.embedder.input_embedding.weight"
    )
    assert canonical_state_key("model.encoder.w_pos") == "encoder.positional_encoder.position_enc"
    assert (
        canonical_state_key("model.encoder.encoder.layers.0.norm_sublayer1.1.weight")
        == "encoder.layers.0.norm_sublayer1.batchnorm.weight"
    )


def test_weight_audit_matches_known_wrapper_prefixes() -> None:
    source = {
        "model.encoder.weight": FakeTensor((2, 3)),
        "model.encoder.bias": FakeTensor((2,)),
    }
    target = {
        "encoder.weight": FakeTensor((2, 3)),
        "encoder.bias": FakeTensor((2,)),
    }

    report, load_state = audit_matching_encoder_weights(target, source)

    assert report.loaded_numel == 8
    assert report.source_encoder_numel == 8
    assert report.loaded_numel_ratio == 1.0
    assert report.unallowed_mismatches == []
    assert sorted(load_state) == ["encoder.bias", "encoder.weight"]


def test_weight_loader_fails_closed_on_shape_mismatch() -> None:
    module = FakeModule({"encoder.weight": FakeTensor((3, 3))})
    source = {"encoder.weight": FakeTensor((2, 3))}

    with pytest.raises(WeightLoadError, match="ratio"):
        load_matching_encoder_weights(module, source, minimum_loaded_numel_ratio=0.8)

    assert module.loaded is None


def test_weight_loader_requires_explicit_allowlist_for_unmatched_buffers() -> None:
    module = FakeModule(
        {
            "encoder.weight": FakeTensor((2, 2)),
            "new_buffer": FakeTensor((1,)),
        }
    )
    source = {
        "encoder.weight": FakeTensor((2, 2)),
        "old_buffer": FakeTensor((1,)),
    }

    with pytest.raises(WeightLoadError, match="Unallowed"):
        load_matching_encoder_weights(module, source, minimum_loaded_numel_ratio=0.75)

    report = load_matching_encoder_weights(
        module,
        source,
        minimum_loaded_numel_ratio=0.75,
        allowlist=("missing:new_buffer", "unexpected:old_buffer"),
    )
    assert report.loaded_numel_ratio == 0.8
    assert len(report.allowed_mismatches) == 2
    assert module.loaded is not None


def test_parameter_only_ratio_excludes_buffers_from_denominator() -> None:
    source = {
        "encoder.weight": FakeTensor((2, 2)),
        "encoder.running_mean": FakeTensor((100,)),
    }
    target = {
        "encoder.weight": FakeTensor((2, 2)),
        "encoder.running_mean": FakeTensor((99,)),
    }

    report, _ = audit_matching_encoder_weights(
        target,
        source,
        allowlist=("shape:encoder.running_mean",),
        source_parameter_names={"encoder.weight"},
    )
    assert report.source_encoder_numel == 4
    assert report.loaded_numel_ratio == 1.0
