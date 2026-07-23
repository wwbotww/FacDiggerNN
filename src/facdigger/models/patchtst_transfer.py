"""Strict initialization of the E2 alpha model from the pinned ETTh1 encoder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import transformers
from huggingface_hub import hf_hub_download
from transformers import PatchTSTConfig, PatchTSTForPretraining

from facdigger.data.snapshots import sha256_file
from facdigger.models.patchtst_adapter import (
    CanonicalPatchTSTConfig,
    WeightLoadError,
    canonical_state_key,
    load_matching_encoder_weights,
)
from facdigger.models.patchtst_alpha import PatchTSTAlphaModel


def module_fingerprint(module: torch.nn.Module) -> str:
    """Hash names, dtype, shape and exact tensor bytes for parameters and buffers."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _build_alpha_model(
    model_config: Any, context_length: int, num_channels: int
) -> PatchTSTAlphaModel:
    return PatchTSTAlphaModel(
        context_length=context_length,
        num_input_channels=num_channels,
        patch_length=model_config.patch_length,
        patch_stride=model_config.patch_stride,
        d_model=model_config.d_model,
        num_attention_heads=model_config.num_attention_heads,
        num_hidden_layers=model_config.num_hidden_layers,
        ffn_dim=model_config.ffn_dim,
        dropout=model_config.dropout,
        attention_dropout=model_config.attention_dropout,
        positional_dropout=model_config.positional_dropout,
        path_dropout=model_config.path_dropout,
        ff_dropout=model_config.ff_dropout,
        norm_type=model_config.norm_type,
        pre_norm=model_config.pre_norm,
        scaling=model_config.scaling,
        alpha_hidden_dim=model_config.alpha_hidden_dim,
        alpha_dropout=model_config.alpha_dropout,
    )


def _validate_source_architecture(
    canonical: CanonicalPatchTSTConfig,
    model_config: Any,
    *,
    context_length: int,
    num_channels: int,
) -> dict[str, dict[str, Any]]:
    expected = {
        "context_length": context_length,
        "num_input_channels": num_channels,
        "patch_length": model_config.patch_length,
        "patch_stride": model_config.patch_stride,
        "d_model": model_config.d_model,
        "num_attention_heads": model_config.num_attention_heads,
        "num_hidden_layers": model_config.num_hidden_layers,
        "ffn_dim": model_config.ffn_dim,
        "dropout": model_config.dropout,
        "attention_dropout": model_config.attention_dropout,
        "positional_dropout": model_config.positional_dropout,
        "path_dropout": model_config.path_dropout,
        "ff_dropout": model_config.ff_dropout,
        "norm_type": model_config.norm_type,
        "pre_norm": model_config.pre_norm,
        "scaling": model_config.scaling,
        "share_embedding": True,
        "channel_attention": False,
        "use_cls_token": False,
        "positional_encoding_type": "sincos",
    }
    checks = {
        name: {
            "source": getattr(canonical, name),
            "target": target,
            "matches": getattr(canonical, name) == target,
        }
        for name, target in expected.items()
    }
    mismatches = [name for name, check in checks.items() if not check["matches"]]
    if mismatches:
        raise WeightLoadError(
            "E2 target architecture differs from pinned source: " + ", ".join(mismatches)
        )
    return checks


def initialize_transferred_alpha_model(
    *,
    model_config: Any,
    source_config: Any,
    context_length: int,
    num_channels: int,
) -> tuple[PatchTSTAlphaModel, dict[str, Any]]:
    config_path = Path(
        hf_hub_download(
            repo_id=source_config.model_id,
            filename="config.json",
            revision=source_config.revision,
            local_files_only=source_config.local_files_only,
        )
    )
    weights_path = Path(
        hf_hub_download(
            repo_id=source_config.model_id,
            filename="pytorch_model.bin",
            revision=source_config.revision,
            local_files_only=source_config.local_files_only,
        )
    )
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    canonical = CanonicalPatchTSTConfig.from_checkpoint_dict(raw_config)
    compatibility = _validate_source_architecture(
        canonical,
        model_config,
        context_length=context_length,
        num_channels=num_channels,
    )

    target = _build_alpha_model(model_config, context_length, num_channels)
    random_target_fingerprint = module_fingerprint(target.backbone)
    source_model = PatchTSTForPretraining(PatchTSTConfig(**canonical.to_transformers_kwargs()))
    source_backbone = source_model.base_model
    raw_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        raw_state = raw_state["state_dict"]
    if not isinstance(raw_state, dict):
        raise WeightLoadError("source checkpoint does not contain a state dict")
    raw_encoder_state = {
        key: value
        for key, value in raw_state.items()
        if canonical_state_key(key).startswith("encoder.")
    }
    source_parameter_names = {
        canonical_state_key(name) for name, _ in source_backbone.named_parameters()
    }
    raw_parameter_names = {
        key for key in raw_encoder_state if canonical_state_key(key) in source_parameter_names
    }
    checkpoint_report = load_matching_encoder_weights(
        target_backbone=source_backbone,
        source_encoder_state=raw_encoder_state,
        minimum_loaded_numel_ratio=source_config.minimum_loaded_numel_ratio,
        allowlist=tuple(source_config.allowlist),
        source_parameter_names=raw_parameter_names,
    )
    source_fingerprint = module_fingerprint(source_backbone)
    transfer_report = load_matching_encoder_weights(
        target_backbone=target.backbone,
        source_encoder_state=source_backbone.state_dict(),
        minimum_loaded_numel_ratio=source_config.minimum_loaded_numel_ratio,
        allowlist=tuple(source_config.allowlist),
        source_parameter_names={name for name, _ in source_backbone.named_parameters()},
    )
    transferred_fingerprint = module_fingerprint(target.backbone)
    if source_fingerprint != transferred_fingerprint:
        raise WeightLoadError("target backbone fingerprint differs after encoder transfer")
    if random_target_fingerprint == transferred_fingerprint:
        raise WeightLoadError("encoder transfer did not change the random target fingerprint")
    audit = {
        "schema_version": 1,
        "source_model": source_config.model_id,
        "source_revision": source_config.revision,
        "source_weights_sha256": sha256_file(weights_path),
        "transformers_version": transformers.__version__,
        "canonical_source_config": canonical.model_dump(mode="json"),
        "architecture_compatibility": compatibility,
        "checkpoint_to_library": checkpoint_report.to_dict(),
        "source_backbone_to_target": transfer_report.to_dict(),
        "fingerprints": {
            "random_target_before_transfer": random_target_fingerprint,
            "source_backbone_after_checkpoint": source_fingerprint,
            "target_backbone_after_transfer": transferred_fingerprint,
        },
    }
    return target, audit
