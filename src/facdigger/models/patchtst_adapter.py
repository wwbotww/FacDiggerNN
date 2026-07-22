"""PatchTST configuration normalization and auditable encoder weight transfer.

This module intentionally does not import torch or transformers at import time. That keeps
configuration and audit unit tests runnable before the optional model environment is installed.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CanonicalPatchTSTConfig(BaseModel):
    """Version-independent representation of the source PatchTST architecture."""

    model_config = ConfigDict(extra="forbid")

    num_input_channels: int = Field(gt=0)
    context_length: int = Field(gt=0)
    patch_length: int = Field(gt=0)
    patch_stride: int = Field(gt=0)
    num_hidden_layers: int = Field(gt=0)
    d_model: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    ffn_dim: int = Field(gt=0)
    share_embedding: bool
    channel_attention: bool
    dropout: float = Field(ge=0, lt=1)
    attention_dropout: float = Field(ge=0, lt=1)
    positional_dropout: float = Field(ge=0, lt=1)
    path_dropout: float = Field(ge=0, lt=1)
    ff_dropout: float = Field(ge=0, lt=1)
    head_dropout: float = Field(ge=0, lt=1)
    norm_type: str
    pre_norm: bool
    positional_encoding_type: str
    use_cls_token: bool
    share_projection: bool
    scaling: str | bool | None
    do_mask_input: bool
    mask_type: str
    random_mask_ratio: float = Field(ge=0, le=1)
    mask_value: float
    prediction_length: int = Field(gt=0)

    @classmethod
    def from_checkpoint_dict(cls, raw: Mapping[str, Any]) -> CanonicalPatchTSTConfig:
        """Normalize both the 2023 IBM schema and current Transformers field names."""

        def pick(*names: str, default: Any = None, required: bool = False) -> Any:
            for name in names:
                if name in raw and raw[name] is not None:
                    return raw[name]
            if required:
                raise ValueError(f"Checkpoint config is missing all aliases: {names}")
            return default

        norm_type = str(pick("norm_type", "norm", default="batchnorm")).lower()
        if norm_type == "batchnorm1d":
            norm_type = "batchnorm"

        return cls(
            num_input_channels=pick("num_input_channels", required=True),
            context_length=pick("context_length", required=True),
            patch_length=pick("patch_length", required=True),
            patch_stride=pick("patch_stride", "stride", required=True),
            num_hidden_layers=pick("num_hidden_layers", "encoder_layers", required=True),
            d_model=pick("d_model", required=True),
            num_attention_heads=pick(
                "num_attention_heads", "encoder_attention_heads", required=True
            ),
            ffn_dim=pick("ffn_dim", "encoder_ffn_dim", required=True),
            share_embedding=pick("share_embedding", "shared_embedding", default=True),
            channel_attention=pick("channel_attention", default=False),
            dropout=pick("dropout", default=0.0),
            attention_dropout=pick("attention_dropout", default=0.0),
            positional_dropout=pick("positional_dropout", default=0.0),
            path_dropout=pick("path_dropout", "dropout_path", default=0.0),
            ff_dropout=pick("ff_dropout", default=0.0),
            head_dropout=pick("head_dropout", default=0.0),
            norm_type=norm_type,
            pre_norm=pick("pre_norm", default=True),
            positional_encoding_type=pick(
                "positional_encoding_type", "positional_encoding", default="sincos"
            ),
            use_cls_token=pick("use_cls_token", default=False),
            share_projection=pick("share_projection", "shared_projection", default=True),
            scaling=pick("scaling", default="std"),
            do_mask_input=pick("do_mask_input", "mask_input", default=True),
            mask_type=pick("mask_type", default="random"),
            random_mask_ratio=pick("random_mask_ratio", "mask_ratio", default=0.4),
            mask_value=pick("mask_value", default=0.0),
            prediction_length=pick("prediction_length", default=24),
        )

    def to_transformers_kwargs(self) -> dict[str, Any]:
        """Map canonical fields to the current Transformers PatchTSTConfig schema."""

        return self.model_dump()


class WeightLoadError(RuntimeError):
    """Raised when an encoder transfer fails an explicit audit gate."""


@dataclass(frozen=True)
class ShapeMismatch:
    key: str
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]


@dataclass
class WeightLoadReport:
    loaded_numel: int
    source_encoder_numel: int
    loaded_numel_ratio: float
    loaded_keys: list[str] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    shape_mismatches: list[ShapeMismatch] = field(default_factory=list)
    allowed_mismatches: list[str] = field(default_factory=list)
    unallowed_mismatches: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape_mismatches"] = [asdict(item) for item in self.shape_mismatches]
        return payload


_KNOWN_PREFIXES = ("module.", "base_model.", "backbone.", "model.")


def canonical_state_key(key: str) -> str:
    """Normalize wrapper prefixes and audited 2023-to-current PatchTST key migrations."""

    previous = None
    normalized = key
    while normalized != previous:
        previous = normalized
        for prefix in _KNOWN_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break
    normalized = normalized.replace("encoder.encoder.", "encoder.")
    normalized = normalized.replace("encoder.w_p.", "encoder.embedder.input_embedding.")
    if normalized == "encoder.w_pos":
        normalized = "encoder.positional_encoder.position_enc"
    normalized = normalized.replace(".norm_sublayer1.1.", ".norm_sublayer1.batchnorm.")
    normalized = normalized.replace(".norm_sublayer3.1.", ".norm_sublayer3.batchnorm.")
    return normalized


def _shape(value: Any) -> tuple[int, ...]:
    if not hasattr(value, "shape"):
        raise TypeError(f"State value has no shape attribute: {type(value).__name__}")
    return tuple(int(item) for item in value.shape)


def _numel(value: Any) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    size = 1
    for dimension in _shape(value):
        size *= dimension
    return size


def _index_state(state: Mapping[str, Any]) -> dict[str, tuple[str, Any]]:
    indexed: dict[str, tuple[str, Any]] = {}
    for original_key, value in state.items():
        normalized = canonical_state_key(original_key)
        if normalized in indexed:
            other = indexed[normalized][0]
            raise WeightLoadError(
                f"State key collision after prefix normalization: {other!r} and {original_key!r}"
            )
        indexed[normalized] = (original_key, value)
    return indexed


def _is_allowed(key: str, allowlist: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(key, pattern) for pattern in allowlist)


def audit_matching_encoder_weights(
    target_state: Mapping[str, Any],
    source_encoder_state: Mapping[str, Any],
    allowlist: tuple[str, ...] = (),
    source_parameter_names: set[str] | None = None,
) -> tuple[WeightLoadReport, dict[str, Any]]:
    """Match encoder state by normalized name and exact shape, without mutating a model."""

    source_index = _index_state(source_encoder_state)
    target_index = _index_state(target_state)
    normalized_parameter_names = (
        {canonical_state_key(key) for key in source_parameter_names}
        if source_parameter_names is not None
        else set(source_index)
    )
    source_numel = sum(
        _numel(value)
        for normalized_key, (_, value) in source_index.items()
        if normalized_key in normalized_parameter_names
    )
    loaded_numel = 0
    loaded_keys: list[str] = []
    unexpected_keys: list[str] = []
    shape_mismatches: list[ShapeMismatch] = []
    target_load_state: dict[str, Any] = {}

    for normalized_key, (source_key, source_value) in source_index.items():
        target_entry = target_index.get(normalized_key)
        if target_entry is None:
            unexpected_keys.append(source_key)
            continue
        target_key, target_value = target_entry
        if _shape(source_value) != _shape(target_value):
            shape_mismatches.append(
                ShapeMismatch(normalized_key, _shape(source_value), _shape(target_value))
            )
            continue
        target_load_state[target_key] = source_value
        loaded_keys.append(normalized_key)
        if normalized_key in normalized_parameter_names:
            loaded_numel += _numel(source_value)

    loaded_normalized = set(loaded_keys)
    missing_keys = [
        original_key
        for normalized_key, (original_key, _) in target_index.items()
        if normalized_key not in loaded_normalized
    ]

    mismatch_names = (
        [f"missing:{canonical_state_key(key)}" for key in missing_keys]
        + [f"unexpected:{canonical_state_key(key)}" for key in unexpected_keys]
        + [f"shape:{item.key}" for item in shape_mismatches]
    )
    allowed = [name for name in mismatch_names if _is_allowed(name, allowlist)]
    unallowed = [name for name in mismatch_names if not _is_allowed(name, allowlist)]
    ratio = loaded_numel / source_numel if source_numel else 0.0
    report = WeightLoadReport(
        loaded_numel=loaded_numel,
        source_encoder_numel=source_numel,
        loaded_numel_ratio=ratio,
        loaded_keys=sorted(loaded_keys),
        missing_keys=sorted(missing_keys),
        unexpected_keys=sorted(unexpected_keys),
        shape_mismatches=sorted(shape_mismatches, key=lambda item: item.key),
        allowed_mismatches=sorted(allowed),
        unallowed_mismatches=sorted(unallowed),
    )
    return report, target_load_state


def load_matching_encoder_weights(
    target_backbone: Any,
    source_encoder_state: Mapping[str, Any],
    minimum_loaded_numel_ratio: float = 0.80,
    allowlist: tuple[str, ...] = (),
    source_parameter_names: set[str] | None = None,
) -> WeightLoadReport:
    """Audit and load matching encoder weights, failing closed on any gate violation."""

    if not 0 <= minimum_loaded_numel_ratio <= 1:
        raise ValueError("minimum_loaded_numel_ratio must be between 0 and 1")
    report, load_state = audit_matching_encoder_weights(
        target_state=target_backbone.state_dict(),
        source_encoder_state=source_encoder_state,
        allowlist=allowlist,
        source_parameter_names=source_parameter_names,
    )
    if report.loaded_numel_ratio < minimum_loaded_numel_ratio:
        raise WeightLoadError(
            "Encoder loaded-numel ratio "
            f"{report.loaded_numel_ratio:.4f} is below {minimum_loaded_numel_ratio:.4f}"
        )
    if report.unallowed_mismatches:
        preview = ", ".join(report.unallowed_mismatches[:5])
        raise WeightLoadError(f"Unallowed encoder weight mismatches: {preview}")
    target_backbone.load_state_dict(load_state, strict=False)
    return report
