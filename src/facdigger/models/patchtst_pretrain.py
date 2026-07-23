"""Observed-only masked reconstruction for financial PatchTST pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn
from transformers import PatchTSTConfig, PatchTSTForPretraining

from facdigger.models.patchtst_adapter import load_matching_encoder_weights
from facdigger.models.patchtst_transfer import (
    initialize_transferred_alpha_model,
    module_fingerprint,
)


@dataclass(frozen=True)
class MaskedReconstructionOutput:
    loss: torch.Tensor
    prediction: torch.Tensor
    patch_mask: torch.Tensor
    valid_element_count: int


class FinancialPatchTSTPretrainer(nn.Module):
    """PatchTST pretrainer whose loss excludes every unobserved input element."""

    def __init__(
        self,
        *,
        context_length: int,
        num_input_channels: int,
        model_config: Any,
        mask_ratio: float,
        loss: Literal["mse", "huber"],
        huber_delta: float,
    ) -> None:
        super().__init__()
        self.patch_length = int(model_config.patch_length)
        self.patch_stride = int(model_config.patch_stride)
        self.loss_name = loss
        self.huber_delta = huber_delta
        config = PatchTSTConfig(
            context_length=context_length,
            num_input_channels=num_input_channels,
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
            do_mask_input=True,
            mask_type="random",
            random_mask_ratio=mask_ratio,
            use_cls_token=False,
            share_embedding=True,
            channel_attention=False,
        )
        self.pretrainer = PatchTSTForPretraining(config)

    @property
    def backbone(self) -> nn.Module:
        return self.pretrainer.model

    def forward(
        self, values: torch.Tensor, observed_mask: torch.Tensor
    ) -> MaskedReconstructionOutput:
        if values.shape != observed_mask.shape or values.ndim != 3:
            raise ValueError("values and observed_mask must share [B,L,C] shape")
        model_output = self.pretrainer.model(
            past_values=values,
            past_observed_mask=observed_mask,
            return_dict=True,
        )
        prediction = self.pretrainer.head(model_output.last_hidden_state)
        observed_elements = (
            observed_mask.transpose(1, 2)
            .unfold(-1, self.patch_length, self.patch_stride)
            .to(dtype=torch.bool)
        )
        valid = model_output.mask.to(dtype=torch.bool).unsqueeze(-1) & observed_elements
        valid_count = int(valid.sum().detach().cpu())
        if valid_count == 0:
            raise RuntimeError("masked reconstruction batch has no observed target elements")
        difference = prediction - model_output.patch_input
        if self.loss_name == "mse":
            element_loss = difference.square()
        else:
            absolute = difference.abs()
            element_loss = torch.where(
                absolute <= self.huber_delta,
                0.5 * difference.square(),
                self.huber_delta * (absolute - 0.5 * self.huber_delta),
            )
        loss = element_loss.masked_select(valid).mean()
        return MaskedReconstructionOutput(
            loss=loss,
            prediction=prediction,
            patch_mask=model_output.mask,
            valid_element_count=valid_count,
        )


def initialize_financial_pretrainer(
    *,
    model_config: Any,
    source_config: Any,
    pretraining_config: Any,
    context_length: int,
    num_channels: int,
) -> tuple[FinancialPatchTSTPretrainer, dict[str, Any]]:
    """Initialize a masked financial model through the audited ETTh1 transfer path."""

    alpha_model, source_audit = initialize_transferred_alpha_model(
        model_config=model_config,
        source_config=source_config,
        context_length=context_length,
        num_channels=num_channels,
    )
    target = FinancialPatchTSTPretrainer(
        context_length=context_length,
        num_input_channels=num_channels,
        model_config=model_config,
        mask_ratio=pretraining_config.mask_ratio,
        loss=pretraining_config.loss,
        huber_delta=pretraining_config.huber_delta,
    )
    random_fingerprint = module_fingerprint(target.backbone)
    report = load_matching_encoder_weights(
        target_backbone=target.backbone,
        source_encoder_state=alpha_model.backbone.state_dict(),
        minimum_loaded_numel_ratio=source_config.minimum_loaded_numel_ratio,
        allowlist=tuple(source_config.allowlist),
        source_parameter_names={name for name, _ in alpha_model.backbone.named_parameters()},
    )
    transferred_fingerprint = module_fingerprint(target.backbone)
    expected_fingerprint = module_fingerprint(alpha_model.backbone)
    if transferred_fingerprint != expected_fingerprint:
        raise RuntimeError("financial pretrainer backbone differs after ETTh1 transfer")
    if transferred_fingerprint == random_fingerprint:
        raise RuntimeError("ETTh1 transfer did not change the financial pretrainer")
    audit = {
        **source_audit,
        "schema_version": 2,
        "source_backbone_to_financial_pretrainer": report.to_dict(),
        "financial_pretraining": {
            "mask_type": "random",
            "mask_ratio": pretraining_config.mask_ratio,
            "loss": pretraining_config.loss,
            "observed_elements_only": True,
        },
        "fingerprints": {
            **source_audit["fingerprints"],
            "random_financial_pretrainer_before_transfer": random_fingerprint,
            "financial_pretrainer_after_transfer": transferred_fingerprint,
        },
    }
    return target, audit
