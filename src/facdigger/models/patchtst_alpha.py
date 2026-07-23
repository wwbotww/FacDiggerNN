"""Randomly initialized PatchTST backbone with a finance alpha head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import PatchTSTConfig, PatchTSTModel


@dataclass(frozen=True)
class EncoderOutput:
    hidden: torch.Tensor
    channel_mask: torch.Tensor
    patch_mask: torch.Tensor


@dataclass(frozen=True)
class AlphaModelOutput:
    score: torch.Tensor
    encoder: EncoderOutput


class PatchTSTEncoderOutputAdapter:
    """Normalize and validate the external backbone's output layout."""

    @staticmethod
    def adapt(
        output: Any,
        observed_mask: torch.Tensor,
        *,
        patch_length: int,
        patch_stride: int,
    ) -> EncoderOutput:
        hidden = output.last_hidden_state
        if hidden.ndim != 4:
            raise ValueError(f"PatchTST hidden state must be [B,C,P,D], got {hidden.shape}")
        if observed_mask.ndim != 3:
            raise ValueError("observed_mask must be [B,L,C]")
        patch_mask = (
            observed_mask.transpose(1, 2)
            .unfold(dimension=-1, size=patch_length, step=patch_stride)
            .any(dim=-1)
        )
        channel_mask = observed_mask.any(dim=1)
        expected = hidden.shape[:3]
        if patch_mask.shape != expected:
            raise ValueError(
                f"derived patch mask {patch_mask.shape} does not match hidden state {expected}"
            )
        return EncoderOutput(hidden=hidden, channel_mask=channel_mask, patch_mask=patch_mask)


class AlphaHead(nn.Module):
    def __init__(
        self,
        *,
        num_channels: int,
        d_model: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.channel_norm = nn.LayerNorm(d_model)
        self.projection = nn.Sequential(
            nn.Linear(num_channels * d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, encoder: EncoderOutput) -> torch.Tensor:
        weights = encoder.patch_mask.unsqueeze(-1).to(encoder.hidden.dtype)
        pooled = (encoder.hidden * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        pooled = self.channel_norm(pooled)
        pooled = pooled * encoder.channel_mask.unsqueeze(-1).to(pooled.dtype)
        return self.projection(pooled.flatten(start_dim=1)).squeeze(-1)


class PatchTSTAlphaModel(nn.Module):
    def __init__(
        self,
        *,
        context_length: int,
        num_input_channels: int,
        patch_length: int,
        patch_stride: int,
        d_model: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        ffn_dim: int,
        dropout: float,
        attention_dropout: float,
        positional_dropout: float,
        path_dropout: float,
        ff_dropout: float,
        norm_type: str,
        pre_norm: bool,
        scaling: str | bool | None,
        alpha_hidden_dim: int,
        alpha_dropout: float,
    ) -> None:
        super().__init__()
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        config = PatchTSTConfig(
            context_length=context_length,
            num_input_channels=num_input_channels,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            positional_dropout=positional_dropout,
            path_dropout=path_dropout,
            ff_dropout=ff_dropout,
            norm_type=norm_type,
            pre_norm=pre_norm,
            scaling=scaling,
            do_mask_input=False,
            use_cls_token=False,
            share_embedding=True,
            channel_attention=False,
        )
        self.backbone = PatchTSTModel(config)
        self.alpha_head = AlphaHead(
            num_channels=num_input_channels,
            d_model=d_model,
            hidden_dim=alpha_hidden_dim,
            dropout=alpha_dropout,
        )

    def forward(self, values: torch.Tensor, observed_mask: torch.Tensor) -> AlphaModelOutput:
        if values.shape != observed_mask.shape:
            raise ValueError("values and observed_mask must have the same [B,L,C] shape")
        backbone_output = self.backbone(
            past_values=values,
            past_observed_mask=observed_mask,
            return_dict=True,
        )
        encoder = PatchTSTEncoderOutputAdapter.adapt(
            backbone_output,
            observed_mask,
            patch_length=self.patch_length,
            patch_stride=self.patch_stride,
        )
        return AlphaModelOutput(score=self.alpha_head(encoder), encoder=encoder)
