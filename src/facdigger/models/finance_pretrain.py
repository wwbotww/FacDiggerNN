"""Continuous-span self-supervision for finance-native Transformer encoders."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from facdigger.models.finance_patch_transformer import (
    LocalTemporalEncoder,
    MarketContextEncoder,
)


@dataclass(frozen=True)
class FinancePretrainingOutput:
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    future_summary_loss: torch.Tensor
    masked_observed_elements: int
    observed_future_summary_elements: int
    masked_patch_ratio: float


def _random_contiguous_patch_mask(
    *,
    num_patches: int,
    target_patches: int,
    minimum_span: int,
    maximum_span: int,
    device: torch.device,
) -> torch.Tensor:
    selected = torch.zeros(num_patches, dtype=torch.bool, device=device)
    if target_patches <= 0:
        return selected
    maximum_attempts = max(num_patches * 4, 16)
    for _ in range(maximum_attempts):
        if int(selected.sum()) >= target_patches:
            break
        length = int(
            torch.randint(
                minimum_span,
                maximum_span + 1,
                (),
                device=device,
            )
        )
        length = min(length, num_patches)
        start = int(torch.randint(0, num_patches - length + 1, (), device=device))
        selected[start : start + length] = True
    selected_count = int(selected.sum())
    if selected_count < target_patches:
        remaining = torch.where(~selected)[0]
        choice = remaining[torch.randperm(len(remaining), device=device)]
        selected[choice[: target_patches - selected_count]] = True
    # Do not trim an overshooting span: trimming would turn a continuous mask
    # back into isolated random patches.  The realized ratio is audited.
    return selected


def contiguous_patch_element_mask(
    observed_mask: torch.Tensor,
    *,
    patch_length: int,
    patch_stride: int,
    sequence_start: int,
    mask_ratio: float,
    minimum_span_patches: int,
    maximum_span_patches: int,
    shared_channel_fraction: float,
) -> tuple[torch.Tensor, float]:
    """Create shared and channel-specific contiguous patch masks."""

    if observed_mask.ndim != 3:
        raise ValueError("observed_mask must be [B,L,C]")
    _, context_length, channels = observed_mask.shape
    if not 0 < mask_ratio < 1:
        raise ValueError("mask_ratio must be inside (0, 1)")
    if not 0 <= shared_channel_fraction <= 1:
        raise ValueError("shared_channel_fraction must be inside [0, 1]")
    usable = context_length - sequence_start - patch_length
    if usable < 0:
        raise ValueError("patch geometry exceeds context length")
    num_patches = usable // patch_stride + 1
    target = max(1, min(num_patches - 1, round(num_patches * mask_ratio)))
    shared_target = round(target * shared_channel_fraction)
    independent_target = target - shared_target
    patch_mask = torch.zeros(
        observed_mask.shape[0],
        channels,
        num_patches,
        dtype=torch.bool,
        device=observed_mask.device,
    )
    for batch_index in range(observed_mask.shape[0]):
        shared = _random_contiguous_patch_mask(
            num_patches=num_patches,
            target_patches=shared_target,
            minimum_span=minimum_span_patches,
            maximum_span=maximum_span_patches,
            device=observed_mask.device,
        )
        for channel in range(channels):
            independent = _random_contiguous_patch_mask(
                num_patches=num_patches,
                target_patches=independent_target,
                minimum_span=minimum_span_patches,
                maximum_span=maximum_span_patches,
                device=observed_mask.device,
            )
            patch_mask[batch_index, channel] = shared | independent

    element_mask = torch.zeros_like(observed_mask, dtype=torch.bool)
    for patch_index in range(num_patches):
        start = sequence_start + patch_index * patch_stride
        stop = start + patch_length
        element_mask[:, start:stop, :] |= patch_mask[:, :, patch_index].unsqueeze(1)
    actual_patch_ratio = float(patch_mask.float().mean().detach().cpu())
    return element_mask, actual_patch_ratio


def future_summary_targets(
    future_values: torch.Tensor,
    future_observed_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return channel-balanced normalized sum/mean/std targets."""

    if future_values.shape != future_observed_mask.shape or future_values.ndim != 3:
        raise ValueError("future values and mask must share [B,H,C] shape")
    observed = future_observed_mask.bool()
    weights = observed.to(future_values.dtype)
    count = weights.sum(dim=1)
    mean = (future_values * weights).sum(dim=1) / count.clamp_min(1.0)
    centered = (future_values - mean.unsqueeze(1)) * weights
    std = (centered.square().sum(dim=1) / count.clamp_min(1.0)).sqrt()
    normalized_sum = (future_values * weights).sum(dim=1) / count.clamp_min(1.0).sqrt()
    target = torch.stack([normalized_sum, mean, std], dim=-1)
    valid = (count > 0).unsqueeze(-1).expand_as(target)
    return target, valid


def _channel_balanced_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    delta: float,
    element_dimension: int,
) -> tuple[torch.Tensor, int]:
    if prediction.shape != target.shape or target.shape != valid.shape:
        raise ValueError("prediction, target and valid mask shapes differ")
    difference = prediction.float() - target.float()
    absolute = difference.abs()
    element_loss = torch.where(
        absolute <= delta,
        0.5 * difference.square(),
        delta * (absolute - 0.5 * delta),
    )
    if prediction.ndim != 3 or element_dimension not in (1, 2):
        raise ValueError("channel-balanced Huber expects a three-dimensional tensor")
    weights = valid.to(element_loss.dtype)
    counts = weights.sum(dim=element_dimension)
    reduced = (element_loss * weights).sum(dim=element_dimension) / counts.clamp_min(
        1.0
    )
    valid_channels = counts > 0
    if not bool(valid_channels.any()):
        raise RuntimeError("self-supervised batch has no observed target elements")
    return reduced.masked_select(valid_channels).mean(), int(valid.sum().detach().cpu())


class FinanceNativePretrainer(nn.Module):
    """Train local and market encoders without importing the supervised labels."""

    def __init__(
        self,
        *,
        local_encoder: LocalTemporalEncoder,
        market_encoder: MarketContextEncoder,
        context_length: int,
        num_local_channels: int,
        num_future_local_channels: int,
        num_market_channels: int,
        embedding_dim: int,
        reconstruction_weight: float,
        future_summary_weight: float,
        huber_delta: float,
        mask_ratio: float,
        minimum_span_patches: int,
        maximum_span_patches: int,
        shared_channel_fraction: float,
    ) -> None:
        super().__init__()
        self.local_encoder = local_encoder
        self.market_encoder = market_encoder
        self.context_length = context_length
        self.num_local_channels = num_local_channels
        self.num_future_local_channels = num_future_local_channels
        self.num_market_channels = num_market_channels
        self.reconstruction_weight = reconstruction_weight
        self.future_summary_weight = future_summary_weight
        self.huber_delta = huber_delta
        self.mask_ratio = mask_ratio
        self.minimum_span_patches = minimum_span_patches
        self.maximum_span_patches = maximum_span_patches
        self.shared_channel_fraction = shared_channel_fraction
        self.local_reconstruction_head = nn.Linear(
            embedding_dim, context_length * num_local_channels
        )
        self.local_future_head = nn.Linear(
            embedding_dim, num_future_local_channels * 3
        )
        self.market_reconstruction_head = nn.Linear(
            embedding_dim, context_length * num_market_channels
        )
        self.market_future_head = nn.Linear(embedding_dim, num_market_channels * 3)

    def _mask(
        self,
        observed_mask: torch.Tensor,
        *,
        patch_length: int,
        patch_stride: int,
        sequence_start: int,
    ) -> tuple[torch.Tensor, float]:
        return contiguous_patch_element_mask(
            observed_mask,
            patch_length=patch_length,
            patch_stride=patch_stride,
            sequence_start=sequence_start,
            mask_ratio=self.mask_ratio,
            minimum_span_patches=self.minimum_span_patches,
            maximum_span_patches=self.maximum_span_patches,
            shared_channel_fraction=self.shared_channel_fraction,
        )

    def _objective(
        self,
        *,
        embedding: torch.Tensor,
        values: torch.Tensor,
        observed_mask: torch.Tensor,
        future_values: torch.Tensor,
        future_observed_mask: torch.Tensor,
        element_mask: torch.Tensor,
        reconstruction_head: nn.Linear,
        future_head: nn.Linear,
        num_channels: int,
        future_channels: int,
        masked_patch_ratio: float,
    ) -> FinancePretrainingOutput:
        reconstruction = reconstruction_head(embedding).reshape(
            -1, self.context_length, num_channels
        )
        reconstruction_loss, masked_count = _channel_balanced_huber(
            reconstruction,
            values,
            observed_mask.bool() & element_mask,
            delta=self.huber_delta,
            element_dimension=1,
        )
        summary, summary_valid = future_summary_targets(
            future_values, future_observed_mask
        )
        summary_prediction = future_head(embedding).reshape(-1, future_channels, 3)
        future_loss, summary_count = _channel_balanced_huber(
            summary_prediction,
            summary,
            summary_valid,
            delta=self.huber_delta,
            element_dimension=2,
        )
        loss = (
            self.reconstruction_weight * reconstruction_loss
            + self.future_summary_weight * future_loss
        )
        return FinancePretrainingOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            future_summary_loss=future_loss,
            masked_observed_elements=masked_count,
            observed_future_summary_elements=summary_count,
            masked_patch_ratio=masked_patch_ratio,
        )

    def forward_local(
        self,
        values: torch.Tensor,
        observed_mask: torch.Tensor,
        future_values: torch.Tensor,
        future_observed_mask: torch.Tensor,
    ) -> FinancePretrainingOutput:
        element_mask, ratio = self._mask(
            observed_mask,
            patch_length=self.local_encoder.patch_length,
            patch_stride=self.local_encoder.patch_stride,
            sequence_start=int(self.local_encoder.backbone.patchifier.sequence_start),
        )
        corrupted_observed = observed_mask.bool() & ~element_mask
        corrupted_values = values.masked_fill(element_mask, 0.0)
        embedding = self.local_encoder(corrupted_values, corrupted_observed).embedding
        return self._objective(
            embedding=embedding,
            values=values,
            observed_mask=observed_mask,
            future_values=future_values,
            future_observed_mask=future_observed_mask,
            element_mask=element_mask,
            reconstruction_head=self.local_reconstruction_head,
            future_head=self.local_future_head,
            num_channels=self.num_local_channels,
            future_channels=self.num_future_local_channels,
            masked_patch_ratio=ratio,
        )

    def forward_market(
        self,
        values: torch.Tensor,
        observed_mask: torch.Tensor,
        future_values: torch.Tensor,
        future_observed_mask: torch.Tensor,
    ) -> FinancePretrainingOutput:
        element_mask, ratio = self._mask(
            observed_mask,
            patch_length=self.market_encoder.patch_length,
            patch_stride=self.market_encoder.patch_stride,
            sequence_start=int(self.market_encoder.backbone.patchifier.sequence_start),
        )
        corrupted_observed = observed_mask.bool() & ~element_mask
        corrupted_values = values.masked_fill(element_mask, 0.0)
        embedding = self.market_encoder(corrupted_values, corrupted_observed)
        return self._objective(
            embedding=embedding,
            values=values,
            observed_mask=observed_mask,
            future_values=future_values,
            future_observed_mask=future_observed_mask,
            element_mask=element_mask,
            reconstruction_head=self.market_reconstruction_head,
            future_head=self.market_future_head,
            num_channels=self.num_market_channels,
            future_channels=self.num_market_channels,
            masked_patch_ratio=ratio,
        )
