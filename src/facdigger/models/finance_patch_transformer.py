"""Finance-native temporal and full-date cross-sectional Transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import PatchTSTConfig, PatchTSTModel

from facdigger.models.patchtst_alpha import PatchTSTEncoderOutputAdapter


@dataclass(frozen=True)
class LocalEncoderOutput:
    embedding: torch.Tensor
    temporal_embedding: torch.Tensor
    statistics_embedding: torch.Tensor


@dataclass(frozen=True)
class DateScoreOutput:
    scores: torch.Tensor
    local_scores: torch.Tensor
    contextual_scores: torch.Tensor
    contextual_embeddings: torch.Tensor
    market_embedding: torch.Tensor
    context_gate: torch.Tensor


@dataclass(frozen=True)
class FinancePatchTransformerOutput:
    date_scores: DateScoreOutput
    local: LocalEncoderOutput


class MaskedTemporalPooling(nn.Module):
    """Combine last, long-run mean and learned recency-aware patch pooling."""

    def __init__(
        self,
        *,
        num_channels: int,
        d_model: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.d_model = d_model
        self.queries = nn.Parameter(torch.empty(num_channels, d_model))
        self.recency_strength = nn.Parameter(torch.zeros(num_channels))
        nn.init.normal_(self.queries, mean=0.0, std=d_model**-0.5)
        self.channel_norm = nn.LayerNorm(3 * d_model)
        self.projection = nn.Sequential(
            nn.Linear(num_channels * 3 * d_model, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

    def forward(self, hidden: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4:
            raise ValueError("hidden must be [B,C,P,D]")
        if patch_mask.shape != hidden.shape[:3]:
            raise ValueError("patch_mask must match hidden [B,C,P]")
        if hidden.shape[1] != self.num_channels or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden channel or model dimensions differ from pooling config")

        valid = patch_mask.to(dtype=torch.bool)
        weights = valid.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)

        positions = torch.arange(hidden.shape[2], device=hidden.device)
        last_indices = torch.where(
            valid,
            positions.view(1, 1, -1),
            torch.full_like(positions.view(1, 1, -1), -1),
        ).amax(dim=2)
        safe_last = last_indices.clamp_min(0)
        last = hidden.gather(
            2,
            safe_last[..., None, None].expand(-1, -1, 1, hidden.shape[-1]),
        ).squeeze(2)
        channel_valid = last_indices >= 0
        last = last * channel_valid.unsqueeze(-1).to(last.dtype)

        logits = torch.einsum("bcpd,cd->bcp", hidden, self.queries) / math.sqrt(
            self.d_model
        )
        if hidden.shape[2] > 1:
            recency = torch.linspace(
                -1.0, 0.0, hidden.shape[2], device=hidden.device, dtype=hidden.dtype
            )
            logits = logits + (
                torch.nn.functional.softplus(self.recency_strength)[None, :, None]
                * recency[None, None, :]
            )
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        safe_logits = torch.where(
            channel_valid.unsqueeze(-1), logits, torch.zeros_like(logits)
        )
        attention_weights = torch.softmax(safe_logits.float(), dim=2).to(hidden.dtype)
        attention_weights = attention_weights * valid.to(hidden.dtype)
        attention_weights = attention_weights / attention_weights.sum(
            dim=2, keepdim=True
        ).clamp_min(1.0)
        attended = (hidden * attention_weights.unsqueeze(-1)).sum(dim=2)

        pooled = self.channel_norm(torch.cat([last, mean, attended], dim=-1))
        pooled = pooled * channel_valid.unsqueeze(-1).to(pooled.dtype)
        return self.projection(pooled.flatten(start_dim=1))


class MultiScaleStatisticsEncoder(nn.Module):
    """Encode deterministic masked statistics over fixed financial horizons."""

    def __init__(
        self,
        *,
        num_asset_channels: int,
        windows: tuple[int, ...],
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not windows or any(window < 1 for window in windows):
            raise ValueError("statistics windows must be positive")
        self.num_asset_channels = num_asset_channels
        self.windows = windows
        input_dim = num_asset_channels * (len(windows) * 5 + 1)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    @staticmethod
    def _window_statistics(
        values: torch.Tensor, observed: torch.Tensor
    ) -> list[torch.Tensor]:
        mask = observed.to(dtype=torch.bool)
        weights = mask.to(values.dtype)
        count = weights.sum(dim=1)
        mean = (values * weights).sum(dim=1) / count.clamp_min(1.0)
        centered = (values - mean.unsqueeze(1)) * weights
        variance = centered.square().sum(dim=1) / count.clamp_min(1.0)
        std = variance.clamp_min(0.0).sqrt()
        maximum = values.masked_fill(~mask, torch.finfo(values.dtype).min).amax(dim=1)
        minimum = values.masked_fill(~mask, torch.finfo(values.dtype).max).amin(dim=1)
        valid = count > 0
        maximum = torch.where(valid, maximum, torch.zeros_like(maximum))
        minimum = torch.where(valid, minimum, torch.zeros_like(minimum))
        observed_ratio = weights.mean(dim=1)
        return [mean, std, minimum, maximum, observed_ratio]

    def forward(self, values: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        if values.shape != observed_mask.shape or values.ndim != 3:
            raise ValueError("values and observed_mask must share [B,L,C] shape")
        if values.shape[2] < self.num_asset_channels:
            raise ValueError("values contain fewer channels than the statistics config")
        asset_values = values[:, :, : self.num_asset_channels]
        asset_observed = observed_mask[:, :, : self.num_asset_channels].bool()
        statistics: list[torch.Tensor] = []
        for window in self.windows:
            if window > values.shape[1]:
                raise ValueError(
                    f"statistics window {window} exceeds context {values.shape[1]}"
                )
            statistics.extend(
                self._window_statistics(
                    asset_values[:, -window:, :], asset_observed[:, -window:, :]
                )
            )
        positions = torch.arange(values.shape[1], device=values.device)
        last_indices = torch.where(
            asset_observed,
            positions.view(1, -1, 1),
            torch.full_like(positions.view(1, -1, 1), -1),
        ).amax(dim=1)
        latest = asset_values.gather(
            1,
            last_indices.clamp_min(0).unsqueeze(1),
        ).squeeze(1)
        latest = latest * (last_indices >= 0).to(latest.dtype)
        statistics.append(latest)
        return self.network(torch.cat(statistics, dim=1))


def _patchtst_backbone(
    *,
    context_length: int,
    num_channels: int,
    patch_length: int,
    patch_stride: int,
    d_model: int,
    num_attention_heads: int,
    num_hidden_layers: int,
    ffn_dim: int,
    dropout: float,
) -> PatchTSTModel:
    config = PatchTSTConfig(
        context_length=context_length,
        num_input_channels=num_channels,
        patch_length=patch_length,
        patch_stride=patch_stride,
        d_model=d_model,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        ffn_dim=ffn_dim,
        dropout=dropout,
        attention_dropout=dropout,
        positional_dropout=dropout,
        path_dropout=0.0,
        ff_dropout=dropout,
        norm_type="layernorm",
        pre_norm=True,
        scaling=None,
        do_mask_input=False,
        use_cls_token=False,
        share_embedding=False,
        channel_attention=True,
    )
    return PatchTSTModel(config)


class LocalTemporalEncoder(nn.Module):
    def __init__(
        self,
        *,
        context_length: int,
        num_channels: int,
        num_asset_channels: int,
        patch_length: int,
        patch_stride: int,
        d_model: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        ffn_dim: int,
        dropout: float,
        temporal_output_dim: int,
        statistics_output_dim: int,
        output_dim: int,
        statistics_windows: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.backbone = _patchtst_backbone(
            context_length=context_length,
            num_channels=num_channels,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.pooling = MaskedTemporalPooling(
            num_channels=num_channels,
            d_model=d_model,
            output_dim=temporal_output_dim,
            dropout=dropout,
        )
        self.statistics = MultiScaleStatisticsEncoder(
            num_asset_channels=num_asset_channels,
            windows=statistics_windows,
            output_dim=statistics_output_dim,
            dropout=dropout,
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(temporal_output_dim + statistics_output_dim),
            nn.Linear(temporal_output_dim + statistics_output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self, values: torch.Tensor, observed_mask: torch.Tensor
    ) -> LocalEncoderOutput:
        output = self.backbone(
            past_values=values,
            past_observed_mask=observed_mask,
            return_dict=True,
        )
        encoded = PatchTSTEncoderOutputAdapter.adapt(
            output,
            observed_mask,
            patch_length=self.patch_length,
            patch_stride=self.patch_stride,
            sequence_start=int(self.backbone.patchifier.sequence_start),
        )
        temporal = self.pooling(encoded.hidden, encoded.patch_mask)
        statistics = self.statistics(values, observed_mask)
        embedding = self.fusion(torch.cat([temporal, statistics], dim=1))
        return LocalEncoderOutput(
            embedding=embedding,
            temporal_embedding=temporal,
            statistics_embedding=statistics,
        )


class MarketContextEncoder(nn.Module):
    def __init__(
        self,
        *,
        context_length: int,
        num_channels: int,
        patch_length: int,
        patch_stride: int,
        d_model: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        ffn_dim: int,
        dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.backbone = _patchtst_backbone(
            context_length=context_length,
            num_channels=num_channels,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.pooling = MaskedTemporalPooling(
            num_channels=num_channels,
            d_model=d_model,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(self, values: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        output = self.backbone(
            past_values=values,
            past_observed_mask=observed_mask,
            return_dict=True,
        )
        encoded = PatchTSTEncoderOutputAdapter.adapt(
            output,
            observed_mask,
            patch_length=self.patch_length,
            patch_stride=self.patch_stride,
            sequence_start=int(self.backbone.patchifier.sequence_start),
        )
        return self.pooling(encoded.hidden, encoded.patch_mask)


class CrossSectionalEncoder(nn.Module):
    """Permutation-equivariant full-date stock set encoder."""

    def __init__(
        self,
        *,
        d_model: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        ffn_dim: int,
        dropout: float,
        num_horizons: int,
        initial_context_gate: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0 < initial_context_gate < 1:
            raise ValueError("initial_context_gate must be inside (0, 1)")
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_attention_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_hidden_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.local_head = nn.Linear(d_model, num_horizons)
        self.context_head = nn.Linear(d_model, num_horizons)
        initial_logit = math.log(initial_context_gate / (1.0 - initial_context_gate))
        self.context_gate_logits = nn.Parameter(
            torch.full((num_horizons,), initial_logit)
        )

    def forward(
        self,
        local_embeddings: torch.Tensor,
        market_embedding: torch.Tensor,
    ) -> DateScoreOutput:
        if local_embeddings.ndim != 2:
            raise ValueError("local_embeddings must be [N,D]")
        if market_embedding.ndim == 2 and market_embedding.shape[0] == 1:
            market_embedding = market_embedding.squeeze(0)
        if market_embedding.ndim != 1:
            raise ValueError("market_embedding must be [D] or [1,D]")
        if local_embeddings.shape[1] != market_embedding.shape[0]:
            raise ValueError("local and market embedding dimensions differ")
        tokens = torch.cat([market_embedding.unsqueeze(0), local_embeddings], dim=0)
        contextual = self.encoder(tokens.unsqueeze(0)).squeeze(0)
        contextual_stocks = contextual[1:]
        local_scores = self.local_head(local_embeddings)
        contextual_scores = self.context_head(contextual_stocks)
        gate = torch.sigmoid(self.context_gate_logits)
        scores = local_scores + contextual_scores * gate
        return DateScoreOutput(
            scores=scores,
            local_scores=local_scores,
            contextual_scores=contextual_scores,
            contextual_embeddings=contextual_stocks,
            market_embedding=contextual[0],
            context_gate=gate,
        )


class FinancePatchTransformer(nn.Module):
    """Hierarchical local-time and full-date cross-sectional factor model."""

    def __init__(
        self,
        *,
        context_length: int,
        num_local_channels: int,
        num_market_channels: int,
        horizons: tuple[int, ...],
        num_asset_channels: int = 7,
        patch_length: int = 16,
        patch_stride: int = 8,
        local_d_model: int = 64,
        local_num_attention_heads: int = 8,
        local_num_hidden_layers: int = 4,
        local_ffn_dim: int = 256,
        market_d_model: int = 32,
        market_num_attention_heads: int = 4,
        market_num_hidden_layers: int = 2,
        market_ffn_dim: int = 128,
        embedding_dim: int = 128,
        cross_num_attention_heads: int = 4,
        cross_num_hidden_layers: int = 2,
        cross_ffn_dim: int = 256,
        statistics_output_dim: int = 64,
        statistics_windows: tuple[int, ...] = (5, 20, 60, 120, 252),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if tuple(sorted(set(horizons))) != horizons:
            raise ValueError("horizons must be sorted and unique")
        self.horizons = horizons
        self.local_encoder = LocalTemporalEncoder(
            context_length=context_length,
            num_channels=num_local_channels,
            num_asset_channels=num_asset_channels,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=local_d_model,
            num_attention_heads=local_num_attention_heads,
            num_hidden_layers=local_num_hidden_layers,
            ffn_dim=local_ffn_dim,
            dropout=dropout,
            temporal_output_dim=embedding_dim,
            statistics_output_dim=statistics_output_dim,
            output_dim=embedding_dim,
            statistics_windows=statistics_windows,
        )
        self.market_encoder = MarketContextEncoder(
            context_length=context_length,
            num_channels=num_market_channels,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=market_d_model,
            num_attention_heads=market_num_attention_heads,
            num_hidden_layers=market_num_hidden_layers,
            ffn_dim=market_ffn_dim,
            dropout=dropout,
            output_dim=embedding_dim,
        )
        self.cross_sectional_encoder = CrossSectionalEncoder(
            d_model=embedding_dim,
            num_attention_heads=cross_num_attention_heads,
            num_hidden_layers=cross_num_hidden_layers,
            ffn_dim=cross_ffn_dim,
            dropout=dropout,
            num_horizons=len(horizons),
        )

    def encode_local(
        self, values: torch.Tensor, observed_mask: torch.Tensor
    ) -> LocalEncoderOutput:
        return self.local_encoder(values, observed_mask)

    def encode_market(
        self, values: torch.Tensor, observed_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.market_encoder(values, observed_mask)

    def score_date(
        self,
        local_embeddings: torch.Tensor,
        market_embedding: torch.Tensor,
    ) -> DateScoreOutput:
        return self.cross_sectional_encoder(local_embeddings, market_embedding)

    def forward(
        self,
        values: torch.Tensor,
        observed_mask: torch.Tensor,
        market_values: torch.Tensor,
        market_observed_mask: torch.Tensor,
    ) -> FinancePatchTransformerOutput:
        local = self.encode_local(values, observed_mask)
        market = self.encode_market(market_values, market_observed_mask)
        date_scores = self.score_date(local.embedding, market)
        return FinancePatchTransformerOutput(date_scores=date_scores, local=local)


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def build_finance_transformer_model(
    config: Any,
    *,
    context_length: int,
) -> FinancePatchTransformer:
    return build_finance_transformer_architecture(
        config.model,
        context_length=context_length,
        num_local_channels=len(config.channels),
        num_market_channels=len(config.market_channels),
        horizons=tuple(config.horizons),
    )


def build_finance_transformer_architecture(
    model: Any,
    *,
    context_length: int,
    num_local_channels: int,
    num_market_channels: int,
    horizons: tuple[int, ...],
) -> FinancePatchTransformer:
    """Construct the one architecture shared by scratch, pretrain and replay."""

    if model.patch_length > context_length:
        raise ValueError("patch_length cannot exceed snapshot context_length")
    if model.statistics_windows[-1] > context_length:
        raise ValueError("statistics windows cannot exceed snapshot context_length")
    return FinancePatchTransformer(
        context_length=context_length,
        num_local_channels=num_local_channels,
        num_market_channels=num_market_channels,
        num_asset_channels=7,
        horizons=horizons,
        patch_length=model.patch_length,
        patch_stride=model.patch_stride,
        local_d_model=model.local_d_model,
        local_num_attention_heads=model.local_num_attention_heads,
        local_num_hidden_layers=model.local_num_hidden_layers,
        local_ffn_dim=model.local_ffn_dim,
        market_d_model=model.market_d_model,
        market_num_attention_heads=model.market_num_attention_heads,
        market_num_hidden_layers=model.market_num_hidden_layers,
        market_ffn_dim=model.market_ffn_dim,
        embedding_dim=model.embedding_dim,
        cross_num_attention_heads=model.cross_num_attention_heads,
        cross_num_hidden_layers=model.cross_num_hidden_layers,
        cross_ffn_dim=model.cross_ffn_dim,
        statistics_output_dim=model.statistics_output_dim,
        statistics_windows=tuple(model.statistics_windows),
        dropout=model.dropout,
    )
