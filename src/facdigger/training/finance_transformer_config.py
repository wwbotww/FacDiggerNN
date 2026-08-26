"""Configuration for finance-native scratch and pretrained Transformer runs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from facdigger.data.config import (
    FINANCE_TRANSFORMER_CHANNELS,
    MARKET_CONTEXT_CHANNELS,
    StrictModel,
)
from facdigger.training.ranking import CrossSectionalRankingConfig


class FinanceTransformerModelConfig(StrictModel):
    patch_length: int = Field(default=16, ge=1)
    patch_stride: int = Field(default=8, ge=1)
    local_d_model: int = Field(default=64, ge=1)
    local_num_attention_heads: int = Field(default=8, ge=1)
    local_num_hidden_layers: int = Field(default=4, ge=1)
    local_ffn_dim: int = Field(default=256, ge=1)
    market_d_model: int = Field(default=32, ge=1)
    market_num_attention_heads: int = Field(default=4, ge=1)
    market_num_hidden_layers: int = Field(default=2, ge=1)
    market_ffn_dim: int = Field(default=128, ge=1)
    embedding_dim: int = Field(default=128, ge=1)
    cross_num_attention_heads: int = Field(default=4, ge=1)
    cross_num_hidden_layers: int = Field(default=2, ge=1)
    cross_ffn_dim: int = Field(default=256, ge=1)
    statistics_output_dim: int = Field(default=64, ge=1)
    statistics_windows: list[int] = Field(
        default_factory=lambda: [5, 20, 60, 120, 252], min_length=1
    )
    dropout: float = Field(default=0.1, ge=0, lt=1)

    @model_validator(mode="after")
    def validate_dimensions(self) -> FinanceTransformerModelConfig:
        dimensions = (
            (self.local_d_model, self.local_num_attention_heads, "local"),
            (self.market_d_model, self.market_num_attention_heads, "market"),
            (self.embedding_dim, self.cross_num_attention_heads, "cross"),
        )
        for dimension, heads, name in dimensions:
            if dimension % heads:
                raise ValueError(f"{name} d_model must be divisible by attention heads")
        if self.statistics_windows != sorted(set(self.statistics_windows)):
            raise ValueError("statistics_windows must be sorted and unique")
        return self


class MultiHorizonObjectiveConfig(CrossSectionalRankingConfig):
    horizon_weights: dict[int, float] = Field(
        default_factory=lambda: {1: 0.2, 5: 0.6, 20: 0.2}
    )
    scale_regularization: float = Field(default=0.01, ge=0)
    selection_stability_penalty: float = Field(default=0.25, ge=0)
    selection_subperiods: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def validate_weights(self) -> MultiHorizonObjectiveConfig:
        if any(weight <= 0 for weight in self.horizon_weights.values()):
            raise ValueError("horizon weights must be positive")
        total = sum(self.horizon_weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError("horizon weights must sum to one")
        return self


class FinanceTransformerTrainingConfig(StrictModel):
    batch_size: int = Field(default=16, ge=1)
    max_epochs: int = Field(default=10, ge=1)
    minimum_epochs: int = Field(default=6, ge=1)
    patience: int = Field(default=3, ge=1)
    encoder_learning_rate: float = Field(default=1e-4, gt=0)
    head_learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=1e-2, ge=0)
    adam_beta1: float = Field(default=0.9, gt=0, lt=1)
    adam_beta2: float = Field(default=0.95, gt=0, lt=1)
    warmup_fraction: float = Field(default=0.05, ge=0, lt=1)
    minimum_learning_rate_ratio: float = Field(default=0.1, gt=0, le=1)
    dates_per_optimizer_step: int = Field(default=4, ge=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["fp32", "fp16"] = "fp16"
    num_workers: int = Field(default=0, ge=0)
    replay_relative_tolerance: float = Field(default=1e-4, ge=0)
    replay_absolute_tolerance: float = Field(default=1e-5, ge=0)
    objective: MultiHorizonObjectiveConfig = Field(
        default_factory=MultiHorizonObjectiveConfig
    )

    @model_validator(mode="after")
    def validate_epochs(self) -> FinanceTransformerTrainingConfig:
        if self.minimum_epochs > self.max_epochs:
            raise ValueError("minimum_epochs cannot exceed max_epochs")
        return self


class FinanceTransformerExperimentConfig(StrictModel):
    experiment_id: str = "finance_patch_transformer_scratch"
    initialization: Literal["scratch", "finance_pretrained"] = "scratch"
    pretrained_checkpoint: Path | None = None
    seed: int = Field(default=42, ge=0)
    output_root: Path = Path("artifacts/finance_transformer")
    evaluation_split: Literal["valid", "test"] = "valid"
    unlock_test: bool = False
    minimum_coverage: float = Field(default=0.98, gt=0, le=1)
    selection_fraction: float = Field(default=0.15, gt=0, lt=0.5)
    channels: list[str] = Field(
        default_factory=lambda: list(FINANCE_TRANSFORMER_CHANNELS)
    )
    market_channels: list[str] = Field(
        default_factory=lambda: list(MARKET_CONTEXT_CHANNELS)
    )
    horizons: list[int] = Field(default_factory=lambda: [1, 5, 20])
    primary_horizon: int = 5
    costs_bps: list[float] = Field(default_factory=lambda: [0.0, 10.0, 20.0, 50.0])
    model: FinanceTransformerModelConfig = Field(
        default_factory=FinanceTransformerModelConfig
    )
    training: FinanceTransformerTrainingConfig = Field(
        default_factory=FinanceTransformerTrainingConfig
    )

    @model_validator(mode="after")
    def validate_protocol(self) -> FinanceTransformerExperimentConfig:
        if self.evaluation_split == "test" and not self.unlock_test:
            raise ValueError("evaluation_split=test requires unlock_test=true")
        if self.channels != FINANCE_TRANSFORMER_CHANNELS:
            raise ValueError(
                f"finance transformer requires ordered channels {FINANCE_TRANSFORMER_CHANNELS}"
            )
        if self.market_channels != MARKET_CONTEXT_CHANNELS:
            raise ValueError(
                "finance transformer requires ordered market channels "
                f"{MARKET_CONTEXT_CHANNELS}"
            )
        if self.horizons != sorted(set(self.horizons)):
            raise ValueError("horizons must be sorted and unique")
        if self.primary_horizon not in self.horizons:
            raise ValueError("primary_horizon must be included in horizons")
        if set(self.training.objective.horizon_weights) != set(self.horizons):
            raise ValueError("objective horizon weights must match experiment horizons")
        if self.initialization == "scratch" and self.pretrained_checkpoint is not None:
            raise ValueError("scratch initialization cannot specify pretrained_checkpoint")
        if self.initialization == "finance_pretrained" and self.pretrained_checkpoint is None:
            raise ValueError("finance_pretrained initialization requires a checkpoint")
        if self.model.statistics_windows[-1] > 512:
            raise ValueError("statistics windows cannot exceed the registered context")
        if any(cost < 0 for cost in self.costs_bps):
            raise ValueError("costs_bps cannot be negative")
        return self


def load_finance_transformer_config(
    path: str | Path,
) -> FinanceTransformerExperimentConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Finance Transformer configuration file not found: {config_path}"
        )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return FinanceTransformerExperimentConfig.model_validate(raw)
