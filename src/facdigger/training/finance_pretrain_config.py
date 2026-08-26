"""Configuration for fold-local finance-native Transformer pretraining."""

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
from facdigger.training.finance_transformer_config import (
    FinanceTransformerModelConfig,
)


class FinancePretrainingObjectiveConfig(StrictModel):
    mask_ratio: float = Field(default=0.3, gt=0, lt=1)
    minimum_span_patches: int = Field(default=1, ge=1)
    maximum_span_patches: int = Field(default=4, ge=1)
    shared_channel_fraction: float = Field(default=0.5, ge=0, le=1)
    reconstruction_weight: float = Field(default=0.4, ge=0)
    future_summary_weight: float = Field(default=0.6, ge=0)
    huber_delta: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def validate_objective(self) -> FinancePretrainingObjectiveConfig:
        if self.maximum_span_patches < self.minimum_span_patches:
            raise ValueError("maximum_span_patches cannot be below minimum_span_patches")
        if abs(self.reconstruction_weight + self.future_summary_weight - 1.0) > 1e-9:
            raise ValueError("pretraining objective weights must sum to one")
        return self


class FinanceLinearProbeConfig(StrictModel):
    fit_dates: int = Field(default=60, ge=2)
    selection_dates: int = Field(default=20, ge=2)
    epochs: int = Field(default=20, ge=1)
    learning_rate: float = Field(default=1e-2, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    minimum_cross_section_size: int = Field(default=32, ge=2)


class FinancePretrainingRunConfig(StrictModel):
    batch_size: int = Field(default=32, ge=1)
    max_epochs: int = Field(default=3, ge=1)
    minimum_epochs: int = Field(default=2, ge=1)
    patience: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-2, ge=0)
    adam_beta1: float = Field(default=0.9, gt=0, lt=1)
    adam_beta2: float = Field(default=0.95, gt=0, lt=1)
    warmup_fraction: float = Field(default=0.05, ge=0, lt=1)
    minimum_learning_rate_ratio: float = Field(default=0.1, gt=0, le=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["fp32", "fp16"] = "fp16"
    num_workers: int = Field(default=0, ge=0)
    objective: FinancePretrainingObjectiveConfig = Field(
        default_factory=FinancePretrainingObjectiveConfig
    )
    probe: FinanceLinearProbeConfig = Field(default_factory=FinanceLinearProbeConfig)

    @model_validator(mode="after")
    def validate_epochs(self) -> FinancePretrainingRunConfig:
        if self.minimum_epochs > self.max_epochs:
            raise ValueError("minimum_epochs cannot exceed max_epochs")
        return self


class FinancePretrainingExperimentConfig(StrictModel):
    experiment_id: str = "finance_patch_pretrain"
    seed: int = Field(default=42, ge=0)
    output_root: Path = Path("artifacts/finance_pretrain")
    future_horizon: Literal[5] = 5
    channels: list[str] = Field(
        default_factory=lambda: list(FINANCE_TRANSFORMER_CHANNELS)
    )
    market_channels: list[str] = Field(
        default_factory=lambda: list(MARKET_CONTEXT_CHANNELS)
    )
    model: FinanceTransformerModelConfig = Field(
        default_factory=FinanceTransformerModelConfig
    )
    training: FinancePretrainingRunConfig = Field(
        default_factory=FinancePretrainingRunConfig
    )

    @model_validator(mode="after")
    def validate_protocol(self) -> FinancePretrainingExperimentConfig:
        if self.channels != FINANCE_TRANSFORMER_CHANNELS:
            raise ValueError(
                f"finance pretraining requires ordered channels {FINANCE_TRANSFORMER_CHANNELS}"
            )
        if self.market_channels != MARKET_CONTEXT_CHANNELS:
            raise ValueError(
                "finance pretraining requires ordered market channels "
                f"{MARKET_CONTEXT_CHANNELS}"
            )
        if not self.model.statistics_windows:
            raise ValueError("statistics_windows cannot be empty")
        return self


def load_finance_pretraining_config(
    path: str | Path,
) -> FinancePretrainingExperimentConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Finance pretraining configuration file not found: {config_path}"
        )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return FinancePretrainingExperimentConfig.model_validate(raw)
