"""Configuration for E1 randomly initialized PatchTST training."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from facdigger.data.config import DEFAULT_CHANNELS, StrictModel


class RandomPatchTSTConfig(StrictModel):
    patch_length: int = Field(default=12, ge=1)
    patch_stride: int = Field(default=12, ge=1)
    d_model: int = Field(default=128, ge=1)
    num_attention_heads: int = Field(default=16, ge=1)
    num_hidden_layers: int = Field(default=6, ge=1)
    ffn_dim: int = Field(default=512, ge=1)
    dropout: float = Field(default=0.3, ge=0, lt=1)
    attention_dropout: float = Field(default=0.0, ge=0, lt=1)
    positional_dropout: float = Field(default=0.0, ge=0, lt=1)
    path_dropout: float = Field(default=0.0, ge=0, lt=1)
    ff_dropout: float = Field(default=0.0, ge=0, lt=1)
    norm_type: Literal["batchnorm", "layernorm"] = "batchnorm"
    pre_norm: bool = False
    scaling: Literal["mean", "std"] | None = "mean"
    alpha_hidden_dim: int = Field(default=128, ge=1)
    alpha_dropout: float = Field(default=0.2, ge=0, lt=1)

    @model_validator(mode="after")
    def validate_dimensions(self) -> RandomPatchTSTConfig:
        if self.d_model % self.num_attention_heads:
            raise ValueError("d_model must be divisible by num_attention_heads")
        return self


class E1TrainingConfig(StrictModel):
    batch_size: int = Field(default=64, ge=1)
    max_epochs: int = Field(default=30, ge=1)
    patience: int = Field(default=7, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["fp32", "fp16"] = "fp16"
    num_workers: int = Field(default=0, ge=0)
    minimum_epochs: int = Field(default=1, ge=1)


class E1ExperimentConfig(StrictModel):
    experiment_id: str = "e1_random_patchtst"
    seed: int = Field(default=42, ge=0)
    output_root: Path = Path("artifacts/e1")
    evaluation_split: Literal["valid", "test"] = "valid"
    unlock_test: bool = False
    minimum_coverage: float = Field(default=0.98, gt=0, le=1)
    channels: list[str] = Field(default_factory=lambda: list(DEFAULT_CHANNELS))
    costs_bps: list[float] = Field(default_factory=lambda: [0.0, 10.0, 20.0, 50.0])
    model: RandomPatchTSTConfig = Field(default_factory=RandomPatchTSTConfig)
    training: E1TrainingConfig = Field(default_factory=E1TrainingConfig)

    @model_validator(mode="after")
    def validate_protocol(self) -> E1ExperimentConfig:
        if self.evaluation_split == "test" and not self.unlock_test:
            raise ValueError("evaluation_split=test requires unlock_test=true")
        if self.training.minimum_epochs > self.training.max_epochs:
            raise ValueError("minimum_epochs cannot exceed max_epochs")
        if any(cost < 0 for cost in self.costs_bps):
            raise ValueError("costs_bps cannot be negative")
        return self


def load_e1_config(path: str | Path) -> E1ExperimentConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"E1 configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return E1ExperimentConfig.model_validate(raw)
