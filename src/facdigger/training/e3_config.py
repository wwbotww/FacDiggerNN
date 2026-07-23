"""Configuration for E3 train-only financial masked pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from facdigger.data.config import DEFAULT_CHANNELS, StrictModel
from facdigger.training.e1_config import RandomPatchTSTConfig
from facdigger.training.e2_config import E2FineTuneConfig, E2SourceConfig


class E3PretrainingConfig(StrictModel):
    batch_size: int = Field(default=64, ge=1)
    max_epochs: int = Field(default=20, ge=1)
    patience: int = Field(default=5, ge=1)
    minimum_epochs: int = Field(default=3, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    mask_ratio: float = Field(default=0.4, gt=0, lt=1)
    loss: Literal["mse", "huber"] = "huber"
    huber_delta: float = Field(default=1.0, gt=0)
    validation_fraction: float = Field(default=0.1, gt=0, lt=0.5)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["fp32", "fp16"] = "fp16"
    num_workers: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_epochs(self) -> E3PretrainingConfig:
        if self.minimum_epochs > self.max_epochs:
            raise ValueError("minimum_epochs cannot exceed max_epochs")
        return self


class E3ExperimentConfig(StrictModel):
    experiment_id: str = "e3_financial_pretrained_patchtst"
    seed: int = Field(default=42, ge=0)
    output_root: Path = Path("artifacts/e3")
    evaluation_split: Literal["valid", "test"] = "valid"
    unlock_test: bool = False
    minimum_coverage: float = Field(default=0.98, gt=0, le=1)
    selection_fraction: float = Field(default=0.1, gt=0, lt=0.5)
    channels: list[str] = Field(default_factory=lambda: list(DEFAULT_CHANNELS))
    costs_bps: list[float] = Field(default_factory=lambda: [0.0, 10.0, 20.0, 50.0])
    model: RandomPatchTSTConfig = Field(default_factory=RandomPatchTSTConfig)
    source: E2SourceConfig = Field(default_factory=E2SourceConfig)
    pretraining: E3PretrainingConfig = Field(default_factory=E3PretrainingConfig)
    finetuning: E2FineTuneConfig = Field(default_factory=E2FineTuneConfig)

    @model_validator(mode="after")
    def validate_protocol(self) -> E3ExperimentConfig:
        if self.evaluation_split == "test" and not self.unlock_test:
            raise ValueError("evaluation_split=test requires unlock_test=true")
        if self.finetuning.unfreeze_last_n_blocks > self.model.num_hidden_layers:
            raise ValueError("unfreeze_last_n_blocks exceeds encoder depth")
        if any(cost < 0 for cost in self.costs_bps):
            raise ValueError("costs_bps cannot be negative")
        return self


def load_e3_config(path: str | Path) -> E3ExperimentConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"E3 configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return E3ExperimentConfig.model_validate(raw)
