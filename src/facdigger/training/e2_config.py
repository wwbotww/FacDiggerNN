"""Configuration for E2 ETTh1-initialized PatchTST fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from facdigger.data.config import DEFAULT_CHANNELS, StrictModel
from facdigger.training.e1_config import RandomPatchTSTConfig


class E2SourceConfig(StrictModel):
    model_id: str = Field(default="ibm-research/patchtst-etth1-pretrain", min_length=1)
    revision: str = Field(default="1212736a0decf12b5cea5a605302421e110a3614", min_length=1)
    minimum_loaded_numel_ratio: float = Field(default=0.80, ge=0, le=1)
    allowlist: list[str] = Field(default_factory=list)
    local_files_only: bool = True


class E2FineTuneConfig(StrictModel):
    batch_size: int = Field(default=64, ge=1)
    max_epochs: int = Field(default=15, ge=2)
    patience: int = Field(default=5, ge=1)
    minimum_epochs: int = Field(default=4, ge=1)
    head_only_epochs: int = Field(default=3, ge=1)
    unfreeze_last_n_blocks: int = Field(default=1, ge=1)
    head_learning_rate: float = Field(default=1e-3, gt=0)
    encoder_learning_rate: float = Field(default=1e-5, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["fp32", "fp16"] = "fp16"
    num_workers: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_stages(self) -> E2FineTuneConfig:
        if self.head_only_epochs >= self.max_epochs:
            raise ValueError("head_only_epochs must leave at least one FT-1 epoch")
        if self.minimum_epochs <= self.head_only_epochs:
            raise ValueError("minimum_epochs must include at least one FT-1 epoch")
        if self.minimum_epochs > self.max_epochs:
            raise ValueError("minimum_epochs cannot exceed max_epochs")
        return self


class E2ExperimentConfig(StrictModel):
    experiment_id: str = "e2_etth1_patchtst"
    seed: int = Field(default=42, ge=0)
    output_root: Path = Path("artifacts/e2")
    evaluation_split: Literal["valid", "test"] = "valid"
    unlock_test: bool = False
    minimum_coverage: float = Field(default=0.98, gt=0, le=1)
    channels: list[str] = Field(default_factory=lambda: list(DEFAULT_CHANNELS))
    costs_bps: list[float] = Field(default_factory=lambda: [0.0, 10.0, 20.0, 50.0])
    model: RandomPatchTSTConfig = Field(default_factory=RandomPatchTSTConfig)
    source: E2SourceConfig = Field(default_factory=E2SourceConfig)
    training: E2FineTuneConfig = Field(default_factory=E2FineTuneConfig)

    @model_validator(mode="after")
    def validate_protocol(self) -> E2ExperimentConfig:
        if self.evaluation_split == "test" and not self.unlock_test:
            raise ValueError("evaluation_split=test requires unlock_test=true")
        if self.training.unfreeze_last_n_blocks > self.model.num_hidden_layers:
            raise ValueError("unfreeze_last_n_blocks exceeds encoder depth")
        if any(cost < 0 for cost in self.costs_bps):
            raise ValueError("costs_bps cannot be negative")
        return self


def load_e2_config(path: str | Path) -> E2ExperimentConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"E2 configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return E2ExperimentConfig.model_validate(raw)
