"""Configuration for E0 tabular baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from facdigger.data.config import DEFAULT_CHANNELS, StrictModel


class MLPBaselineConfig(StrictModel):
    hidden_dims: list[int] = Field(default_factory=lambda: [128, 64], min_length=1)
    dropout: float = Field(default=0.2, ge=0, lt=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    batch_size: int = Field(default=256, ge=1)
    max_epochs: int = Field(default=100, ge=1)
    patience: int = Field(default=15, ge=1)
    device: Literal["auto", "cpu", "cuda"] = "auto"


class LightGBMBaselineConfig(StrictModel):
    n_estimators: int = Field(default=500, ge=1)
    learning_rate: float = Field(default=0.03, gt=0)
    num_leaves: int = Field(default=31, ge=2)
    min_child_samples: int = Field(default=20, ge=1)
    reg_lambda: float = Field(default=1.0, ge=0)
    early_stopping_rounds: int = Field(default=30, ge=1)


class E0ExperimentConfig(StrictModel):
    experiment_id: str = "e0b_mlp"
    model_type: Literal["mlp", "lightgbm"] = "mlp"
    seed: int = Field(default=42, ge=0)
    output_root: Path = Path("artifacts/e0")
    evaluation_split: Literal["valid", "test"] = "valid"
    unlock_test: bool = False
    minimum_coverage: float = Field(default=0.98, gt=0, le=1)
    selection_fraction: float = Field(default=0.1, gt=0, lt=0.5)
    channels: list[str] = Field(default_factory=lambda: list(DEFAULT_CHANNELS))
    windows: list[int] = Field(default_factory=lambda: [5, 20, 60, 120, 252])
    costs_bps: list[float] = Field(default_factory=lambda: [0.0, 10.0, 20.0, 50.0])
    mlp: MLPBaselineConfig = Field(default_factory=MLPBaselineConfig)
    lightgbm: LightGBMBaselineConfig = Field(default_factory=LightGBMBaselineConfig)

    @model_validator(mode="after")
    def validate_protocol(self) -> E0ExperimentConfig:
        if self.evaluation_split == "test" and not self.unlock_test:
            raise ValueError("evaluation_split=test requires unlock_test=true")
        if not self.windows or any(window < 1 for window in self.windows):
            raise ValueError("windows must contain positive values")
        if sorted(set(self.windows)) != self.windows:
            raise ValueError("windows must be sorted and unique")
        if any(cost < 0 for cost in self.costs_bps):
            raise ValueError("costs_bps cannot be negative")
        return self


def load_e0_config(path: str | Path) -> E0ExperimentConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"E0 configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return E0ExperimentConfig.model_validate(raw)
