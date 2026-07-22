"""Configuration for the standard-Parquet dataset pipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CHANNELS = [
    "r_close",
    "r_gap",
    "r_intraday",
    "range",
    "dlog_volume",
    "vol20",
    "dollar_volume_z20",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ParquetSourceConfig(StrictModel):
    bars: Path
    universe: Path
    corporate_actions: Path | None = None
    delistings: Path | None = None


class FeatureSetConfig(StrictModel):
    name: str = "price_volume_v1"
    context_length: int = Field(default=512, gt=0)
    channels: list[str] = Field(default_factory=lambda: list(DEFAULT_CHANNELS))
    scaler: Literal["train_global_robust"] = "train_global_robust"
    winsor_lower: float = Field(default=0.005, ge=0, lt=0.5)
    winsor_upper: float = Field(default=0.995, gt=0.5, le=1)

    @model_validator(mode="after")
    def validate_channels(self) -> FeatureSetConfig:
        if self.channels != DEFAULT_CHANNELS:
            raise ValueError(f"M1 requires the ordered channels {DEFAULT_CHANNELS}")
        if self.winsor_lower >= self.winsor_upper:
            raise ValueError("winsor_lower must be smaller than winsor_upper")
        return self


class LabelConfig(StrictModel):
    name: str = "next_open_to_fifth_close_excess_return"
    execution_lag: int = Field(default=1, ge=1)
    horizon: int = Field(default=5, ge=1)
    benchmark: Literal["eligible_equal_weight"] = "eligible_equal_weight"


class SplitConfig(StrictModel):
    train_end: date
    valid_end: date
    test_end: date
    embargo_sessions: int = Field(default=5, ge=0)

    @model_validator(mode="after")
    def dates_are_chronological(self) -> SplitConfig:
        if not self.train_end < self.valid_end < self.test_end:
            raise ValueError("split dates must satisfy train_end < valid_end < test_end")
        return self


class DatasetBuildConfig(StrictModel):
    dataset_name: str = "us_equities_daily_v1"
    sources: ParquetSourceConfig
    output_root: Path = Path("data/snapshots")
    features: FeatureSetConfig = Field(default_factory=FeatureSetConfig)
    label: LabelConfig = Field(default_factory=LabelConfig)
    split: SplitConfig


def load_dataset_build_config(path: str | Path) -> DatasetBuildConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return DatasetBuildConfig.model_validate(raw)
