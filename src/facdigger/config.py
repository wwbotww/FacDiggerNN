"""Validated project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class MarketConfig(StrictModel):
    name: Literal["us_equities_daily"] = "us_equities_daily"
    timezone: str = "America/New_York"
    exchanges: list[str] = Field(default_factory=lambda: ["XNYS", "XNAS", "XASE"])
    security_types: list[str] = Field(default_factory=lambda: ["common_stock"])
    minimum_close_usd: float = Field(default=5.0, ge=0)
    minimum_adv20_usd: float = Field(default=1_000_000.0, ge=0)


class SourceCheckpointConfig(StrictModel):
    model_id: str = "ibm-research/patchtst-etth1-pretrain"
    revision: str = "1212736a0decf12b5cea5a605302421e110a3614"
    minimum_loaded_numel_ratio: float = Field(default=0.80, ge=0, le=1)

    @field_validator("model_id", "revision")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class ModelConfig(StrictModel):
    context_length: int = Field(default=512, gt=0)
    input_channels: int = Field(default=7, gt=0)
    patch_length: int = Field(default=12, gt=0)
    patch_stride: int = Field(default=12, gt=0)
    source: SourceCheckpointConfig = Field(default_factory=SourceCheckpointConfig)

    @field_validator("patch_length")
    @classmethod
    def patch_must_fit_context(cls, value: int, info) -> int:
        context_length = info.data.get("context_length")
        if context_length is not None and value > context_length:
            raise ValueError("patch_length must not exceed context_length")
        return value


class RuntimeConfig(StrictModel):
    preferred_device: Literal["auto", "cpu", "cuda"] = "auto"
    cuda_precision: Literal["fp16"] = "fp16"
    output_root: Path = Path("artifacts")


class ProjectConfig(StrictModel):
    project_name: str = "facdigger"
    seed: int = Field(default=42, ge=0)
    market: MarketConfig = Field(default_factory=MarketConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


def load_project_config(path: str | Path) -> ProjectConfig:
    """Load a YAML configuration and reject unknown fields."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return ProjectConfig.model_validate(raw)
