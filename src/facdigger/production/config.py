"""Strict configuration for the daily FacDigger production service."""

from __future__ import annotations

import re
from datetime import time
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator, model_validator

from facdigger.data.config import StrictModel

_RELEASE_ID = re.compile(r"^[0-9a-f]{64}$")


class ProductionScheduleConfig(StrictModel):
    timezone: Literal["America/New_York"] = "America/New_York"
    first_attempt: time = time(hour=19)
    retry_minutes: int = Field(default=30, ge=1, le=240)
    cutoff: Literal["next_regular_session_open"] = "next_regular_session_open"


class ProductionDataConfig(StrictModel):
    provider_config: Path
    store_root: Path = Path("data/production/eodhd")
    revision_sessions: int = Field(default=10, ge=2, le=60)
    feature_buffer_sessions: int = Field(default=20, ge=20, le=120)
    bootstrap_source: Path = Path("data/bronze/eodhd_us_historical_liquid")


class ProductionModelConfig(StrictModel):
    release_root: Path = Path("artifacts/releases")
    release_id: str
    device: Literal["cpu", "cuda", "auto"] = "cpu"

    @field_validator("release_id")
    @classmethod
    def validate_release_id(cls, value: str) -> str:
        if _RELEASE_ID.fullmatch(value) is None:
            raise ValueError("release_id must be a 64-character lowercase SHA-256")
        if value == "0" * 64:
            raise ValueError("release_id placeholder must be replaced before production")
        return value


class ProductionInferenceConfig(StrictModel):
    output_root: Path = Path("data/inference_snapshots/production")
    retention_sessions: int = Field(default=10, ge=1, le=120)


class ProductionFactorBatchConfig(StrictModel):
    output_root: Path = Path("artifacts/factor_batches")
    retention: Literal["forever"] = "forever"
    minimum_candidate_rows: int = Field(default=100, ge=1)
    minimum_eligible_rows: int = Field(default=100, ge=1)


class ProductionServiceConfig(StrictModel):
    contract: Literal["facdigger.daily_production"] = "facdigger.daily_production"
    schedule: ProductionScheduleConfig = Field(default_factory=ProductionScheduleConfig)
    data: ProductionDataConfig
    model: ProductionModelConfig
    inference: ProductionInferenceConfig = Field(default_factory=ProductionInferenceConfig)
    factor_batch: ProductionFactorBatchConfig = Field(
        default_factory=ProductionFactorBatchConfig
    )
    state_database: Path = Path("data/production/production.sqlite3")
    poll_seconds: int = Field(default=60, ge=5, le=1800)

    @model_validator(mode="after")
    def validate_production_protocol(self) -> ProductionServiceConfig:
        if self.schedule.first_attempt != time(hour=19):
            raise ValueError("daily production first_attempt is fixed at 19:00 America/New_York")
        if self.schedule.retry_minutes != 30:
            raise ValueError("daily production retry_minutes is fixed at 30")
        if self.data.revision_sessions != 10:
            raise ValueError("daily production revision_sessions is fixed at 10")
        if self.inference.retention_sessions != 10:
            raise ValueError("daily production inference retention is fixed at 10 sessions")
        training_roots = (
            Path("data/snapshots").resolve(),
            Path("data/walk_forward_snapshots").resolve(),
        )
        generated_roots = {
            "production store_root": self.data.store_root.resolve(),
            "production inference output_root": self.inference.output_root.resolve(),
            "production state_database": self.state_database.resolve(),
        }
        for label, generated in generated_roots.items():
            for training_root in training_roots:
                try:
                    generated.relative_to(training_root)
                except ValueError:
                    continue
                raise ValueError(f"{label} must not be inside {training_root}")
        if self.factor_batch.minimum_eligible_rows > self.factor_batch.minimum_candidate_rows:
            raise ValueError("minimum_eligible_rows cannot exceed minimum_candidate_rows")
        return self


def load_production_config(path: str | Path) -> ProductionServiceConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Production configuration not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return ProductionServiceConfig.model_validate(raw)
