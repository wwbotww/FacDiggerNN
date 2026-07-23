"""Configuration contracts for M6 walk-forward research."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field, model_validator

from facdigger.data.config import SplitConfig, StrictModel


class ResearchFoldConfig(SplitConfig):
    fold_id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")


class ResearchModelConfigs(StrictModel):
    e0: Path
    e1: Path
    e2: Path
    e3: Path


class ResearchDecisionConfig(StrictModel):
    minimum_positive_cell_ratio: float = Field(default=0.5, ge=0, le=1)
    cost_bps: float = Field(default=20.0, ge=0)
    require_neutralized_positive: bool = True
    require_source_research_ready: bool = True


class M6ResearchConfig(StrictModel):
    research_id: str = Field(default="m6_e0_e3_research", min_length=1)
    base_dataset_config: Path
    output_root: Path = Path("artifacts/research")
    snapshot_output_root: Path = Path("data/walk_forward_snapshots")
    seeds: list[int] = Field(default_factory=lambda: [17, 42, 73], min_length=3)
    folds: list[ResearchFoldConfig] = Field(min_length=3)
    models: ResearchModelConfigs
    hac_lags: int = Field(default=5, ge=0)
    non_overlapping_stride: int = Field(default=5, ge=1)
    non_overlapping_offset: int = Field(default=0, ge=0)
    decisions: ResearchDecisionConfig = Field(default_factory=ResearchDecisionConfig)

    @model_validator(mode="after")
    def validate_research_protocol(self) -> M6ResearchConfig:
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("research seeds must be unique")
        if len({fold.fold_id for fold in self.folds}) != len(self.folds):
            raise ValueError("research fold_id values must be unique")
        if self.non_overlapping_offset >= self.non_overlapping_stride:
            raise ValueError("non_overlapping_offset must be smaller than stride")
        for previous, current in zip(self.folds, self.folds[1:], strict=False):
            if not previous.train_end < current.train_end:
                raise ValueError("fold train_end values must strictly expand")
            if not previous.valid_end < current.valid_end:
                raise ValueError("fold valid_end values must strictly increase")
            if not previous.test_end < current.test_end:
                raise ValueError("fold test_end values must strictly increase")
            if previous.valid_end >= current.valid_end:
                raise ValueError("walk-forward validation boundaries must be chronological")
            if current.train_end < previous.valid_end:
                raise ValueError(
                    "each expanding train_end must include the previous validation period"
                )
        return self


def load_m6_config(path: str | Path) -> M6ResearchConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"M6 research configuration not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return M6ResearchConfig.model_validate(raw)
