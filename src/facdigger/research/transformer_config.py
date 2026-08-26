"""Frozen nine-stage protocol for the streamlined Transformer comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from facdigger.data.config import StrictModel
from facdigger.research.config import ResearchFoldConfig


class TransformerExperimentPaths(StrictModel):
    pretraining: Path
    scratch: Path
    pretrained: Path


class TransformerComparisonDecision(StrictModel):
    minimum_paired_mean_rank_ic_delta: float = 0.001
    minimum_positive_folds: Literal[2] = 2
    require_positive_worst_fold: Literal[True] = True
    hac_lags: int = Field(default=5, ge=0)
    non_overlapping_stride: int = Field(default=5, ge=1)


class TransformerComparisonConfig(StrictModel):
    research_id: str = "finance_transformer_scratch_vs_pretrained"
    base_dataset_config: Path
    output_root: Path = Path("artifacts/transformer_comparison")
    snapshot_output_root: Path = Path("data/transformer_walk_forward_snapshots")
    admission_report: Path = Path(
        "artifacts/benchmarks/finance-transformer-rtx2070s.json"
    )
    seed: Literal[42] = 42
    folds: list[ResearchFoldConfig] = Field(min_length=3, max_length=3)
    experiments: TransformerExperimentPaths
    decisions: TransformerComparisonDecision = Field(
        default_factory=TransformerComparisonDecision
    )

    @model_validator(mode="after")
    def validate_protocol(self) -> TransformerComparisonConfig:
        if len({fold.fold_id for fold in self.folds}) != 3:
            raise ValueError("streamlined Transformer folds must have unique IDs")
        for previous, current in zip(self.folds, self.folds[1:], strict=False):
            if current.train_end <= previous.train_end:
                raise ValueError("Transformer Train boundaries must strictly expand")
            if current.valid_end <= previous.valid_end:
                raise ValueError("Transformer validation boundaries must strictly increase")
            if current.train_end < previous.valid_end:
                raise ValueError("each Train must include the previous validation period")
        return self


def load_transformer_comparison_config(
    path: str | Path,
) -> TransformerComparisonConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Transformer comparison configuration not found: {config_path}"
        )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return TransformerComparisonConfig.model_validate(raw)


def validate_transformer_experiment_paths(
    config: TransformerComparisonConfig,
) -> dict[str, Path]:
    paths = {
        key: Path(value).resolve()
        for key, value in config.experiments.model_dump().items()
    }
    missing = [f"{key}:{path}" for key, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Transformer experiment configuration missing: " + ", ".join(missing)
        )
    return paths
