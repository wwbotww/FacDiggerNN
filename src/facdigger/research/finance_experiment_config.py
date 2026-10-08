"""Frozen multi-fold/seed Finance research cells, separate from the nine-stage protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from facdigger.data.config import StrictModel
from facdigger.research.config import ResearchFoldConfig
from facdigger.training.finance_transformer_config import FinanceTransformerExperimentConfig

Candidate = Literal[
    "statistics_linear", "statistics_mlp", "statistics_mlp_dropout03", "finance"
]


class FinanceExperimentConfig(StrictModel):
    research_id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    supervised_config: Path
    folds: list[ResearchFoldConfig] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    candidates: list[Candidate] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_matrix(self) -> FinanceExperimentConfig:
        for name, values in (
            ("seed", self.seeds), ("candidate", self.candidates),
            ("fold", [fold.fold_id for fold in self.folds]),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate {name} in experiment matrix")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonnegative")
        for fold in self.folds:
            if fold.train_end >= fold.valid_end:
                raise ValueError("each fold requires Train before V")
        for before, after in zip(self.folds, self.folds[1:], strict=False):
            if after.train_end < before.valid_end or after.valid_end <= before.valid_end:
                raise ValueError("folds must have chronological, non-overlapping V periods")
        return self


def load_finance_experiment_config(path: Path) -> FinanceExperimentConfig:
    path = path.resolve()
    config = FinanceExperimentConfig.model_validate(yaml.safe_load(path.read_text()))
    config.supervised_config = (path.parent / config.supervised_config).resolve()
    return config


def candidate_settings(candidate: str) -> tuple[str | None, float]:
    if candidate == "finance":
        return None, 0.1
    if candidate == "statistics_mlp_dropout03":
        return "statistics_mlp", 0.3
    if candidate in {"statistics_linear", "statistics_mlp"}:
        return candidate, 0.1
    raise ValueError(f"unknown Finance experiment candidate: {candidate}")


def experiment_cells(config: FinanceExperimentConfig) -> list[dict]:
    return [
        {
            "cell_id": f"{fold.fold_id}/seed-{seed}/{candidate}",
            "fold_id": fold.fold_id, "seed": seed, "candidate": candidate,
        }
        for fold in config.folds for seed in config.seeds for candidate in config.candidates
    ]


def cell_config(base: dict, seed: int) -> FinanceTransformerExperimentConfig:
    config = FinanceTransformerExperimentConfig.model_validate({**base, "seed": seed})
    if (
        config.initialization != "scratch"
        or config.evaluation_split != "valid" or config.unlock_test
    ):
        raise ValueError("complete reference cells require scratch with locked test/holdout")
    if config.training.num_workers:
        raise ValueError("complete reference recovery requires num_workers=0")
    if not {5, 20}.issubset(config.model.statistics_windows):
        raise ValueError("complete F/S style observations require statistics windows 5 and 20")
    return config
