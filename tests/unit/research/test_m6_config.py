from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from facdigger.research.config import M6ResearchConfig


def _payload() -> dict:
    return {
        "base_dataset_config": "dataset.yaml",
        "models": {"e0": "e0.yaml", "e1": "e1.yaml", "e2": "e2.yaml", "e3": "e3.yaml"},
        "seeds": [1, 2, 3],
        "folds": [
            {
                "fold_id": f"fold-{index}",
                "train_end": date(2020 + index, 1, 1),
                "valid_end": date(2020 + index, 6, 1),
                "test_end": date(2020 + index, 12, 1),
            }
            for index in range(3)
        ],
    }


def test_m6_requires_three_unique_seeds_and_expanding_folds() -> None:
    config = M6ResearchConfig.model_validate(_payload())
    assert len(config.folds) == 3

    duplicate = _payload()
    duplicate["seeds"] = [1, 1, 2]
    with pytest.raises(ValidationError, match="unique"):
        M6ResearchConfig.model_validate(duplicate)

    reversed_folds = _payload()
    reversed_folds["folds"][1]["train_end"] = date(2019, 1, 1)
    with pytest.raises(ValidationError, match="strictly expand"):
        M6ResearchConfig.model_validate(reversed_folds)
