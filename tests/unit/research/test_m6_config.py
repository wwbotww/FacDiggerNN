from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic import ValidationError

from facdigger.experiments.manifest import sha256_json
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

    empty_outer_validation = _payload()
    empty_outer_validation["folds"][0]["valid_end"] = empty_outer_validation["folds"][
        0
    ]["train_end"]
    with pytest.raises(ValidationError, match="research folds require"):
        M6ResearchConfig.model_validate(empty_outer_validation)


def test_m6_positive_cell_ratio_is_an_exact_fraction() -> None:
    config = M6ResearchConfig.model_validate(_payload())
    ratio = config.decisions.minimum_positive_cell_ratio

    assert ratio.model_dump() == {"numerator": 2, "denominator": 3}
    assert ratio.required_count(9) == 6
    assert ratio.required_count(3) == 2
    assert ratio.required_count(10) == 7


@pytest.mark.parametrize(
    "ratio",
    [
        {"numerator": 2, "denominator": 0},
        {"numerator": -1, "denominator": 3},
        {"numerator": 4, "denominator": 3},
        {"numerator": 2.0, "denominator": 3},
        {"numerator": True, "denominator": 3},
    ],
)
def test_m6_rejects_invalid_positive_cell_fraction(ratio) -> None:
    payload = _payload()
    payload["decisions"] = {"minimum_positive_cell_ratio": ratio}
    with pytest.raises(ValidationError):
        M6ResearchConfig.model_validate(payload)


def test_m6_float_ratio_requires_explicit_migration() -> None:
    payload = _payload()
    payload["decisions"] = {"minimum_positive_cell_ratio": 0.6666666667}
    with pytest.raises(ValidationError, match="numerator.*denominator"):
        M6ResearchConfig.model_validate(payload)


@pytest.mark.parametrize("status", ["running", "validation_complete", "holdout_failed"])
def test_exact_ratio_migration_rejects_legacy_resume_without_changing_artifacts(
    tmp_path, status
) -> None:
    from facdigger.research.runner import _resume_research_run

    config = M6ResearchConfig.model_validate(_payload())
    legacy_config = config.model_dump(mode="json")
    legacy_config["decisions"]["minimum_positive_cell_ratio"] = 0.6666666667
    manifest = {"config_hash": sha256_json(legacy_config), "status": status}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "freeze.json").write_text('{"historical_decision":"no_go"}')
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    with pytest.raises(ValueError, match="configuration does not match"):
        _resume_research_run(tmp_path, config)

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
