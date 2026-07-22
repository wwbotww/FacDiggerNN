from pathlib import Path

import pytest
from pydantic import ValidationError

from facdigger.config import ProjectConfig, load_project_config


def test_base_config_loads_with_us_equities_defaults() -> None:
    config = load_project_config(Path("configs/base.yaml"))

    assert config.market.name == "us_equities_daily"
    assert config.market.exchanges == ["XNYS", "XNAS", "XASE"]
    assert config.model.context_length == 512
    assert config.model.input_channels == 7
    assert config.model.source.minimum_loaded_numel_ratio == 0.8


def test_unknown_config_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate({"unknown": True})


def test_patch_length_cannot_exceed_context() -> None:
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(
            {
                "model": {
                    "context_length": 8,
                    "patch_length": 12,
                }
            }
        )


def test_blank_checkpoint_revision_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate({"model": {"source": {"revision": "  "}}})
