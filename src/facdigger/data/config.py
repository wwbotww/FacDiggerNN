"""Configuration for the standard-Parquet dataset pipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

DEFAULT_CHANNELS = [
    "r_close",
    "r_gap",
    "r_intraday",
    "range",
    "dlog_volume",
    "vol20",
    "dollar_volume_z20",
]

RANK_CHANNELS = [f"rank_{channel}" for channel in DEFAULT_CHANNELS]
FINANCE_TRANSFORMER_CHANNELS = [*DEFAULT_CHANNELS, *RANK_CHANNELS]
MARKET_CONTEXT_CHANNELS = [
    "market_return_median",
    "market_breadth",
    "market_return_dispersion",
    "market_range_median",
    "market_volume_activity",
    "market_vol20",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ParquetSourceConfig(StrictModel):
    bars: Path
    universe: Path
    corporate_actions: Path | None = None
    delistings: Path | None = None
    source_manifest: Path | None = None


class FeatureSetConfig(StrictModel):
    name: str = "price_volume_v1"
    context_length: int = Field(default=512, gt=0)
    channels: list[str] = Field(default_factory=lambda: list(DEFAULT_CHANNELS))
    market_channels: list[str] = Field(default_factory=list)
    scaler: Literal["train_global_robust"] = "train_global_robust"
    winsor_lower: float = Field(default=0.005, ge=0, lt=0.5)
    winsor_upper: float = Field(default=0.995, gt=0.5, le=1)

    @model_validator(mode="after")
    def validate_channels(self) -> FeatureSetConfig:
        expected = {
            "price_volume_v1": (DEFAULT_CHANNELS, []),
            "finance_transformer": (
                FINANCE_TRANSFORMER_CHANNELS,
                MARKET_CONTEXT_CHANNELS,
            ),
        }
        if self.name not in expected:
            raise ValueError(f"Unsupported feature set: {self.name}")
        expected_channels, expected_market_channels = expected[self.name]
        if self.channels != expected_channels:
            raise ValueError(
                f"{self.name} requires the ordered channels {expected_channels}"
            )
        if self.market_channels != expected_market_channels:
            raise ValueError(
                f"{self.name} requires the ordered market channels "
                f"{expected_market_channels}"
            )
        if self.winsor_lower >= self.winsor_upper:
            raise ValueError("winsor_lower must be smaller than winsor_upper")
        return self


class LabelConfig(StrictModel):
    name: str = "next_open_to_fifth_close_excess_return"
    execution_lag: int = Field(default=1, ge=1)
    horizon: int = Field(default=5, ge=1)
    auxiliary_horizons: list[int] = Field(default_factory=list)
    benchmark: Literal["eligible_equal_weight"] = "eligible_equal_weight"

    @model_validator(mode="after")
    def validate_horizons(self) -> LabelConfig:
        horizons = [self.horizon, *self.auxiliary_horizons]
        if len(set(horizons)) != len(horizons):
            raise ValueError("label horizons must be unique")
        if any(horizon < self.execution_lag for horizon in horizons):
            raise ValueError("all label horizons must be >= execution_lag")
        if self.auxiliary_horizons != sorted(self.auxiliary_horizons):
            raise ValueError("auxiliary_horizons must be sorted")
        return self

    @property
    def all_horizons(self) -> list[int]:
        return sorted([self.horizon, *self.auxiliary_horizons])


class SplitConfig(StrictModel):
    train_end: date
    valid_end: date
    test_end: date
    embargo_sessions: int = Field(default=5, ge=0)

    @model_validator(mode="after")
    def dates_are_chronological(self) -> SplitConfig:
        if not self.train_end <= self.valid_end < self.test_end:
            raise ValueError("split dates must satisfy train_end <= valid_end < test_end")
        return self


class FinanceSelectionConfig(StrictModel):
    """Frozen inner selection shared by every Finance training entry point."""

    supervised_selection_fraction: float = Field(default=0.15, gt=0, lt=0.5)
    probe_fit_dates: int = Field(default=60, ge=2, strict=True)
    probe_selection_dates: int = Field(default=20, ge=2, strict=True)
    future_horizon: Literal[5] = 5

    def validate_expectations(self, **expectations: object) -> None:
        """Experiment declarations are assertions, never another plan source."""
        from facdigger.data.contracts import DataContractError

        actual = self.model_dump(mode="json")
        unknown = sorted(set(expectations) - set(actual))
        if unknown:
            raise DataContractError(f"Unknown Finance selection expectations: {unknown}")
        mismatched = sorted(
            field for field, expected in expectations.items() if actual[field] != expected
        )
        if mismatched:
            raise DataContractError(
                f"Finance selection experiment differs from snapshot plan: {mismatched}; "
                "align configurations and rebuild the snapshot"
            )


class DatasetBuildConfig(StrictModel):
    dataset_name: str = "us_equities_daily_v1"
    sources: ParquetSourceConfig
    output_root: Path = Path("data/snapshots")
    features: FeatureSetConfig = Field(default_factory=FeatureSetConfig)
    label: LabelConfig = Field(default_factory=LabelConfig)
    split: SplitConfig
    finance_selection: FinanceSelectionConfig | None = None

    @model_serializer(mode="wrap")
    def serialize_existing_fields(self, handler: Any) -> dict[str, Any]:
        # Full config dumps also enter older M6 protocol hashes. Omit only the
        # absent new field, preserving every existing explicit None elsewhere.
        payload = handler(self)
        if self.finance_selection is None:
            payload.pop("finance_selection", None)
        return payload

    @model_validator(mode="after")
    def validate_finance_selection(self) -> DatasetBuildConfig:
        if self.finance_selection is not None and (
            self.features.name != "finance_transformer"
            or self.label.horizon != 5
            or self.label.all_horizons != [1, 5, 20]
        ):
            raise ValueError("finance_selection requires Finance features and 1/5/20 labels")
        return self


def semantic_dataset_config(config: DatasetBuildConfig) -> dict[str, Any]:
    """Preserve existing identities when no Finance selection plan is enabled."""
    semantic = config.model_dump(mode="json", exclude={"sources", "output_root"})
    if semantic.get("finance_selection") is None:
        semantic.pop("finance_selection", None)
    return semantic


class InferenceSnapshotConfig(StrictModel):
    """Target-free snapshot input configured independently from training splits."""

    dataset_name: str = "us_equities_daily_inference"
    sources: ParquetSourceConfig
    output_root: Path = Path("data/inference_snapshots")


def load_dataset_build_config(path: str | Path) -> DatasetBuildConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return DatasetBuildConfig.model_validate(raw)


def load_inference_snapshot_config(path: str | Path) -> InferenceSnapshotConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Inference snapshot configuration not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return InferenceSnapshotConfig.model_validate(raw)
