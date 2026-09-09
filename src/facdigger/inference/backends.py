"""Concrete model-family adapters behind one checkpoint inference boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import polars as pl

from facdigger.data.contracts import DataContractError
from facdigger.datasets.window import (
    FinanceTransformerInferenceWindowDataset,
    SnapshotInferenceWindowDataset,
)
from facdigger.training.common import (
    load_required_market_features,
    load_required_snapshot_features,
)
from facdigger.training.e1_config import E1ExperimentConfig
from facdigger.training.e2_config import E2ExperimentConfig
from facdigger.training.e3_config import E3ExperimentConfig

PATCHTST_MODEL_TYPES = {
    "random_patchtst": "e1",
    "etth1_transferred_patchtst": "e2",
    "financial_pretrained_patchtst": "e2",
}
ReleasableModelType = Literal[
    "random_patchtst",
    "etth1_transferred_patchtst",
    "financial_pretrained_patchtst",
    "finance_patch_transformer",
]
RELEASABLE_MODEL_TYPES = {*PATCHTST_MODEL_TYPES, "finance_patch_transformer"}


def build_patchtst_model(model_config: Any, *, context_length: int, num_channels: int) -> Any:
    """Build the one alpha-model architecture used by replay and signal inference."""

    from facdigger.models.patchtst_alpha import PatchTSTAlphaModel

    return PatchTSTAlphaModel(
        context_length=context_length,
        num_input_channels=num_channels,
        patch_length=model_config.patch_length,
        patch_stride=model_config.patch_stride,
        d_model=model_config.d_model,
        num_attention_heads=model_config.num_attention_heads,
        num_hidden_layers=model_config.num_hidden_layers,
        ffn_dim=model_config.ffn_dim,
        dropout=model_config.dropout,
        attention_dropout=model_config.attention_dropout,
        positional_dropout=model_config.positional_dropout,
        path_dropout=model_config.path_dropout,
        ff_dropout=model_config.ff_dropout,
        norm_type=model_config.norm_type,
        pre_norm=model_config.pre_norm,
        scaling=model_config.scaling,
        alpha_hidden_dim=model_config.alpha_hidden_dim,
        alpha_dropout=model_config.alpha_dropout,
    )


def patch_config(model_type: str, config_payload: dict[str, Any]) -> tuple[Any, Any]:
    """Parse one resolved PatchTST configuration and return its inference settings."""

    if model_type == "random_patchtst":
        config = E1ExperimentConfig.model_validate(config_payload)
        return config, config.training
    if model_type == "etth1_transferred_patchtst":
        config = E2ExperimentConfig.model_validate(config_payload)
        return config, config.training
    if model_type == "financial_pretrained_patchtst":
        config = E3ExperimentConfig.model_validate(config_payload)
        return config, config.finetuning
    raise DataContractError(f"unsupported PatchTST model_type: {model_type}")


class CheckpointBackend(ABC):
    """Cache one strictly loaded model; concrete families own config and forward."""

    full_date_cross_section = False
    loader_name = "strict_patchtst_state_dict"

    def __init__(
        self,
        *,
        model_type: str,
        config_payload: dict[str, Any],
        checkpoint_path: Path,
        training_dataset_id: str,
        context_length: int,
        device: str = "cpu",
        batch_size: int | None = None,
        num_workers: int | None = None,
    ) -> None:
        self.model_type = model_type
        self.config, self.settings = self.parse_config(model_type, config_payload)
        self.checkpoint_path = checkpoint_path
        self.training_dataset_id = training_dataset_id
        self.context_length = context_length
        self.device_preference = device
        self.batch_size = self.settings.batch_size if batch_size is None else batch_size
        self.num_workers = self.settings.num_workers if num_workers is None else num_workers
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("inference batch_size must be positive and num_workers non-negative")
        self._model: Any = None
        self._device: str | None = None
        self._precision: str | None = None
        self._audit: dict[str, Any] = {}

    @staticmethod
    @abstractmethod
    def parse_config(model_type: str, payload: dict[str, Any]) -> tuple[Any, Any]:
        """Return validated model configuration and physical inference settings."""

    @abstractmethod
    def build_model(self) -> Any:
        """Construct the exact architecture declared by the source configuration."""

    def model_runtime(self) -> tuple[Any, str, str]:
        if self._model is None:
            import torch

            from facdigger.training.e1_engine import select_device

            device = select_device(self.device_preference)
            model = self.build_model().to(device)
            checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
            if checkpoint.get("dataset_id") != self.training_dataset_id:
                raise DataContractError("checkpoint training dataset differs from source")
            self.validate_checkpoint(checkpoint)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            self._model = model
            self._device = device
            self._precision = self.settings.precision if device == "cuda" else "fp32"
            self._audit = {
                "loader": self.loader_name,
                "device": device,
                "precision": self._precision,
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_best_epoch": checkpoint.get("best_epoch"),
            }
            if self.full_date_cross_section:
                self._audit["full_date_cross_section"] = True
        assert self._device is not None and self._precision is not None
        return self._model, self._device, self._precision

    def validate_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Families may additionally verify their distinct checkpoint envelope."""
        if not isinstance(checkpoint.get("model_state"), dict):
            raise DataContractError("checkpoint has no model state dictionary")

    @abstractmethod
    def predict(
        self,
        snapshot_dir: Path,
        snapshot_manifest: dict[str, Any],
        rows: pl.DataFrame,
    ) -> pl.DataFrame:
        """Score the supplied computational universe, never read labels."""

    @property
    def audit(self) -> dict[str, Any]:
        return dict(self._audit)

    @staticmethod
    def score_frame(dataset: SnapshotInferenceWindowDataset, scores: Any) -> pl.DataFrame:
        return (
            dataset.sample_rows.select("security_id", "symbol", "asof_date")
            .with_columns(pl.Series("score", scores, dtype=pl.Float64))
            .sort("asof_date", "security_id")
        )


class PatchTSTBackend(CheckpointBackend):
    @staticmethod
    def parse_config(model_type: str, payload: dict[str, Any]) -> tuple[Any, Any]:
        return patch_config(model_type, payload)

    def build_model(self) -> Any:
        return build_patchtst_model(
            self.config.model,
            context_length=self.context_length,
            num_channels=len(self.config.channels),
        )

    def predict(
        self, snapshot_dir: Path, snapshot_manifest: dict[str, Any], rows: pl.DataFrame
    ) -> pl.DataFrame:
        from facdigger.training.e1_engine import predict_e1

        features = load_required_snapshot_features(snapshot_dir, snapshot_manifest, rows)
        dataset = SnapshotInferenceWindowDataset(
            features=features,
            inference_index=rows,
            channels=self.config.channels,
            context_length=self.context_length,
        )
        model, device, precision = self.model_runtime()
        scores = predict_e1(
            model,
            dataset,
            batch_size=self.batch_size,
            device=device,
            precision=precision,
            num_workers=self.num_workers,
        )
        return self.score_frame(dataset, scores)


class FinanceTransformerBackend(CheckpointBackend):
    full_date_cross_section = True
    loader_name = "strict_finance_transformer_state_dict"

    @staticmethod
    def parse_config(model_type: str, payload: dict[str, Any]) -> tuple[Any, Any]:
        from facdigger.training.finance_transformer_config import (
            FinanceTransformerExperimentConfig,
        )

        config = FinanceTransformerExperimentConfig.model_validate(payload)
        return config, config.training

    def build_model(self) -> Any:
        from facdigger.models.finance_patch_transformer import build_finance_transformer_model

        return build_finance_transformer_model(self.config, context_length=self.context_length)

    def validate_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().validate_checkpoint(checkpoint)
        if checkpoint.get("contract") != "finance_patch_transformer_checkpoint":
            raise DataContractError("checkpoint is not a finance Transformer checkpoint")

    def predict(
        self, snapshot_dir: Path, snapshot_manifest: dict[str, Any], rows: pl.DataFrame
    ) -> pl.DataFrame:
        from facdigger.models.finance_scoring import predict_finance_transformer

        dataset = FinanceTransformerInferenceWindowDataset(
            features=load_required_snapshot_features(snapshot_dir, snapshot_manifest, rows),
            market_features=load_required_market_features(snapshot_dir, snapshot_manifest, rows),
            inference_index=rows,
            channels=self.config.channels,
            market_channels=self.config.market_channels,
            context_length=self.context_length,
            primary_horizon=self.config.primary_horizon,
        )
        model, device, precision = self.model_runtime()
        scores = predict_finance_transformer(
            model,
            dataset,
            batch_size=self.batch_size,
            device=device,
            precision=precision,
            num_workers=self.num_workers,
        )
        return self.score_frame(dataset, scores)


def load_checkpoint_backend(model_type: str, **kwargs: Any) -> CheckpointBackend:
    """The sole explicit model-family selection point; unknown families fail."""
    backend: type[CheckpointBackend]
    if model_type in PATCHTST_MODEL_TYPES:
        backend = PatchTSTBackend
    elif model_type == "finance_patch_transformer":
        backend = FinanceTransformerBackend
    else:
        raise DataContractError(f"unsupported inference model_type: {model_type}")
    return backend(model_type=model_type, **kwargs)
