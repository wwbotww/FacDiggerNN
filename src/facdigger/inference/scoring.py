"""Shared PatchTST model construction and target-free E3 scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from facdigger.data.contracts import DataContractError
from facdigger.datasets.window import SnapshotInferenceWindowDataset
from facdigger.models.patchtst_alpha import PatchTSTAlphaModel
from facdigger.training.common import load_required_snapshot_features
from facdigger.training.e1_config import E1ExperimentConfig
from facdigger.training.e1_engine import predict_e1, select_device
from facdigger.training.e2_config import E2ExperimentConfig
from facdigger.training.e3_config import E3ExperimentConfig

if TYPE_CHECKING:
    from facdigger.inference.releases import ModelReleaseManifest


def build_patchtst_model(
    model_config: Any, *, context_length: int, num_channels: int
) -> PatchTSTAlphaModel:
    """Build the one alpha-model architecture used by replay and signal inference."""

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
    config = E3ExperimentConfig.model_validate(config_payload)
    return config, config.finetuning


@dataclass
class E3InferenceRuntime:
    """Verified fixed E3 release with a model loaded lazily once per process."""

    release: ModelReleaseManifest
    config: E3ExperimentConfig
    checkpoint_path: Path
    device_preference: Literal["auto", "cpu", "cuda"]
    batch_size: int
    num_workers: int
    _model: PatchTSTAlphaModel | None = field(default=None, init=False, repr=False)
    _device: str | None = field(default=None, init=False, repr=False)
    _precision: str | None = field(default=None, init=False, repr=False)

    def model_runtime(self) -> tuple[PatchTSTAlphaModel, str, str]:
        """Load and strictly verify the checkpoint on first use."""

        if self._model is not None:
            assert self._device is not None
            assert self._precision is not None
            return self._model, self._device, self._precision

        import torch

        selected_device = select_device(self.device_preference)
        model = build_patchtst_model(
            self.config.model,
            context_length=self.release.feature_contract.context_length,
            num_channels=len(self.config.channels),
        ).to(selected_device)
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=selected_device,
            weights_only=False,
        )
        if checkpoint.get("dataset_id") != self.release.training_data.dataset_id:
            raise DataContractError("checkpoint training dataset differs from ModelRelease")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        self._model = model
        self._device = selected_device
        self._precision = (
            self.config.finetuning.precision if selected_device == "cuda" else "fp32"
        )
        return self._model, self._device, self._precision


def load_e3_inference_runtime(
    release_dir: str | Path,
    *,
    device: Literal["auto", "cpu", "cuda"] = "cpu",
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> E3InferenceRuntime:
    """Verify an E3 release without loading its model until rows need scoring."""

    # Delayed import avoids a cycle: releases validates source runs through runner.
    from facdigger.inference.releases import release_runtime

    release, config_payload, checkpoint_path = release_runtime(release_dir)
    if release.model_type != "financial_pretrained_patchtst":
        raise DataContractError("target-free factor inference accepts only an E3 ModelRelease")
    config = E3ExperimentConfig.model_validate(config_payload)
    channels = list(release.feature_contract.channels)
    if config.channels != channels:
        raise DataContractError("release channels differ from resolved E3 configuration")
    effective_batch_size = batch_size or config.finetuning.batch_size
    effective_workers = (
        config.finetuning.num_workers if num_workers is None else num_workers
    )
    if effective_batch_size < 1:
        raise ValueError("inference batch_size must be positive")
    if effective_workers < 0:
        raise ValueError("inference num_workers must be non-negative")
    return E3InferenceRuntime(
        release=release,
        config=config,
        checkpoint_path=checkpoint_path,
        device_preference=device,
        batch_size=effective_batch_size,
        num_workers=effective_workers,
    )


def score_e3_inference_rows(
    runtime: E3InferenceRuntime,
    *,
    snapshot_dir: str | Path,
    snapshot_manifest: dict[str, Any],
    rows: pl.DataFrame,
) -> pl.DataFrame:
    """Score canonical target-free rows with the shared fixed E3 model."""

    output_schema = {
        "security_id": pl.String,
        "symbol": pl.String,
        "asof_date": pl.Date,
        "score": pl.Float64,
    }
    if rows.is_empty():
        return pl.DataFrame(schema=output_schema)
    required = {
        "sample_id",
        "security_id",
        "symbol",
        "asof_date",
        "feature_start",
        "feature_end",
        "eligible",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise DataContractError(f"inference rows are missing required fields: {missing}")
    canonical_rows = rows.sort("asof_date", "security_id")
    if not canonical_rows["eligible"].all():
        raise DataContractError("E3 scoring rows must all be eligible")
    if canonical_rows["sample_id"].n_unique() != canonical_rows.height:
        raise DataContractError("inference rows contain duplicate sample IDs")
    if canonical_rows.select("security_id", "asof_date").n_unique() != (
        canonical_rows.height
    ):
        raise DataContractError("inference rows contain duplicate security/date keys")

    snapshot_path = Path(snapshot_dir).resolve()
    channels = list(runtime.release.feature_contract.channels)
    required_features = load_required_snapshot_features(
        snapshot_path,
        {
            "config": {"features": {"channels": channels}},
            "artifacts": snapshot_manifest["artifacts"],
        },
        canonical_rows,
    )
    dataset = SnapshotInferenceWindowDataset(
        features=required_features,
        inference_index=canonical_rows,
        channels=channels,
        context_length=runtime.release.feature_contract.context_length,
    )
    model, selected_device, precision = runtime.model_runtime()
    scores = predict_e1(
        model,
        dataset,
        batch_size=runtime.batch_size,
        device=selected_device,
        precision=precision,
        num_workers=runtime.num_workers,
    )
    return (
        dataset.sample_rows.select("security_id", "symbol", "asof_date")
        .with_columns(pl.Series("score", scores, dtype=pl.Float64))
        .select(list(output_schema))
        .sort("asof_date", "security_id")
    )
