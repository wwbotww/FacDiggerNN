"""Train-only masked financial pretraining and E3 initialization chaining."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import torch
from torch.utils.data import DataLoader

from facdigger.data.snapshots import sha256_file
from facdigger.datasets.sampler import SequenceBatchSampler
from facdigger.datasets.window import SnapshotWindowDataset
from facdigger.experiments.manifest import sha256_json
from facdigger.models.patchtst_adapter import load_matching_encoder_weights
from facdigger.models.patchtst_alpha import PatchTSTAlphaModel
from facdigger.models.patchtst_pretrain import (
    FinancialPatchTSTPretrainer,
    initialize_financial_pretrainer,
)
from facdigger.models.patchtst_transfer import module_fingerprint
from facdigger.training.e1_engine import (
    _restore_rng_state,
    _rng_state,
    seed_everything,
    select_device,
)
from facdigger.training.e3_config import E3ExperimentConfig

PretrainingInitializer = Callable[[], tuple[FinancialPatchTSTPretrainer, dict[str, Any]]]


def split_pretraining_index(
    sample_index: pl.DataFrame, *, validation_fraction: float
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Chronologically divide only the official train split for reconstruction selection."""

    official_train = sample_index.filter(pl.col("split") == "train").sort(
        ["asof_date", "security_id"]
    )
    dates = official_train["asof_date"].unique().sort().to_list()
    if len(dates) < 2:
        raise ValueError("E3 pretraining requires at least two official train dates")
    holdout_count = max(1, math.ceil(len(dates) * validation_fraction))
    holdout_count = min(holdout_count, len(dates) - 1)
    holdout_dates = dates[-holdout_count:]
    pretrain_rows = official_train.filter(~pl.col("asof_date").is_in(holdout_dates))
    selection_rows = official_train.filter(pl.col("asof_date").is_in(holdout_dates))
    if pretrain_rows.is_empty() or selection_rows.is_empty():
        raise ValueError("E3 train-only pretraining split produced an empty partition")
    pretrain_index = pretrain_rows.with_columns(pl.lit("pretrain_train").alias("split"))
    selection_index = selection_rows.with_columns(pl.lit("pretrain_selection").alias("split"))
    formal_valid = sample_index.filter(pl.col("split") == "valid")
    formal_test = sample_index.filter(pl.col("split") == "test")
    audit = {
        "schema_version": 1,
        "source_split": "train",
        "selection_policy": "chronological_tail_within_official_train",
        "validation_fraction": validation_fraction,
        "official_train_rows": len(official_train),
        "pretrain_rows": len(pretrain_rows),
        "selection_rows": len(selection_rows),
        "official_train_min_asof_date": min(dates),
        "official_train_max_asof_date": max(dates),
        "pretrain_max_asof_date": pretrain_rows["asof_date"].max(),
        "selection_min_asof_date": selection_rows["asof_date"].min(),
        "selection_max_asof_date": selection_rows["asof_date"].max(),
        "formal_valid_min_asof_date": (
            formal_valid["asof_date"].min() if not formal_valid.is_empty() else None
        ),
        "formal_test_min_asof_date": (
            formal_test["asof_date"].min() if not formal_test.is_empty() else None
        ),
        "formal_validation_rows_used": 0,
        "formal_test_rows_used": 0,
    }
    if audit["formal_valid_min_asof_date"] is not None and not (
        audit["selection_max_asof_date"] < audit["formal_valid_min_asof_date"]
    ):
        raise ValueError("E3 pretraining selection overlaps the official validation period")
    if audit["formal_test_min_asof_date"] is not None and not (
        audit["selection_max_asof_date"] < audit["formal_test_min_asof_date"]
    ):
        raise ValueError("E3 pretraining selection overlaps the official test period")
    return pretrain_index, selection_index, audit


def _loader(
    dataset: SnapshotWindowDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, SequenceBatchSampler]:
    sampler = SequenceBatchSampler(
        len(dataset), batch_size=batch_size, shuffle=shuffle, seed=seed
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers), sampler


def _forward(
    model: FinancialPatchTSTPretrainer,
    batch: dict[str, torch.Tensor],
    *,
    device: str,
    amp_enabled: bool,
) -> tuple[torch.Tensor, int]:
    values = batch["values"].to(device=device, dtype=torch.float32)
    observed = batch["observed_mask"].to(device=device, dtype=torch.bool)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
        output = model(values, observed)
    return output.loss, output.valid_element_count


def evaluate_pretraining_loss(
    model: FinancialPatchTSTPretrainer,
    loader: DataLoader,
    *,
    device: str,
    amp_enabled: bool,
    mask_seed: int,
) -> tuple[float, int]:
    model.eval()
    total_loss = 0.0
    total_elements = 0
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(mask_seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(mask_seed)
        for batch in loader:
            loss, elements = _forward(
                model, batch, device=device, amp_enabled=amp_enabled
            )
            total_loss += float(loss.detach().cpu()) * elements
            total_elements += elements
    return total_loss / max(total_elements, 1), total_elements


def _save_pretraining_checkpoint(
    path: Path,
    *,
    model: FinancialPatchTSTPretrainer,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_selection_loss: float,
    best_epoch: int,
    stale_epochs: int,
    history: list[dict[str, Any]],
    sampler: SequenceBatchSampler,
    dataset_id: str,
    protocol_hash: str,
    config_payload: dict[str, Any],
    initialization_audit: dict[str, Any],
    leakage_audit: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "experiment_family": "e3_pretraining",
            "model_state": model.state_dict(),
            "backbone_state": model.backbone.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_selection_loss": best_selection_loss,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "history": history,
            "sampler_state": sampler.state_dict(),
            "rng_state": _rng_state(),
            "dataset_id": dataset_id,
            "protocol_hash": protocol_hash,
            "config": config_payload,
            "initialization_audit": initialization_audit,
            "leakage_audit": leakage_audit,
        },
        temporary,
    )
    temporary.replace(path)


def train_financial_pretraining(
    config: E3ExperimentConfig,
    *,
    train_dataset: SnapshotWindowDataset,
    selection_dataset: SnapshotWindowDataset,
    leakage_audit: dict[str, Any],
    dataset_id: str,
    checkpoint_dir: Path,
    resume_from: Path | None = None,
    stop_after_epoch: int | None = None,
    initializer: PretrainingInitializer | None = None,
) -> tuple[FinancialPatchTSTPretrainer, dict[str, Any], dict[str, Any]]:
    seed_everything(config.seed)
    training = config.pretraining
    device = select_device(training.device)
    amp_enabled = device == "cuda" and training.precision == "fp16"
    initialize = initializer or (
        lambda: initialize_financial_pretrainer(
            model_config=config.model,
            source_config=config.source,
            pretraining_config=config.pretraining,
            context_length=train_dataset.context_length,
            num_channels=len(train_dataset.channels),
        )
    )
    model, initialization_audit = initialize()
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training.max_epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_loader, train_sampler = _loader(
        train_dataset,
        batch_size=training.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=training.num_workers,
    )
    selection_loader, _ = _loader(
        selection_dataset,
        batch_size=training.batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=training.num_workers,
    )
    config_payload = config.model_dump(mode="json")
    protocol_hash = sha256_json(config_payload)
    start_epoch = 1
    global_step = 0
    best_selection_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    resumed_from_epoch: int | None = None
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        if checkpoint.get("experiment_family") != "e3_pretraining":
            raise ValueError("resume checkpoint is not an E3 pretraining checkpoint")
        if checkpoint["dataset_id"] != dataset_id:
            raise ValueError("resume checkpoint dataset_id does not match")
        if checkpoint["protocol_hash"] != protocol_hash:
            raise ValueError("resume checkpoint training protocol does not match")
        saved_hash = checkpoint["initialization_audit"]["source_weights_sha256"]
        if initialization_audit["source_weights_sha256"] != saved_hash:
            raise ValueError("resume checkpoint source weight hash does not match")
        if checkpoint["leakage_audit"] != leakage_audit:
            raise ValueError("resume checkpoint pretraining split audit does not match")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_selection_loss = float(checkpoint["best_selection_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        history = list(checkpoint["history"])
        train_sampler.load_state_dict(checkpoint["sampler_state"])
        _restore_rng_state(checkpoint["rng_state"])
        resumed_from_epoch = int(checkpoint["epoch"])

    last_checkpoint = checkpoint_dir / "last.pt"
    best_checkpoint = checkpoint_dir / "best.pt"
    for epoch in range(start_epoch, training.max_epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_elements = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            loss, elements = _forward(model, batch, device=device, amp_enabled=amp_enabled)
            scaler.scale(loss / training.gradient_accumulation_steps).backward()
            total_loss += float(loss.detach().cpu()) * elements
            total_elements += elements
            should_step = (
                batch_index % training.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        train_loss = total_loss / max(total_elements, 1)
        selection_loss, selection_elements = evaluate_pretraining_loss(
            model,
            selection_loader,
            device=device,
            amp_enabled=amp_enabled,
            mask_seed=config.seed + 1_000_000,
        )
        scheduler.step()
        improved = selection_loss < best_selection_loss - 1e-12
        if improved:
            best_selection_loss = selection_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "selection_loss": selection_loss,
                "train_masked_observed_elements": total_elements,
                "selection_masked_observed_elements": selection_elements,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "global_step": global_step,
            }
        )
        checkpoint_kwargs = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "epoch": epoch,
            "global_step": global_step,
            "best_selection_loss": best_selection_loss,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "history": history,
            "sampler": train_sampler,
            "dataset_id": dataset_id,
            "protocol_hash": protocol_hash,
            "config_payload": config_payload,
            "initialization_audit": initialization_audit,
            "leakage_audit": leakage_audit,
        }
        _save_pretraining_checkpoint(last_checkpoint, **checkpoint_kwargs)
        if improved:
            _save_pretraining_checkpoint(best_checkpoint, **checkpoint_kwargs)
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            break
        if epoch >= training.minimum_epochs and stale_epochs >= training.patience:
            break
    if not best_checkpoint.is_file():
        raise RuntimeError("E3 pretraining did not produce best.pt")
    best = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    audit = {
        "device": device,
        "amp_enabled": amp_enabled,
        "precision": "fp16" if amp_enabled else "fp32",
        "best_epoch": best_epoch,
        "best_selection_loss": best_selection_loss,
        "epochs_completed": history[-1]["epoch"] if history else start_epoch - 1,
        "global_step": global_step,
        "resumed_from_epoch": resumed_from_epoch,
        "history": history,
        "leakage_audit": leakage_audit,
        "last_checkpoint": str(last_checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "protocol_hash": protocol_hash,
    }
    return model, audit, initialization_audit


def _build_alpha(config: E3ExperimentConfig, context_length: int) -> PatchTSTAlphaModel:
    model = config.model
    return PatchTSTAlphaModel(
        context_length=context_length,
        num_input_channels=len(config.channels),
        patch_length=model.patch_length,
        patch_stride=model.patch_stride,
        d_model=model.d_model,
        num_attention_heads=model.num_attention_heads,
        num_hidden_layers=model.num_hidden_layers,
        ffn_dim=model.ffn_dim,
        dropout=model.dropout,
        attention_dropout=model.attention_dropout,
        positional_dropout=model.positional_dropout,
        path_dropout=model.path_dropout,
        ff_dropout=model.ff_dropout,
        norm_type=model.norm_type,
        pre_norm=model.pre_norm,
        scaling=model.scaling,
        alpha_hidden_dim=model.alpha_hidden_dim,
        alpha_dropout=model.alpha_dropout,
    )


def initialize_alpha_from_financial_checkpoint(
    config: E3ExperimentConfig,
    *,
    context_length: int,
    checkpoint_path: Path,
) -> tuple[PatchTSTAlphaModel, dict[str, Any]]:
    """Strictly transfer the selected financial encoder into a fresh alpha head."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment_family") != "e3_pretraining":
        raise ValueError("financial initializer is not an E3 pretraining checkpoint")
    target = _build_alpha(config, context_length)
    random_fingerprint = module_fingerprint(target.backbone)
    backbone_state = checkpoint["backbone_state"]
    parameter_names = {name for name, _ in target.backbone.named_parameters()}
    report = load_matching_encoder_weights(
        target_backbone=target.backbone,
        source_encoder_state=backbone_state,
        minimum_loaded_numel_ratio=1.0,
        allowlist=(),
        source_parameter_names=parameter_names,
    )
    transferred_fingerprint = module_fingerprint(target.backbone)
    source_fingerprint = module_fingerprint_from_state(backbone_state)
    if transferred_fingerprint != source_fingerprint:
        raise RuntimeError("alpha backbone differs from selected financial encoder")
    audit = {
        "schema_version": 1,
        "initialization_chain": "ETTh1 -> financial masked pretraining -> alpha",
        "source_model": checkpoint["initialization_audit"]["source_model"],
        "source_revision": checkpoint["initialization_audit"]["source_revision"],
        "etth1_weights_sha256": checkpoint["initialization_audit"][
            "source_weights_sha256"
        ],
        "source_weights_sha256": sha256_file(checkpoint_path),
        "financial_checkpoint": str(checkpoint_path.resolve()),
        "financial_checkpoint_sha256": sha256_file(checkpoint_path),
        "financial_checkpoint_epoch": checkpoint["epoch"],
        "financial_selection_loss": checkpoint["best_selection_loss"],
        "financial_pretraining_leakage_audit": checkpoint["leakage_audit"],
        "financial_backbone_to_alpha": report.to_dict(),
        "upstream_weight_loading": checkpoint["initialization_audit"],
        "fingerprints": {
            "random_alpha_before_financial_transfer": random_fingerprint,
            "financial_backbone": source_fingerprint,
            "alpha_backbone_after_financial_transfer": transferred_fingerprint,
        },
    }
    return target, audit


def module_fingerprint_from_state(state: dict[str, torch.Tensor]) -> str:
    """Fingerprint a state dict through a temporary module-independent wrapper."""

    class _StateModule(torch.nn.Module):
        def __init__(self, values: dict[str, torch.Tensor]) -> None:
            super().__init__()
            self._values = values

        def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            return self._values

    return module_fingerprint(_StateModule(state))
