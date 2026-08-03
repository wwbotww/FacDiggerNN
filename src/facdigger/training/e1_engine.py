"""Supervised E1 PatchTST training with AMP, accumulation and exact epoch resume."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from facdigger.datasets.sampler import DateGroupedBatchSampler
from facdigger.datasets.window import SnapshotWindowDataset
from facdigger.experiments.manifest import sha256_json
from facdigger.models.patchtst_alpha import PatchTSTAlphaModel
from facdigger.training.e1_config import E1ExperimentConfig
from facdigger.training.ranking import (
    RANKING_OBJECTIVE,
    TARGET_TRANSFORM,
    cross_sectional_rank_correlation_loss,
    cross_sectional_rank_targets,
    grouped_rank_ic_audit,
)


def select_device(preference: str) -> str:
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for E1 but is unavailable")
        return "cuda"
    if preference == "auto" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def build_e1_model(
    config: E1ExperimentConfig, *, context_length: int, num_channels: int
) -> PatchTSTAlphaModel:
    model = config.model
    if model.patch_length > context_length:
        raise ValueError("patch_length cannot exceed snapshot context_length")
    return PatchTSTAlphaModel(
        context_length=context_length,
        num_input_channels=num_channels,
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


def _loader(
    dataset: SnapshotWindowDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    minimum_group_size: int = 1,
) -> tuple[DataLoader, DateGroupedBatchSampler]:
    sampler = DateGroupedBatchSampler(
        dataset.asof_dates,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        minimum_group_size=minimum_group_size,
    )
    return (
        DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers),
        sampler,
    )


def _forward_loss(
    model: PatchTSTAlphaModel,
    batch: dict[str, torch.Tensor],
    *,
    device: str,
    target_rank_lookup: torch.Tensor,
    epsilon: float,
    amp_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = batch["values"].to(device=device, dtype=torch.float32)
    observed = batch["observed_mask"].to(device=device, dtype=torch.bool)
    target_rank = target_rank_lookup[batch["sample_index"].long()].to(
        device=device, dtype=torch.float32
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
        score = model(values, observed).score
    with torch.autocast(device_type="cuda", enabled=False):
        loss = cross_sectional_rank_correlation_loss(
            score, target_rank, epsilon=epsilon
        )
    return score, loss


def evaluate_e1_selection(
    model: PatchTSTAlphaModel,
    dataset: SnapshotWindowDataset,
    *,
    batch_size: int,
    device: str,
    precision: str,
    num_workers: int,
    minimum_cross_section_size: int,
    minimum_dates: int,
    minimum_coverage: float,
) -> dict[str, Any]:
    scores = predict_e1(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
        precision=precision,
        num_workers=num_workers,
    )
    return grouped_rank_ic_audit(
        scores,
        dataset.sample_rows["target"].to_numpy(),
        dataset.asof_dates,
        minimum_cross_section_size=minimum_cross_section_size,
        minimum_dates=minimum_dates,
        minimum_coverage=minimum_coverage,
    )


def predict_e1(
    model: PatchTSTAlphaModel,
    dataset: SnapshotWindowDataset,
    *,
    batch_size: int,
    device: str,
    precision: str,
    num_workers: int,
) -> np.ndarray:
    loader, sampler = _loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
    )
    sampler.set_epoch(0)
    amp_enabled = device == "cuda" and precision == "fp16"
    model.eval()
    predictions = np.empty(len(dataset), dtype=np.float64)
    with torch.no_grad():
        for batch in loader:
            values = batch["values"].to(device=device, dtype=torch.float32)
            observed = batch["observed_mask"].to(device=device, dtype=torch.bool)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                score = model(values, observed).score
            indices = batch["sample_index"].numpy()
            predictions[indices] = score.detach().float().cpu().numpy()
    return predictions


def _save_checkpoint(
    path: Path,
    *,
    model: PatchTSTAlphaModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_selection_rank_ic: float,
    best_selection_audit: dict[str, Any],
    best_epoch: int,
    stale_epochs: int,
    history: list[dict[str, Any]],
    sampler: DateGroupedBatchSampler,
    dataset_id: str,
    protocol_hash: str,
    config_payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 2,
            "objective": RANKING_OBJECTIVE,
            "target_transform": TARGET_TRANSFORM,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_selection_rank_ic": best_selection_rank_ic,
            "best_selection_audit": best_selection_audit,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "history": history,
            "sampler_state": sampler.state_dict(),
            "rng_state": _rng_state(),
            "dataset_id": dataset_id,
            "protocol_hash": protocol_hash,
            "config": config_payload,
        },
        temporary,
    )
    temporary.replace(path)


def train_e1(
    config: E1ExperimentConfig,
    *,
    train_dataset: SnapshotWindowDataset,
    valid_dataset: SnapshotWindowDataset,
    dataset_id: str,
    checkpoint_dir: Path,
    resume_from: Path | None = None,
    stop_after_epoch: int | None = None,
) -> tuple[PatchTSTAlphaModel, dict[str, Any]]:
    seed_everything(config.seed)
    device = select_device(config.training.device)
    amp_enabled = device == "cuda" and config.training.precision == "fp16"
    model = build_e1_model(
        config,
        context_length=train_dataset.context_length,
        num_channels=len(train_dataset.channels),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.training.max_epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_loader, train_sampler = _loader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.training.num_workers,
        minimum_group_size=config.training.objective.minimum_cross_section_size,
    )
    train_target_ranks = torch.from_numpy(
        cross_sectional_rank_targets(
            train_dataset.sample_rows["target"].to_numpy(),
            train_dataset.asof_dates,
            minimum_cross_section_size=(
                config.training.objective.minimum_cross_section_size
            ),
        )
    )
    config_payload = config.model_dump(mode="json")
    protocol_hash = sha256_json(config_payload)
    start_epoch = 1
    global_step = 0
    best_selection_rank_ic = float("-inf")
    best_selection_audit: dict[str, Any] = {}
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    resumed_from_epoch: int | None = None
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        if checkpoint.get("schema_version") != 2 or checkpoint.get("objective") != (
            RANKING_OBJECTIVE
        ):
            raise ValueError(
                "resume checkpoint predates the cross-sectional ranking protocol"
            )
        if checkpoint["dataset_id"] != dataset_id:
            raise ValueError("resume checkpoint dataset_id does not match")
        if checkpoint["protocol_hash"] != protocol_hash:
            raise ValueError("resume checkpoint training protocol does not match")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_selection_rank_ic = float(checkpoint["best_selection_rank_ic"])
        best_selection_audit = dict(checkpoint["best_selection_audit"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        history = list(checkpoint["history"])
        train_sampler.load_state_dict(checkpoint["sampler_state"])
        _restore_rng_state(checkpoint["rng_state"])
        resumed_from_epoch = int(checkpoint["epoch"])

    last_checkpoint = checkpoint_dir / "last.pt"
    best_checkpoint = checkpoint_dir / "best.pt"
    for epoch in range(start_epoch, config.training.max_epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_batches = 0
        total_score_std = 0.0
        for batch_index, batch in enumerate(train_loader, start=1):
            score, loss = _forward_loss(
                model,
                batch,
                device=device,
                target_rank_lookup=train_target_ranks,
                epsilon=config.training.objective.epsilon,
                amp_enabled=amp_enabled,
            )
            scaler.scale(loss / config.training.gradient_accumulation_steps).backward()
            total_loss += float(loss.detach().cpu())
            total_batches += 1
            total_score_std += float(score.detach().float().std(unbiased=False).cpu())
            should_step = (
                batch_index % config.training.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        train_loss = total_loss / max(total_batches, 1)
        selection = evaluate_e1_selection(
            model,
            valid_dataset,
            batch_size=config.training.batch_size,
            device=device,
            precision=config.training.precision,
            num_workers=config.training.num_workers,
            minimum_cross_section_size=(
                config.training.objective.minimum_cross_section_size
            ),
            minimum_dates=config.training.objective.minimum_selection_dates,
            minimum_coverage=config.training.objective.minimum_selection_coverage,
        )
        selection_rank_ic = float(selection["mean_rank_ic"])
        scheduler.step()
        improved = selection_rank_ic > best_selection_rank_ic + 1e-12
        if improved:
            best_selection_rank_ic = selection_rank_ic
            best_selection_audit = selection
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_score_std": total_score_std / max(total_batches, 1),
                "selection_mean_rank_ic": selection_rank_ic,
                "selection_valid_dates": selection["valid_dates"],
                "selection_skipped_dates": selection["skipped_dates"],
                "selection_score_std": selection["score_std"],
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
            "best_selection_rank_ic": best_selection_rank_ic,
            "best_selection_audit": best_selection_audit,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "history": history,
            "sampler": train_sampler,
            "dataset_id": dataset_id,
            "protocol_hash": protocol_hash,
            "config_payload": config_payload,
        }
        _save_checkpoint(last_checkpoint, **checkpoint_kwargs)
        if improved:
            _save_checkpoint(best_checkpoint, **checkpoint_kwargs)
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            break
        if epoch >= config.training.minimum_epochs and stale_epochs >= config.training.patience:
            break
    if not best_checkpoint.is_file():
        raise RuntimeError("E1 training did not produce best.pt")
    best = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    audit = {
        "device": device,
        "amp_enabled": amp_enabled,
        "precision": "fp16" if amp_enabled else "fp32",
        "objective": RANKING_OBJECTIVE,
        "target_transform": TARGET_TRANSFORM,
        "best_epoch": best_epoch,
        "best_selection_rank_ic": best_selection_rank_ic,
        "best_selection_audit": best_selection_audit,
        "epochs_completed": history[-1]["epoch"] if history else start_epoch - 1,
        "global_step": global_step,
        "resumed_from_epoch": resumed_from_epoch,
        "history": history,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "last_checkpoint": str(last_checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "protocol_hash": protocol_hash,
    }
    return model, audit
