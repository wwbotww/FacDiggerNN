"""E2 staged fine-tuning with strict transfer and auditable parameter changes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from facdigger.datasets.sampler import DateGroupedBatchSampler
from facdigger.datasets.window import SnapshotWindowDataset
from facdigger.experiments.manifest import sha256_json
from facdigger.models.patchtst_alpha import PatchTSTAlphaModel
from facdigger.models.patchtst_transfer import (
    initialize_transferred_alpha_model,
    module_fingerprint,
)
from facdigger.training.e1_engine import (
    _forward_loss,
    _loader,
    _restore_rng_state,
    _rng_state,
    evaluate_e1_loss,
    seed_everything,
    select_device,
)
from facdigger.training.e2_config import E2ExperimentConfig

Initializer = Callable[[], tuple[PatchTSTAlphaModel, dict[str, Any]]]


def finetune_stage(config: E2ExperimentConfig, epoch: int) -> str:
    return "ft0_head_only" if epoch <= config.training.head_only_epochs else "ft1_last_blocks"


def configure_finetune_stage(
    model: PatchTSTAlphaModel, config: E2ExperimentConfig, epoch: int
) -> dict[str, Any]:
    stage = finetune_stage(config, epoch)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.alpha_head.parameters():
        parameter.requires_grad = True
    unfrozen_blocks: list[int] = []
    if stage == "ft1_last_blocks":
        depth = len(model.backbone.encoder.layers)
        start = depth - config.training.unfreeze_last_n_blocks
        for index in range(start, depth):
            for parameter in model.backbone.encoder.layers[index].parameters():
                parameter.requires_grad = True
            unfrozen_blocks.append(index)

    model.train()
    model.backbone.eval()
    if stage == "ft1_last_blocks":
        for index in unfrozen_blocks:
            model.backbone.encoder.layers[index].train()
    model.alpha_head.train()
    return {
        "name": stage,
        "unfrozen_blocks": unfrozen_blocks,
        "trainable_encoder_parameters": sum(
            parameter.numel()
            for parameter in model.backbone.parameters()
            if parameter.requires_grad
        ),
        "trainable_head_parameters": sum(
            parameter.numel()
            for parameter in model.alpha_head.parameters()
            if parameter.requires_grad
        ),
    }


def _save_checkpoint(
    path: Path,
    *,
    model: PatchTSTAlphaModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_valid_loss: float,
    best_epoch: int,
    stale_epochs: int,
    history: list[dict[str, Any]],
    stage_audits: dict[str, dict[str, Any]],
    sampler: DateGroupedBatchSampler,
    dataset_id: str,
    protocol_hash: str,
    config_payload: dict[str, Any],
    initialization_audit: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "experiment_family": "e2",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_valid_loss": best_valid_loss,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "history": history,
            "stage_audits": stage_audits,
            "sampler_state": sampler.state_dict(),
            "rng_state": _rng_state(),
            "dataset_id": dataset_id,
            "protocol_hash": protocol_hash,
            "config": config_payload,
            "initialization_audit": initialization_audit,
        },
        temporary,
    )
    temporary.replace(path)


def train_e2(
    config: E2ExperimentConfig,
    *,
    train_dataset: SnapshotWindowDataset,
    valid_dataset: SnapshotWindowDataset,
    dataset_id: str,
    checkpoint_dir: Path,
    resume_from: Path | None = None,
    stop_after_epoch: int | None = None,
    initializer: Initializer | None = None,
) -> tuple[PatchTSTAlphaModel, dict[str, Any], dict[str, Any]]:
    seed_everything(config.seed)
    device = select_device(config.training.device)
    amp_enabled = device == "cuda" and config.training.precision == "fp16"
    initialize = initializer or (
        lambda: initialize_transferred_alpha_model(
            model_config=config.model,
            source_config=config.source,
            context_length=train_dataset.context_length,
            num_channels=len(train_dataset.channels),
        )
    )
    model, initialization_audit = initialize()
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.backbone.parameters()),
                "lr": config.training.encoder_learning_rate,
                "name": "encoder",
            },
            {
                "params": list(model.alpha_head.parameters()),
                "lr": config.training.head_learning_rate,
                "name": "head",
            },
        ],
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
    )
    valid_loader, _ = _loader(
        valid_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.training.num_workers,
    )
    config_payload = config.model_dump(mode="json")
    protocol_hash = sha256_json(config_payload)
    start_epoch = 1
    global_step = 0
    best_valid_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    stage_audits: dict[str, dict[str, Any]] = {}
    resumed_from_epoch: int | None = None
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        if checkpoint.get("experiment_family") != "e2":
            raise ValueError("resume checkpoint is not an E2 checkpoint")
        if checkpoint["dataset_id"] != dataset_id:
            raise ValueError("resume checkpoint dataset_id does not match")
        if checkpoint["protocol_hash"] != protocol_hash:
            raise ValueError("resume checkpoint training protocol does not match")
        saved_source_hash = checkpoint["initialization_audit"]["source_weights_sha256"]
        if initialization_audit["source_weights_sha256"] != saved_source_hash:
            raise ValueError("resume checkpoint source weight hash does not match")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_valid_loss = float(checkpoint["best_valid_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        history = list(checkpoint["history"])
        stage_audits = dict(checkpoint["stage_audits"])
        train_sampler.load_state_dict(checkpoint["sampler_state"])
        _restore_rng_state(checkpoint["rng_state"])
        resumed_from_epoch = int(checkpoint["epoch"])

    loss_function = nn.HuberLoss()
    last_checkpoint = checkpoint_dir / "last.pt"
    best_checkpoint = checkpoint_dir / "best.pt"
    for epoch in range(start_epoch, config.training.max_epochs + 1):
        train_sampler.set_epoch(epoch)
        stage = configure_finetune_stage(model, config, epoch)
        stage_name = str(stage["name"])
        capture_stage_step = stage_name not in stage_audits
        encoder_before = module_fingerprint(model.backbone) if capture_stage_step else None
        head_before = module_fingerprint(model.alpha_head) if capture_stage_step else None
        stage_step_captured = False
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_rows = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            _, loss = _forward_loss(
                model,
                batch,
                device=device,
                loss_function=loss_function,
                amp_enabled=amp_enabled,
            )
            scaler.scale(loss / config.training.gradient_accumulation_steps).backward()
            total_loss += float(loss.detach().cpu()) * len(batch["target"])
            total_rows += len(batch["target"])
            should_step = (
                batch_index % config.training.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                trainable = [
                    parameter for parameter in model.parameters() if parameter.requires_grad
                ]
                torch.nn.utils.clip_grad_norm_(trainable, config.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if capture_stage_step and not stage_step_captured:
                    encoder_after = module_fingerprint(model.backbone)
                    head_after = module_fingerprint(model.alpha_head)
                    encoder_changed = encoder_before != encoder_after
                    head_changed = head_before != head_after
                    expected_encoder_change = stage_name == "ft1_last_blocks"
                    if encoder_changed != expected_encoder_change:
                        raise RuntimeError(
                            f"{stage_name} encoder change audit failed: {encoder_changed}"
                        )
                    if not head_changed:
                        raise RuntimeError(f"{stage_name} alpha head did not update")
                    stage_audits[stage_name] = {
                        **stage,
                        "first_step_global_step": global_step,
                        "encoder_before": encoder_before,
                        "encoder_after": encoder_after,
                        "encoder_changed": encoder_changed,
                        "head_before": head_before,
                        "head_after": head_after,
                        "head_changed": head_changed,
                    }
                    stage_step_captured = True

        train_loss = total_loss / max(total_rows, 1)
        valid_loss = evaluate_e1_loss(model, valid_loader, device=device, amp_enabled=amp_enabled)
        scheduler.step()
        improved = valid_loss < best_valid_loss - 1e-12
        if improved:
            best_valid_loss = valid_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "stage": stage_name,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "encoder_learning_rate": optimizer.param_groups[0]["lr"],
                "head_learning_rate": optimizer.param_groups[1]["lr"],
                "global_step": global_step,
                "trainable_encoder_parameters": stage["trainable_encoder_parameters"],
                "trainable_head_parameters": stage["trainable_head_parameters"],
            }
        )
        checkpoint_kwargs = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "epoch": epoch,
            "global_step": global_step,
            "best_valid_loss": best_valid_loss,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "history": history,
            "stage_audits": stage_audits,
            "sampler": train_sampler,
            "dataset_id": dataset_id,
            "protocol_hash": protocol_hash,
            "config_payload": config_payload,
            "initialization_audit": initialization_audit,
        }
        _save_checkpoint(last_checkpoint, **checkpoint_kwargs)
        if improved:
            _save_checkpoint(best_checkpoint, **checkpoint_kwargs)
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            break
        if epoch >= config.training.minimum_epochs and stale_epochs >= config.training.patience:
            break
    required_stages = {"ft0_head_only", "ft1_last_blocks"}
    if stop_after_epoch is None and not required_stages.issubset(stage_audits):
        raise RuntimeError("E2 training did not execute and audit both FT-0 and FT-1")
    if not best_checkpoint.is_file():
        raise RuntimeError("E2 training did not produce best.pt")
    best = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    audit = {
        "device": device,
        "amp_enabled": amp_enabled,
        "precision": "fp16" if amp_enabled else "fp32",
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "epochs_completed": history[-1]["epoch"] if history else start_epoch - 1,
        "global_step": global_step,
        "resumed_from_epoch": resumed_from_epoch,
        "history": history,
        "stage_audits": stage_audits,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "last_checkpoint": str(last_checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "protocol_hash": protocol_hash,
    }
    return model, audit, initialization_audit
