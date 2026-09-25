"""Fold-local self-supervised training and Train-only linear-probe selection."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from facdigger.datasets.sampler import DateSecurityBalancedBatchSampler
from facdigger.datasets.window import (
    FinancePretrainingWindowDataset,
    FinanceTransformerWindowDataset,
)
from facdigger.experiments.manifest import sha256_json
from facdigger.models.finance_patch_transformer import build_finance_transformer_architecture
from facdigger.models.finance_pretrain import FinanceNativePretrainer
from facdigger.training.e1_engine import (
    _restore_rng_state,
    _rng_state,
    _step_optimizer,
    seed_everything,
    select_device,
)
from facdigger.training.finance_pretrain_config import (
    FinancePretrainingExperimentConfig,
)
from facdigger.training.finance_transformer_engine import (
    FINANCE_PRETRAIN_ENCODER_CHECKPOINT,
)
from facdigger.training.ranking import (
    cross_sectional_rank_correlation_loss,
    cross_sectional_rank_targets,
    grouped_rank_ic_audit,
)
from facdigger.training.runtime import (
    TrainingControl,
    checkpoint_copy,
    remaining_loader,
    save_checkpoint,
)

FINANCE_PRETRAIN_RESUME_CHECKPOINT = "finance_patch_pretrain_resume"
FINANCE_PRETRAIN_PROGRESS_CHECKPOINT = "finance_patch_pretrain_progress"
FINANCE_PRETRAIN_OBJECTIVE = "continuous_patch_reconstruction_plus_future_summary"


def build_finance_pretrainer(
    config: FinancePretrainingExperimentConfig,
    *,
    context_length: int,
) -> FinanceNativePretrainer:
    architecture = build_finance_transformer_architecture(
        config.model,
        context_length=context_length,
        num_local_channels=len(config.channels),
        num_market_channels=len(config.market_channels),
        horizons=(5,),
    )
    objective = config.training.objective
    return FinanceNativePretrainer(
        local_encoder=architecture.local_encoder,
        market_encoder=architecture.market_encoder,
        context_length=context_length,
        num_local_channels=len(config.channels),
        num_future_local_channels=7,
        num_market_channels=len(config.market_channels),
        embedding_dim=config.model.embedding_dim,
        reconstruction_weight=objective.reconstruction_weight,
        future_summary_weight=objective.future_summary_weight,
        huber_delta=objective.huber_delta,
        mask_ratio=objective.mask_ratio,
        minimum_span_patches=objective.minimum_span_patches,
        maximum_span_patches=objective.maximum_span_patches,
        shared_channel_fraction=objective.shared_channel_fraction,
    )


def _pretraining_loader(
    dataset: FinancePretrainingWindowDataset,
    *,
    batch_size: int,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, DateSecurityBalancedBatchSampler]:
    sampler = DateSecurityBalancedBatchSampler(
        dataset.asof_dates,
        batch_size=batch_size,
        seed=seed,
    )
    return (
        DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        ),
        sampler,
    )


def _market_batches(
    dataset: FinancePretrainingWindowDataset,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    dates = list(dataset.unique_asof_dates)
    random.Random(seed + epoch).shuffle(dates)
    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for start in range(0, len(dates), batch_size):
        histories = []
        history_masks = []
        futures = []
        future_masks = []
        for asof_date in dates[start : start + batch_size]:
            history, future = dataset.market_pair(asof_date)
            histories.append(history.values)
            history_masks.append(history.observed_mask)
            futures.append(future.values)
            future_masks.append(future.observed_mask)
        batches.append(
            (
                np.stack(histories),
                np.stack(history_masks),
                np.stack(futures),
                np.stack(future_masks),
            )
        )
    return batches


def _pretraining_optimizer(
    model: FinanceNativePretrainer,
    config: FinancePretrainingExperimentConfig,
) -> torch.optim.AdamW:
    decay: list[torch.Tensor] = []
    no_decay: list[torch.Tensor] = []
    for name, parameter in model.named_parameters():
        lower = name.lower()
        target = (
            decay
            if parameter.ndim > 1 and not lower.endswith("bias") and "norm" not in lower
            else no_decay
        )
        target.append(parameter)
    return torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": config.training.weight_decay,
                "group_name": "pretrain_decay",
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
                "group_name": "pretrain_no_decay",
            },
        ],
        lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
    )


def _warmup_cosine(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    minimum_ratio: float,
) -> float:
    if warmup_steps and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1.0 / warmup_steps)
    denominator = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / denominator, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def _apply_pretraining_update(
    output: Any,
    *,
    model: FinanceNativePretrainer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    max_grad_norm: float,
    amp_enabled: bool,
) -> tuple[dict[str, float] | None, float | None]:
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(output.loss).backward()
    gradient_norm, updated = _step_optimizer(
        optimizer,
        scaler,
        model.parameters(),
        max_grad_norm=max_grad_norm,
        amp_enabled=amp_enabled,
        experiment_name="finance_patch_pretrain",
    )
    if not updated:
        return None, None
    if gradient_norm is None:
        raise RuntimeError("pretraining update has no finite gradient norm")
    return (
        {
            "loss": float(output.loss.detach().cpu()),
            "reconstruction_loss": float(output.reconstruction_loss.detach().cpu()),
            "future_summary_loss": float(output.future_summary_loss.detach().cpu()),
            "masked_patch_ratio": output.masked_patch_ratio,
            "masked_observed_elements": float(output.masked_observed_elements),
            "future_summary_elements": float(output.observed_future_summary_elements),
        },
        gradient_norm,
    )


def _extract_local_embeddings(
    model: FinanceNativePretrainer,
    dataset: FinanceTransformerWindowDataset,
    *,
    batch_size: int,
    device: str,
    amp_enabled: bool,
    num_workers: int,
    check_stop: Callable[[], None] | None = None,
) -> torch.Tensor:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    output_dim = model.local_encoder.fusion[-1].normalized_shape[0]
    if not isinstance(output_dim, int):
        raise RuntimeError("local encoder output dimension is not scalar")
    result = torch.empty(len(dataset), output_dim)
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            if check_stop is not None:
                check_stop()
            values = batch["values"].to(device=device, dtype=torch.float32, non_blocking=True)
            observed = batch["observed_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                embeddings = model.local_encoder(values, observed).embedding
            indices = batch["sample_index"].long()
            result[indices] = embeddings.detach().float().cpu()
    model.train(was_training)
    return result


def evaluate_train_only_linear_probe(
    model: FinanceNativePretrainer,
    *,
    fit_dataset: FinanceTransformerWindowDataset,
    selection_dataset: FinanceTransformerWindowDataset,
    config: FinancePretrainingExperimentConfig,
    device: str,
    amp_enabled: bool,
    check_stop: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Select an encoder by complete-date 5-day Rank IC, never outer valid/test."""

    probe = config.training.probe
    fit_embeddings = _extract_local_embeddings(
        model,
        fit_dataset,
        batch_size=config.training.batch_size,
        device=device,
        amp_enabled=amp_enabled,
        num_workers=config.training.num_workers,
        check_stop=check_stop,
    ).to(device)
    selection_embeddings = _extract_local_embeddings(
        model,
        selection_dataset,
        batch_size=config.training.batch_size,
        device=device,
        amp_enabled=amp_enabled,
        num_workers=config.training.num_workers,
        check_stop=check_stop,
    ).to(device)
    fit_ranks = torch.from_numpy(
        cross_sectional_rank_targets(
            fit_dataset.sample_rows["target_5"].to_numpy(),
            fit_dataset.asof_dates,
            minimum_cross_section_size=probe.minimum_cross_section_size,
        )
    ).to(device)
    groups: list[torch.Tensor] = []
    start = 0
    dates = fit_dataset.asof_dates
    while start < len(dates):
        stop = start + 1
        while stop < len(dates) and dates[stop] == dates[start]:
            stop += 1
        groups.append(torch.arange(start, stop, device=device))
        start = stop
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + 10_000)
    fork_devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(config.seed + 10_000)
        if device == "cuda":
            torch.cuda.manual_seed_all(config.seed + 10_000)
        head = torch.nn.Linear(fit_embeddings.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=probe.learning_rate,
        weight_decay=probe.weight_decay,
    )
    head.train()
    losses: list[float] = []
    for _ in range(probe.epochs):
        order = torch.randperm(len(groups), generator=generator).tolist()
        for index in order:
            if check_stop is not None:
                check_stop()
            group = groups[index]
            scores = head(fit_embeddings[group]).squeeze(1)
            loss = cross_sectional_rank_correlation_loss(scores, fit_ranks[group], epsilon=1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    head.eval()
    with torch.inference_mode():
        selection_scores = head(selection_embeddings).squeeze(1).detach().float().cpu().numpy()
    audit = grouped_rank_ic_audit(
        selection_scores,
        selection_dataset.sample_rows["target_5"].to_numpy(),
        selection_dataset.asof_dates,
        minimum_cross_section_size=probe.minimum_cross_section_size,
        minimum_dates=min(probe.selection_dates, len(set(selection_dataset.asof_dates))),
        minimum_coverage=1.0,
    )
    return {
        **audit,
        "fit_rows": len(fit_dataset),
        "fit_dates": len(set(fit_dataset.asof_dates)),
        "selection_rows": len(selection_dataset),
        "selection_dates": len(set(selection_dataset.asof_dates)),
        "probe_epochs": probe.epochs,
        "mean_fit_loss": float(np.mean(losses)),
        "outer_validation_rows_used": 0,
        "outer_test_rows_used": 0,
    }


def _resume_payload(
    *,
    model: FinanceNativePretrainer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    sampler: DateSecurityBalancedBatchSampler,
    epoch: int,
    global_step: int,
    best_probe_rank_ic: float,
    best_epoch: int,
    stale_epochs: int,
    history: list[dict[str, Any]],
    dataset_id: str,
    protocol_hash: str,
) -> dict[str, Any]:
    return {
        "contract": FINANCE_PRETRAIN_RESUME_CHECKPOINT,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "sampler_state": sampler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_probe_rank_ic": best_probe_rank_ic,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "history": history,
        "dataset_id": dataset_id,
        "protocol_hash": protocol_hash,
        "rng_state": _rng_state(),
    }


def _encoder_payload(
    *,
    model: FinanceNativePretrainer,
    epoch: int,
    probe: dict[str, Any],
    dataset_id: str,
    protocol_hash: str,
    config: FinancePretrainingExperimentConfig,
) -> dict[str, Any]:
    return {
        "contract": FINANCE_PRETRAIN_ENCODER_CHECKPOINT,
        "objective": FINANCE_PRETRAIN_OBJECTIVE,
        "local_encoder_state": model.local_encoder.state_dict(),
        "market_encoder_state": model.market_encoder.state_dict(),
        "epoch": epoch,
        "probe_rank_ic": probe["mean_rank_ic"],
        "probe": probe,
        "dataset_id": dataset_id,
        "protocol_hash": protocol_hash,
        "context_length": model.context_length,
        "channels": config.channels,
        "market_channels": config.market_channels,
        "model_config": config.model.model_dump(mode="json"),
    }


def train_finance_pretraining(
    config: FinancePretrainingExperimentConfig,
    *,
    pretraining_dataset: FinancePretrainingWindowDataset,
    probe_fit_dataset: FinanceTransformerWindowDataset,
    probe_selection_dataset: FinanceTransformerWindowDataset,
    dataset_id: str,
    checkpoint_dir: Path,
    resume_from: Path | None = None,
    stop_after_epoch: int | None = None,
    control: TrainingControl | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[FinanceNativePretrainer, dict[str, Any]]:
    started_at = time.perf_counter()
    control = control or TrainingControl()
    if control.enabled and config.training.num_workers:
        raise ValueError("mid-epoch recovery requires num_workers=0")
    seed_everything(config.seed)
    device = select_device(config.training.device)
    amp_enabled = device == "cuda" and config.training.precision == "fp16"
    model = build_finance_pretrainer(config, context_length=pretraining_dataset.context_length).to(
        device
    )
    optimizer = _pretraining_optimizer(model, config)
    loader, sampler = _pretraining_loader(
        pretraining_dataset,
        batch_size=config.training.batch_size,
        seed=config.seed,
        num_workers=config.training.num_workers,
    )
    market_updates = math.ceil(
        len(pretraining_dataset.unique_asof_dates) / config.training.batch_size
    )
    updates_per_epoch = len(loader) + market_updates
    total_steps = updates_per_epoch * config.training.max_epochs
    warmup_steps = math.ceil(total_steps * config.training.warmup_fraction)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _warmup_cosine(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            minimum_ratio=config.training.minimum_learning_rate_ratio,
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    protocol_hash = sha256_json(config.model_dump(mode="json"))
    start_epoch = 1
    global_step = 0
    best_probe_rank_ic = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    resumed_from_epoch: int | None = None
    progress: dict[str, Any] = {}
    best_payload: dict[str, Any] | None = None
    last_checkpoint = checkpoint_dir / "last.pt"
    best_checkpoint = checkpoint_dir / "best_encoder.pt"
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        if checkpoint.get("contract") not in {
            FINANCE_PRETRAIN_RESUME_CHECKPOINT,
            FINANCE_PRETRAIN_PROGRESS_CHECKPOINT,
        }:
            raise ValueError("resume checkpoint is not a finance pretraining checkpoint")
        if checkpoint["dataset_id"] != dataset_id:
            raise ValueError("pretraining resume dataset_id does not match")
        if checkpoint["protocol_hash"] != protocol_hash:
            raise ValueError("pretraining resume protocol does not match")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        sampler.load_state_dict(checkpoint["sampler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_probe_rank_ic = float(checkpoint["best_probe_rank_ic"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        history = list(checkpoint["history"])
        _restore_rng_state(checkpoint["rng_state"])
        resumed_from_epoch = int(checkpoint["epoch"])
        if checkpoint["contract"] == FINANCE_PRETRAIN_PROGRESS_CHECKPOINT:
            progress = checkpoint["progress"]
            if progress["phase"] not in {"local", "market", "probe", "epoch_complete"}:
                raise ValueError("unknown pretraining resume phase")
            if progress["phase"] != "epoch_complete":
                if config.training.num_workers:
                    raise ValueError("mid-epoch recovery requires num_workers=0")
                start_epoch = int(checkpoint["epoch"])
            best_payload = checkpoint["best_checkpoint"]
        elif best_epoch:
            best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
        if best_epoch:
            if (
                best_payload is None
                or best_payload.get("epoch") != best_epoch
                or (
                    best_payload.get("dataset_id") != dataset_id
                    or best_payload.get("protocol_hash") != protocol_hash
                    or best_payload.get("probe_rank_ic") != best_probe_rank_ic
                )
            ):
                raise ValueError("best encoder differs from resume selection state")
            save_checkpoint(best_checkpoint, best_payload)
        if progress.get("finished") or (
            int(checkpoint["epoch"]) >= config.training.minimum_epochs
            and stale_epochs >= config.training.patience
        ):
            start_epoch = config.training.max_epochs + 1

    epoch = int(checkpoint["epoch"]) if resume_from else 1
    phase = progress.get("phase", "local")
    local_cursor = int(progress.get("local_cursor", 0))
    market_cursor = int(progress.get("market_cursor", 0))
    if not 0 <= local_cursor <= len(loader) or not 0 <= market_cursor <= market_updates:
        raise ValueError("pretraining resume cursor is out of bounds")
    local_audits = list(progress.get("local_audits", []))
    market_audits = list(progress.get("market_audits", []))
    gradient_norms = list(progress.get("gradient_norms", []))
    clipped_steps = int(progress.get("clipped_steps", 0))
    skipped_steps = int(progress.get("skipped_steps", 0))
    finished = bool(progress.get("finished", False))

    def persist(*, force: bool = False) -> None:
        if force or control.checkpoint_due():
            state = _resume_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                sampler=sampler,
                epoch=epoch,
                global_step=global_step,
                best_probe_rank_ic=best_probe_rank_ic,
                best_epoch=best_epoch,
                stale_epochs=stale_epochs,
                history=history,
                dataset_id=dataset_id,
                protocol_hash=protocol_hash,
            )
            state.update(
                {
                    "contract": FINANCE_PRETRAIN_PROGRESS_CHECKPOINT,
                    "best_checkpoint": best_payload,
                    "progress": {
                        "phase": phase,
                        "local_cursor": local_cursor,
                        "market_cursor": market_cursor,
                        "finished": finished,
                        "local_audits": local_audits,
                        "market_audits": market_audits,
                        "gradient_norms": gradient_norms,
                        "clipped_steps": clipped_steps,
                        "skipped_steps": skipped_steps,
                    },
                }
            )
            save_checkpoint(last_checkpoint, state)
            control.saved()
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "checkpoint_saved",
                        "epoch": epoch,
                        "phase": phase,
                        "local_cursor": local_cursor,
                        "market_cursor": market_cursor,
                        "global_step": global_step,
                    }
                )
            control.raise_if_stopping(last_checkpoint)

    if resume_from is None and control.enabled:
        persist(force=True)
    elif resume_from is not None:
        control.raise_if_stopping(last_checkpoint)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "pretraining_started",
                "device": device,
                "precision": "fp16" if amp_enabled else "fp32",
                "epochs": config.training.max_epochs,
                "local_updates_per_epoch": len(loader),
                "market_updates_per_epoch": market_updates,
                "optimizer_update_budget": total_steps,
            }
        )
    for epoch in range(start_epoch, config.training.max_epochs + 1):
        epoch_started_at = time.perf_counter()
        sampler.set_epoch(epoch)
        model.train()
        if not progress or progress["phase"] == "epoch_complete":
            local_cursor = market_cursor = 0
            local_audits = []
            market_audits = []
            gradient_norms = []
            clipped_steps = skipped_steps = 0
        progress = {}
        phase = "local"
        for batch in remaining_loader(loader, sampler, local_cursor):
            values = batch["values"].to(device=device, dtype=torch.float32, non_blocking=True)
            observed = batch["observed_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
            future = batch["future_values"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            future_observed = batch["future_observed_mask"].to(
                device=device, dtype=torch.bool, non_blocking=True
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                output = model.forward_local(values, observed, future, future_observed)
            audit, gradient_norm = _apply_pretraining_update(
                output,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                max_grad_norm=config.training.max_grad_norm,
                amp_enabled=amp_enabled,
            )
            if audit is None or gradient_norm is None:
                skipped_steps += 1
            else:
                local_audits.append(audit)
                gradient_norms.append(gradient_norm)
                clipped_steps += int(gradient_norm > config.training.max_grad_norm)
                global_step += 1
                scheduler.step()
                if progress_callback is not None and (global_step == 1 or global_step % 100 == 0):
                    progress_callback(
                        {
                            "event": "pretraining_optimizer_progress",
                            "phase": "local",
                            "epoch": epoch,
                            "global_step": global_step,
                            "optimizer_update_budget": total_steps,
                            "elapsed_seconds": time.perf_counter() - started_at,
                            "latest_loss": audit["loss"],
                        }
                    )

            local_cursor += 1
            persist()

        phase = "market"
        persist(force=control.enabled)
        for values_np, observed_np, future_np, future_observed_np in _market_batches(
            pretraining_dataset,
            batch_size=config.training.batch_size,
            seed=config.seed,
            epoch=epoch,
        )[market_cursor:]:
            values = torch.from_numpy(values_np).to(device=device, dtype=torch.float32)
            observed = torch.from_numpy(observed_np).to(device=device, dtype=torch.bool)
            future = torch.from_numpy(future_np).to(device=device, dtype=torch.float32)
            future_observed = torch.from_numpy(future_observed_np).to(
                device=device, dtype=torch.bool
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                output = model.forward_market(values, observed, future, future_observed)
            audit, gradient_norm = _apply_pretraining_update(
                output,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                max_grad_norm=config.training.max_grad_norm,
                amp_enabled=amp_enabled,
            )
            if audit is None or gradient_norm is None:
                skipped_steps += 1
            else:
                market_audits.append(audit)
                gradient_norms.append(gradient_norm)
                clipped_steps += int(gradient_norm > config.training.max_grad_norm)
                global_step += 1
                scheduler.step()
                if progress_callback is not None and global_step % 100 == 0:
                    progress_callback(
                        {
                            "event": "pretraining_optimizer_progress",
                            "phase": "market",
                            "epoch": epoch,
                            "global_step": global_step,
                            "optimizer_update_budget": total_steps,
                            "elapsed_seconds": time.perf_counter() - started_at,
                            "latest_loss": audit["loss"],
                        }
                    )

            market_cursor += 1
            persist()

        phase = "probe"
        persist(force=control.enabled)
        if not local_audits or not market_audits:
            raise RuntimeError("pretraining epoch produced no successful updates")
        probe = evaluate_train_only_linear_probe(
            model,
            fit_dataset=probe_fit_dataset,
            selection_dataset=probe_selection_dataset,
            config=config,
            device=device,
            amp_enabled=amp_enabled,
            check_stop=lambda: control.raise_if_stopping(last_checkpoint),
        )
        control.raise_if_stopping(last_checkpoint)
        probe_rank_ic = float(probe["mean_rank_ic"])
        improved = probe_rank_ic > best_probe_rank_ic + 1e-12
        if improved:
            best_probe_rank_ic = probe_rank_ic
            best_epoch = epoch
            stale_epochs = 0
            best_payload = checkpoint_copy(
                _encoder_payload(
                    model=model,
                    epoch=epoch,
                    probe=probe,
                    dataset_id=dataset_id,
                    protocol_hash=protocol_hash,
                    config=config,
                )
            )
        else:
            stale_epochs += 1

        def means(audits: list[dict[str, float]]) -> dict[str, float]:
            return {key: float(np.mean([audit[key] for audit in audits])) for key in audits[0]}

        history.append(
            {
                "epoch": epoch,
                "local": means(local_audits),
                "market": means(market_audits),
                "local_updates": len(local_audits),
                "market_updates": len(market_audits),
                "mean_pre_clip_gradient_norm": float(np.mean(gradient_norms)),
                "p95_pre_clip_gradient_norm": float(np.percentile(gradient_norms, 95)),
                "gradient_clip_ratio": clipped_steps / len(gradient_norms),
                "amp_skipped_optimizer_steps": skipped_steps,
                "probe": probe,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "global_step": global_step,
                "epoch_elapsed_seconds": time.perf_counter() - epoch_started_at,
            }
        )
        phase = "epoch_complete"
        finished = epoch == config.training.max_epochs or (
            epoch >= config.training.minimum_epochs and stale_epochs >= config.training.patience
        )
        persist(force=True)
        if best_payload is not None:
            save_checkpoint(best_checkpoint, best_payload)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "pretraining_epoch_completed",
                    "epoch": epoch,
                    "global_step": global_step,
                    "elapsed_seconds": time.perf_counter() - started_at,
                    "epoch_elapsed_seconds": history[-1]["epoch_elapsed_seconds"],
                    "probe_rank_ic": probe_rank_ic,
                    "best_probe_rank_ic": best_probe_rank_ic,
                    "best_epoch": best_epoch,
                }
            )
        control.raise_if_stopping(last_checkpoint)
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            break
        if epoch >= config.training.minimum_epochs and stale_epochs >= config.training.patience:
            break

    if not best_checkpoint.is_file():
        raise RuntimeError("finance pretraining did not produce best_encoder.pt")
    best = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.local_encoder.load_state_dict(best["local_encoder_state"])
    model.market_encoder.load_state_dict(best["market_encoder_state"])
    return model, {
        "device": device,
        "precision": "fp16" if amp_enabled else "fp32",
        "objective": FINANCE_PRETRAIN_OBJECTIVE,
        "best_epoch": best_epoch,
        "best_probe_rank_ic": best_probe_rank_ic,
        "epochs_completed": history[-1]["epoch"] if history else start_epoch - 1,
        "global_step": global_step,
        "resumed_from_epoch": resumed_from_epoch,
        "history": history,
        "pretraining_rows": len(pretraining_dataset),
        "pretraining_dates": len(pretraining_dataset.unique_asof_dates),
        "probe_fit_rows": len(probe_fit_dataset),
        "probe_selection_rows": len(probe_selection_dataset),
        "outer_validation_rows_used": 0,
        "outer_test_rows_used": 0,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.perf_counter() - started_at,
    }


def benchmark_finance_pretraining_updates(
    config: FinancePretrainingExperimentConfig,
    *,
    pretraining_dataset: FinancePretrainingWindowDataset,
    dataset_id: str,
    local_optimizer_updates: int = 100,
    market_optimizer_updates: int = 20,
    warmup_updates: int = 10,
) -> dict[str, Any]:
    """Measure the two self-supervised update types without fitting a probe."""

    if local_optimizer_updates < 2 or market_optimizer_updates < 1:
        raise ValueError("pretraining benchmark update counts must be positive")
    if not 0 <= warmup_updates < local_optimizer_updates:
        raise ValueError("benchmark warmup_updates must be below local updates")
    seed_everything(config.seed)
    device = select_device(config.training.device)
    amp_enabled = device == "cuda" and config.training.precision == "fp16"
    model = build_finance_pretrainer(config, context_length=pretraining_dataset.context_length).to(
        device
    )
    optimizer = _pretraining_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loader, sampler = _pretraining_loader(
        pretraining_dataset,
        batch_size=config.training.batch_size,
        seed=config.seed,
        num_workers=config.training.num_workers,
    )
    market_updates_per_epoch = math.ceil(
        len(pretraining_dataset.unique_asof_dates) / config.training.batch_size
    )
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    local_timings: list[float] = []
    local_rows = 0
    local_updates = 0
    epoch = 0
    while local_updates < local_optimizer_updates:
        epoch += 1
        sampler.set_epoch(epoch)
        for batch in loader:
            started = time.perf_counter()
            values = batch["values"].to(device=device, dtype=torch.float32, non_blocking=True)
            observed = batch["observed_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
            future = batch["future_values"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            future_observed = batch["future_observed_mask"].to(
                device=device, dtype=torch.bool, non_blocking=True
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                output = model.forward_local(values, observed, future, future_observed)
            audit, gradient_norm = _apply_pretraining_update(
                output,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                max_grad_norm=config.training.max_grad_norm,
                amp_enabled=amp_enabled,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            if audit is None or gradient_norm is None:
                continue
            local_updates += 1
            local_rows += int(batch["sample_index"].numel())
            local_timings.append(time.perf_counter() - started)
            if local_updates >= local_optimizer_updates:
                break

    market_timings: list[float] = []
    available_market_batches = _market_batches(
        pretraining_dataset,
        batch_size=config.training.batch_size,
        seed=config.seed,
        epoch=1,
    )
    for batch_index in range(market_optimizer_updates):
        values_np, observed_np, future_np, future_observed_np = available_market_batches[
            batch_index % len(available_market_batches)
        ]
        started = time.perf_counter()
        values = torch.from_numpy(values_np).to(device=device, dtype=torch.float32)
        observed = torch.from_numpy(observed_np).to(device=device, dtype=torch.bool)
        future = torch.from_numpy(future_np).to(device=device, dtype=torch.float32)
        future_observed = torch.from_numpy(future_observed_np).to(device=device, dtype=torch.bool)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            output = model.forward_market(values, observed, future, future_observed)
        audit, gradient_norm = _apply_pretraining_update(
            output,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            max_grad_norm=config.training.max_grad_norm,
            amp_enabled=amp_enabled,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        if audit is None or gradient_norm is None:
            raise RuntimeError("market benchmark update was rejected")
        market_timings.append(time.perf_counter() - started)

    measured_local = local_timings[warmup_updates:]
    local_seconds = float(np.mean(measured_local))
    market_seconds = float(np.mean(market_timings))
    projected_epoch_seconds = (
        local_seconds * len(loader) + market_seconds * market_updates_per_epoch
    )
    return {
        "kind": "finance_native_pretraining_updates",
        "dataset_id": dataset_id,
        "device": device,
        "precision": "fp16" if amp_enabled else "fp32",
        "local_updates": local_updates,
        "market_updates": len(market_timings),
        "warmup_local_updates_excluded": warmup_updates,
        "local_rows_processed": local_rows,
        "mean_local_seconds_per_update": local_seconds,
        "p95_local_seconds_per_update": float(np.percentile(measured_local, 95)),
        "mean_market_seconds_per_update": market_seconds,
        "local_updates_per_epoch": len(loader),
        "market_updates_per_epoch": market_updates_per_epoch,
        "projected_epoch_hours": projected_epoch_seconds / 3600.0,
        "projected_run_hours": (projected_epoch_seconds * config.training.max_epochs / 3600.0),
        "probe_time_not_included": True,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
        ),
        "cuda_peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if device == "cuda" else None
        ),
        "full_data_training_unchanged": True,
        "benchmark_only_stops_early": True,
    }
