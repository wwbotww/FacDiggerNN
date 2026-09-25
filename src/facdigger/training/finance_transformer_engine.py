"""Full-date finance Transformer training with exact embedding replay."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from facdigger.datasets.sampler import FullDateBatchSampler
from facdigger.datasets.window import FinanceTransformerWindowDataset
from facdigger.experiments.manifest import sha256_json
from facdigger.models.finance_patch_transformer import (
    FinancePatchTransformer,
    build_finance_transformer_model,
)
from facdigger.models.finance_scoring import (
    _device_microbatches,
    _full_date_loader,
    _market_tensors,
    predict_finance_transformer,
)
from facdigger.training.e1_engine import (
    _dates_in_current_optimizer_step,
    _restore_rng_state,
    _rng_state,
    _step_optimizer,
    seed_everything,
    select_device,
)
from facdigger.training.finance_transformer_config import (
    FinanceTransformerExperimentConfig,
)
from facdigger.training.ranking import (
    TARGET_TRANSFORM,
    average_ranks,
    cross_sectional_rank_correlation_loss,
    cross_sectional_rank_targets,
)
from facdigger.training.runtime import (
    TrainingControl,
    checkpoint_copy,
    remaining_loader,
    save_checkpoint,
)

FINANCE_TRANSFORMER_OBJECTIVE = "multi_horizon_full_date_rank_correlation"
FINANCE_TRANSFORMER_CHECKPOINT = "finance_patch_transformer_checkpoint"
FINANCE_TRANSFORMER_RESUME = "finance_patch_transformer_training_resume"
FINANCE_PRETRAIN_ENCODER_CHECKPOINT = "finance_patch_pretrain_encoder"


def load_finance_pretrained_encoders(
    model: FinancePatchTransformer,
    checkpoint_path: Path,
    *,
    expected_dataset_id: str | None = None,
    expected_context_length: int | None = None,
    expected_channels: list[str] | None = None,
    expected_market_channels: list[str] | None = None,
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"financial pretraining checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != FINANCE_PRETRAIN_ENCODER_CHECKPOINT:
        raise ValueError("checkpoint is not a finance-native pretraining encoder")
    expectations = {
        "dataset_id": expected_dataset_id,
        "context_length": expected_context_length,
        "channels": expected_channels,
        "market_channels": expected_market_channels,
    }
    for field, expected in expectations.items():
        if expected is not None and checkpoint.get(field) != expected:
            raise ValueError(f"pretraining checkpoint {field} does not match this run")
    model.local_encoder.load_state_dict(checkpoint["local_encoder_state"], strict=True)
    model.market_encoder.load_state_dict(checkpoint["market_encoder_state"], strict=True)
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "pretraining_dataset_id": checkpoint.get("dataset_id"),
        "pretraining_epoch": checkpoint.get("epoch"),
        "probe_rank_ic": checkpoint.get("probe_rank_ic"),
    }


def _multi_horizon_loss(
    scores: torch.Tensor,
    target_ranks: torch.Tensor,
    *,
    horizons: tuple[int, ...],
    horizon_weights: dict[int, float],
    epsilon: float,
    scale_regularization: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if scores.shape != target_ranks.shape or scores.ndim != 2:
        raise ValueError("scores and target ranks must share [N,H] shape")
    total = scores.new_zeros((), dtype=torch.float32)
    audit: dict[str, float] = {}
    scale_penalty = scores.new_zeros((), dtype=torch.float32)
    for column, horizon in enumerate(horizons):
        weight = horizon_weights[horizon]
        horizon_loss = cross_sectional_rank_correlation_loss(
            scores[:, column], target_ranks[:, column], epsilon=epsilon
        )
        score_std = scores[:, column].float().std(unbiased=False)
        total = total + weight * horizon_loss
        scale_penalty = scale_penalty + weight * torch.log(score_std + epsilon).square()
        audit[f"rank_loss_{horizon}"] = float(horizon_loss.detach().cpu())
        audit[f"score_std_{horizon}"] = float(score_std.detach().cpu())
    total = total + scale_regularization * scale_penalty
    audit["scale_penalty"] = float(scale_penalty.detach().cpu())
    audit["loss"] = float(total.detach().cpu())
    return total, audit


def _replay_matches(
    expected: torch.Tensor,
    replayed: torch.Tensor,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    name: str,
) -> float:
    expected_float = expected.detach().float()
    replayed_float = replayed.detach().float()
    maximum_error = float((expected_float - replayed_float).abs().max().cpu())
    if not torch.allclose(
        expected_float,
        replayed_float,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    ):
        raise RuntimeError(
            f"{name} embedding replay differs from first pass; max_abs={maximum_error}"
        )
    return maximum_error


def backward_complete_date_with_embedding_replay(
    model: FinancePatchTransformer,
    dataset: FinanceTransformerWindowDataset,
    full_date_batch: dict[str, torch.Tensor],
    *,
    device: str,
    target_rank_lookup: torch.Tensor,
    horizon_weights: dict[int, float],
    epsilon: float,
    scale_regularization: float,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler,
    physical_microbatch_size: int,
    dates_in_optimizer_step: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    """Backpropagate one exact full-date hierarchy with bounded temporal graphs."""

    microbatches = _device_microbatches(full_date_batch, batch_size=physical_microbatch_size)
    local_rng_states: list[dict[str, Any]] = []
    detached_local: list[torch.Tensor] = []
    with torch.no_grad():
        for microbatch in microbatches:
            local_rng_states.append(_rng_state())
            values = microbatch["values"].to(device=device, dtype=torch.float32, non_blocking=True)
            observed = microbatch["observed_mask"].to(
                device=device, dtype=torch.bool, non_blocking=True
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                local = model.encode_local(values, observed).embedding
            detached_local.append(local.detach().float())
        market_values, market_observed = _market_tensors(
            dataset, full_date_batch["sample_index"], device=device
        )
        market_rng_state = _rng_state()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            detached_market = model.encode_market(market_values, market_observed).detach().float()

    local_leaf = torch.cat(detached_local, dim=0).requires_grad_(True)
    market_leaf = detached_market.requires_grad_(True)
    indices = full_date_batch["sample_index"].long()
    target_ranks = target_rank_lookup[indices].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
        date_output = model.score_date(local_leaf, market_leaf)
    loss, loss_audit = _multi_horizon_loss(
        date_output.scores,
        target_ranks,
        horizons=model.horizons,
        horizon_weights=horizon_weights,
        epsilon=epsilon,
        scale_regularization=scale_regularization,
    )
    scaler.scale(loss / dates_in_optimizer_step).backward()
    if local_leaf.grad is None or market_leaf.grad is None:
        raise RuntimeError("cross-sectional graph did not return embedding gradients")
    local_gradients = local_leaf.grad.detach().clone()
    market_gradient = market_leaf.grad.detach().clone()
    post_cross_rng_state = _rng_state()

    maximum_local_replay_error = 0.0
    start = 0
    for microbatch, rng_state, expected in zip(
        microbatches, local_rng_states, detached_local, strict=True
    ):
        _restore_rng_state(rng_state)
        values = microbatch["values"].to(device=device, dtype=torch.float32, non_blocking=True)
        observed = microbatch["observed_mask"].to(
            device=device, dtype=torch.bool, non_blocking=True
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            replayed = model.encode_local(values, observed).embedding
        maximum_local_replay_error = max(
            maximum_local_replay_error,
            _replay_matches(
                expected,
                replayed,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                name="local",
            ),
        )
        stop = start + replayed.shape[0]
        torch.autograd.backward(
            replayed,
            grad_tensors=local_gradients[start:stop].to(replayed.dtype),
        )
        start = stop

    _restore_rng_state(market_rng_state)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
        replayed_market = model.encode_market(market_values, market_observed)
    market_replay_error = _replay_matches(
        detached_market,
        replayed_market,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        name="market",
    )
    torch.autograd.backward(
        replayed_market,
        grad_tensors=market_gradient.to(replayed_market.dtype),
    )
    _restore_rng_state(post_cross_rng_state)

    return {
        **loss_audit,
        "rows": int(local_leaf.shape[0]),
        "physical_microbatches": len(microbatches),
        "maximum_local_replay_error": maximum_local_replay_error,
        "market_replay_error": market_replay_error,
        "context_gate": date_output.context_gate.detach().float().cpu().tolist(),
    }


def _daily_rank_ics(
    scores: np.ndarray,
    targets: np.ndarray,
    dates: list[Any],
    *,
    minimum_cross_section_size: int,
) -> np.ndarray:
    result: list[float] = []
    start = 0
    while start < len(dates):
        stop = start + 1
        while stop < len(dates) and dates[stop] == dates[start]:
            stop += 1
        if stop - start < minimum_cross_section_size:
            start = stop
            continue
        score_ranks = average_ranks(scores[start:stop])
        target_ranks = average_ranks(targets[start:stop])
        correlation = float(np.corrcoef(score_ranks, target_ranks)[0, 1])
        if math.isfinite(correlation):
            result.append(correlation)
        start = stop
    return np.asarray(result, dtype=np.float64)


def evaluate_finance_transformer_selection(
    model: FinancePatchTransformer,
    dataset: FinanceTransformerWindowDataset,
    *,
    batch_size: int,
    device: str,
    precision: str,
    num_workers: int,
    minimum_cross_section_size: int,
    minimum_dates: int,
    minimum_coverage: float,
    subperiods: int,
    stability_penalty: float,
    check_stop: Callable[[], None] | None = None,
) -> dict[str, Any]:
    scores = predict_finance_transformer(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
        precision=precision,
        num_workers=num_workers,
        check_stop=check_stop,
    )
    targets = dataset.sample_rows["target"].to_numpy()
    daily = _daily_rank_ics(
        scores,
        targets,
        dataset.asof_dates,
        minimum_cross_section_size=minimum_cross_section_size,
    )
    total_dates = len(dataset.sample_rows["asof_date"].unique())
    coverage = len(daily) / total_dates if total_dates else 0.0
    if len(daily) < minimum_dates:
        raise ValueError(f"selection has too few valid dates: {len(daily)} < {minimum_dates}")
    if coverage < minimum_coverage:
        raise ValueError(
            f"selection Rank IC coverage is below threshold: {coverage:.6f} < "
            f"{minimum_coverage:.6f}"
        )
    if len(daily) < subperiods:
        raise ValueError("selection has fewer valid dates than requested subperiods")
    subperiod_means = [float(part.mean()) for part in np.array_split(daily, subperiods)]
    mean_rank_ic = float(daily.mean())
    stability_std = float(np.std(subperiod_means))
    return {
        "mean_rank_ic": mean_rank_ic,
        "selection_score": mean_rank_ic - stability_penalty * stability_std,
        "subperiod_mean_rank_ic": subperiod_means,
        "subperiod_std": stability_std,
        "valid_dates": len(daily),
        "skipped_dates": total_dates - len(daily),
        "date_coverage": coverage,
        "score_std": float(np.std(scores)),
    }


def _decay_parameter(name: str, parameter: torch.Tensor) -> bool:
    lower = name.lower()
    return parameter.ndim > 1 and not lower.endswith("bias") and "norm" not in lower


def _optimizer_parameter_groups(
    model: FinancePatchTransformer,
    *,
    encoder_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    encoder_ids = {
        id(parameter)
        for module in (
            model.local_encoder.backbone,
            model.market_encoder.backbone,
        )
        for parameter in module.parameters()
    }
    buckets: dict[tuple[str, bool], list[torch.Tensor]] = {
        ("encoder", True): [],
        ("encoder", False): [],
        ("head", True): [],
        ("head", False): [],
    }
    for name, parameter in model.named_parameters():
        family = "encoder" if id(parameter) in encoder_ids else "head"
        buckets[(family, _decay_parameter(name, parameter))].append(parameter)
    groups: list[dict[str, Any]] = []
    for (family, decay), parameters in buckets.items():
        if not parameters:
            continue
        groups.append(
            {
                "params": parameters,
                "lr": (encoder_learning_rate if family == "encoder" else head_learning_rate),
                "weight_decay": weight_decay if decay else 0.0,
                "group_name": f"{family}_{'decay' if decay else 'no_decay'}",
            }
        )
    if sum(len(group["params"]) for group in groups) != len(list(model.parameters())):
        raise RuntimeError("optimizer parameter grouping lost or duplicated parameters")
    return groups


def _warmup_cosine_lambda(
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


def _checkpoint_payload(
    *,
    model: FinancePatchTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_selection_score: float,
    best_selection_audit: dict[str, Any],
    best_epoch: int,
    stale_epochs: int,
    history: list[dict[str, Any]],
    sampler: FullDateBatchSampler,
    dataset_id: str,
    protocol_hash: str,
    config_payload: dict[str, Any],
    initialization_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 4,
        "contract": FINANCE_TRANSFORMER_CHECKPOINT,
        "model_type": "finance_patch_transformer",
        "objective": FINANCE_TRANSFORMER_OBJECTIVE,
        "target_transform": TARGET_TRANSFORM,
        "optimization_protocol": {
            "unit": "complete_date",
            "method": "exact_leaf_embedding_replay",
            "physical_microbatch_size": config_payload["training"]["batch_size"],
            "dates_per_optimizer_step": config_payload["training"]["dates_per_optimizer_step"],
        },
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_selection_score": best_selection_score,
        "best_selection_audit": best_selection_audit,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "history": history,
        "sampler_state": sampler.state_dict(),
        "rng_state": _rng_state(),
        "dataset_id": dataset_id,
        "protocol_hash": protocol_hash,
        "config": config_payload,
        "initialization": initialization_audit,
    }


def train_finance_transformer(
    config: FinanceTransformerExperimentConfig,
    *,
    train_dataset: FinanceTransformerWindowDataset,
    valid_dataset: FinanceTransformerWindowDataset,
    dataset_id: str,
    checkpoint_dir: Path,
    resume_from: Path | None = None,
    stop_after_epoch: int | None = None,
    control: TrainingControl | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[FinancePatchTransformer, dict[str, Any]]:
    started_at = time.perf_counter()
    control = control or TrainingControl()
    if control.enabled and config.training.num_workers:
        raise ValueError("mid-epoch recovery requires num_workers=0")
    seed_everything(config.seed)
    device = select_device(config.training.device)
    amp_enabled = device == "cuda" and config.training.precision == "fp16"
    model = build_finance_transformer_model(config, context_length=train_dataset.context_length)
    initialization_audit: dict[str, Any] = {"method": config.initialization}
    if config.initialization == "finance_pretrained" and resume_from is None:
        assert config.pretrained_checkpoint is not None
        initialization_audit.update(
            load_finance_pretrained_encoders(
                model,
                config.pretrained_checkpoint,
                expected_dataset_id=dataset_id,
                expected_context_length=train_dataset.context_length,
                expected_channels=config.channels,
                expected_market_channels=config.market_channels,
            )
        )
    model = model.to(device)
    groups = _optimizer_parameter_groups(
        model,
        encoder_learning_rate=config.training.encoder_learning_rate,
        head_learning_rate=config.training.head_learning_rate,
        weight_decay=config.training.weight_decay,
    )
    optimizer = torch.optim.AdamW(
        groups,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
    )
    train_loader, train_sampler = _full_date_loader(
        train_dataset,
        shuffle=True,
        seed=config.seed,
        num_workers=config.training.num_workers,
        minimum_group_size=config.training.objective.minimum_cross_section_size,
    )
    updates_per_epoch = math.ceil(len(train_loader) / config.training.dates_per_optimizer_step)
    total_update_budget = updates_per_epoch * config.training.max_epochs
    warmup_steps = math.ceil(total_update_budget * config.training.warmup_fraction)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _warmup_cosine_lambda(
            step,
            total_steps=total_update_budget,
            warmup_steps=warmup_steps,
            minimum_ratio=config.training.minimum_learning_rate_ratio,
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    target_rank_lookup = torch.stack(
        [
            torch.from_numpy(
                cross_sectional_rank_targets(
                    train_dataset.sample_rows[f"target_{horizon}"].to_numpy(),
                    train_dataset.asof_dates,
                    minimum_cross_section_size=(
                        config.training.objective.minimum_cross_section_size
                    ),
                )
            )
            for horizon in config.horizons
        ],
        dim=1,
    )
    config_payload = config.model_dump(mode="json")
    protocol_hash = sha256_json(config_payload)
    start_epoch = 1
    global_step = 0
    best_selection_score = float("-inf")
    best_selection_audit: dict[str, Any] = {}
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    resumed_from_epoch: int | None = None
    progress: dict[str, Any] = {}
    best_payload: dict[str, Any] | None = None
    last_checkpoint = checkpoint_dir / "last.pt"
    best_checkpoint = checkpoint_dir / "best.pt"
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        if checkpoint.get("contract") not in {
            FINANCE_TRANSFORMER_CHECKPOINT,
            FINANCE_TRANSFORMER_RESUME,
        }:
            raise ValueError("resume checkpoint is not a finance Transformer checkpoint")
        if checkpoint["dataset_id"] != dataset_id:
            raise ValueError("resume checkpoint dataset_id does not match")
        if checkpoint["protocol_hash"] != protocol_hash:
            raise ValueError("resume checkpoint training protocol does not match")
        initialization_audit = dict(checkpoint["initialization"])
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_selection_score = float(checkpoint["best_selection_score"])
        best_selection_audit = dict(checkpoint["best_selection_audit"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        history = list(checkpoint["history"])
        train_sampler.load_state_dict(checkpoint["sampler_state"])
        _restore_rng_state(checkpoint["rng_state"])
        resumed_from_epoch = int(checkpoint["epoch"])
        if checkpoint["contract"] == FINANCE_TRANSFORMER_RESUME:
            progress = checkpoint["progress"]
            if progress["phase"] not in {"train", "selection", "epoch_complete"}:
                raise ValueError("unknown supervised resume phase")
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
                    or best_payload.get("best_selection_score") != best_selection_score
                )
            ):
                raise ValueError("best checkpoint differs from resume selection state")
            save_checkpoint(best_checkpoint, best_payload)
        if progress.get("finished") or (
            int(checkpoint["epoch"]) >= config.training.minimum_epochs
            and stale_epochs >= config.training.patience
        ):
            start_epoch = config.training.max_epochs + 1

    epoch = int(checkpoint["epoch"]) if resume_from else 1
    phase = progress.get("phase", "train")
    cursor = int(progress.get("cursor", 0))
    if (
        cursor < 0
        or cursor > len(train_loader)
        or (
            phase == "train"
            and cursor != len(train_loader)
            and cursor % config.training.dates_per_optimizer_step
        )
    ):
        raise ValueError("resume cursor is not a complete optimizer boundary")
    date_audits = list(progress.get("date_audits", []))
    gradient_norms = list(progress.get("gradient_norms", []))
    clipped_steps = int(progress.get("clipped_steps", 0))
    amp_skipped_optimizer_steps = int(progress.get("amp_skipped_optimizer_steps", 0))
    finished = bool(progress.get("finished", False))

    def payload() -> dict[str, Any]:
        return _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_selection_score=best_selection_score,
            best_selection_audit=best_selection_audit,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
            history=history,
            sampler=train_sampler,
            dataset_id=dataset_id,
            protocol_hash=protocol_hash,
            config_payload=config_payload,
            initialization_audit=initialization_audit,
        )

    def persist(*, force: bool = False) -> None:
        if force or control.checkpoint_due():
            state = payload()
            state.update(
                {
                    "contract": FINANCE_TRANSFORMER_RESUME,
                    "best_checkpoint": best_payload,
                    "progress": {
                        "phase": phase,
                        "cursor": cursor,
                        "finished": finished,
                        "date_audits": date_audits,
                        "gradient_norms": gradient_norms,
                        "clipped_steps": clipped_steps,
                        "amp_skipped_optimizer_steps": amp_skipped_optimizer_steps,
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
                        "cursor": cursor,
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
                "event": "training_started",
                "device": device,
                "precision": "fp16" if amp_enabled else "fp32",
                "epochs": config.training.max_epochs,
                "dates_per_epoch": len(train_loader),
                "optimizer_update_budget": total_update_budget,
            }
        )
    for epoch in range(start_epoch, config.training.max_epochs + 1):
        epoch_started_at = time.perf_counter()
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if not progress or progress["phase"] == "epoch_complete":
            cursor = 0
            date_audits = []
            gradient_norms = []
            clipped_steps = 0
            amp_skipped_optimizer_steps = 0
        progress = {}
        phase = "train"
        for date_index, batch in enumerate(
            remaining_loader(train_loader, train_sampler, cursor), start=cursor + 1
        ):
            dates_in_step = _dates_in_current_optimizer_step(
                date_index,
                total_dates=len(train_loader),
                configured_dates=config.training.dates_per_optimizer_step,
            )
            date_audits.append(
                backward_complete_date_with_embedding_replay(
                    model,
                    train_dataset,
                    batch,
                    device=device,
                    target_rank_lookup=target_rank_lookup,
                    horizon_weights=config.training.objective.horizon_weights,
                    epsilon=config.training.objective.epsilon,
                    scale_regularization=(config.training.objective.scale_regularization),
                    amp_enabled=amp_enabled,
                    scaler=scaler,
                    physical_microbatch_size=config.training.batch_size,
                    dates_in_optimizer_step=dates_in_step,
                    relative_tolerance=config.training.replay_relative_tolerance,
                    absolute_tolerance=config.training.replay_absolute_tolerance,
                )
            )
            should_step = (
                date_index % config.training.dates_per_optimizer_step == 0
                or date_index == len(train_loader)
            )
            if not should_step:
                continue
            gradient_norm, optimizer_updated = _step_optimizer(
                optimizer,
                scaler,
                model.parameters(),
                max_grad_norm=config.training.max_grad_norm,
                amp_enabled=amp_enabled,
                experiment_name="finance_patch_transformer",
            )
            cursor = date_index
            if not optimizer_updated:
                amp_skipped_optimizer_steps += 1
                persist()
                continue
            if gradient_norm is None:
                raise RuntimeError("optimizer update has no finite gradient norm")
            gradient_norms.append(gradient_norm)
            clipped_steps += int(gradient_norm > config.training.max_grad_norm)
            global_step += 1
            scheduler.step()
            if progress_callback is not None and (global_step == 1 or global_step % 25 == 0):
                progress_callback(
                    {
                        "event": "optimizer_progress",
                        "epoch": epoch,
                        "date": date_index,
                        "dates_in_epoch": len(train_loader),
                        "global_step": global_step,
                        "optimizer_update_budget": total_update_budget,
                        "elapsed_seconds": time.perf_counter() - started_at,
                        "latest_loss": date_audits[-1]["loss"],
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    }
                )

            persist()

        phase = "selection"
        persist(force=control.enabled)
        selection = evaluate_finance_transformer_selection(
            model,
            valid_dataset,
            batch_size=config.training.batch_size,
            device=device,
            precision=config.training.precision,
            num_workers=config.training.num_workers,
            check_stop=lambda: control.raise_if_stopping(last_checkpoint),
            minimum_cross_section_size=(config.training.objective.minimum_cross_section_size),
            minimum_dates=config.training.objective.minimum_selection_dates,
            minimum_coverage=config.training.objective.minimum_selection_coverage,
            subperiods=config.training.objective.selection_subperiods,
            stability_penalty=(config.training.objective.selection_stability_penalty),
        )
        control.raise_if_stopping(last_checkpoint)
        selection_score = float(selection["selection_score"])
        improved = selection_score > best_selection_score + 1e-12
        if improved:
            best_selection_score = selection_score
            best_selection_audit = selection
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        horizon_audit = {
            key: float(np.mean([audit[key] for audit in date_audits]))
            for key in date_audits[0]
            if key.startswith("rank_loss_") or key.startswith("score_std_")
        }
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean([audit["loss"] for audit in date_audits])),
                "train_scale_penalty": float(
                    np.mean([audit["scale_penalty"] for audit in date_audits])
                ),
                **horizon_audit,
                "train_dates": len(date_audits),
                "physical_microbatches": sum(
                    int(audit["physical_microbatches"]) for audit in date_audits
                ),
                "maximum_local_replay_error": max(
                    float(audit["maximum_local_replay_error"]) for audit in date_audits
                ),
                "maximum_market_replay_error": max(
                    float(audit["market_replay_error"]) for audit in date_audits
                ),
                "mean_pre_clip_gradient_norm": (
                    float(np.mean(gradient_norms)) if gradient_norms else None
                ),
                "p95_pre_clip_gradient_norm": (
                    float(np.percentile(gradient_norms, 95)) if gradient_norms else None
                ),
                "max_pre_clip_gradient_norm": (
                    float(np.max(gradient_norms)) if gradient_norms else None
                ),
                "gradient_clip_ratio": (
                    clipped_steps / len(gradient_norms) if gradient_norms else None
                ),
                "amp_skipped_optimizer_steps": amp_skipped_optimizer_steps,
                "selection": selection,
                "learning_rates": {
                    str(group["group_name"]): float(group["lr"]) for group in optimizer.param_groups
                },
                "context_gate": date_audits[-1]["context_gate"],
                "global_step": global_step,
                "epoch_elapsed_seconds": time.perf_counter() - epoch_started_at,
            }
        )
        if improved:
            best_payload = checkpoint_copy(payload())
        phase = "epoch_complete"
        finished = epoch == config.training.max_epochs or (
            epoch >= config.training.minimum_epochs and stale_epochs >= config.training.patience
        )
        persist(force=True)
        if improved:
            save_checkpoint(best_checkpoint, best_payload)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "epoch_completed",
                    "epoch": epoch,
                    "global_step": global_step,
                    "elapsed_seconds": time.perf_counter() - started_at,
                    "epoch_elapsed_seconds": history[-1]["epoch_elapsed_seconds"],
                    "selection_score": selection_score,
                    "best_selection_score": best_selection_score,
                    "best_epoch": best_epoch,
                }
            )
        control.raise_if_stopping(last_checkpoint)
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            break
        if epoch >= config.training.minimum_epochs and stale_epochs >= config.training.patience:
            break

    if not best_checkpoint.is_file():
        raise RuntimeError("finance Transformer training did not produce best.pt")
    best = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    return model, {
        "device": device,
        "amp_enabled": amp_enabled,
        "precision": "fp16" if amp_enabled else "fp32",
        "objective": FINANCE_TRANSFORMER_OBJECTIVE,
        "target_transform": TARGET_TRANSFORM,
        "optimization_protocol": {
            "unit": "complete_date",
            "method": "exact_leaf_embedding_replay",
            "physical_microbatch_size": config.training.batch_size,
            "dates_per_optimizer_step": config.training.dates_per_optimizer_step,
            "total_update_budget": total_update_budget,
            "warmup_steps": warmup_steps,
        },
        "initialization": initialization_audit,
        "best_epoch": best_epoch,
        "best_selection_score": best_selection_score,
        "best_selection_audit": best_selection_audit,
        "epochs_completed": history[-1]["epoch"] if history else start_epoch - 1,
        "global_step": global_step,
        "resumed_from_epoch": resumed_from_epoch,
        "history": history,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.perf_counter() - started_at,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "last_checkpoint": str(last_checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "protocol_hash": protocol_hash,
    }


def benchmark_finance_transformer_updates(
    config: FinanceTransformerExperimentConfig,
    *,
    train_dataset: FinanceTransformerWindowDataset,
    dataset_id: str,
    optimizer_updates: int = 100,
    warmup_updates: int = 10,
) -> dict[str, Any]:
    """Measure real complete-date optimizer updates without selecting a checkpoint."""

    if optimizer_updates < 2:
        raise ValueError("benchmark requires at least two optimizer updates")
    if not 0 <= warmup_updates < optimizer_updates:
        raise ValueError("benchmark warmup_updates must be below optimizer_updates")
    seed_everything(config.seed)
    device = select_device(config.training.device)
    amp_enabled = device == "cuda" and config.training.precision == "fp16"
    model = build_finance_transformer_model(config, context_length=train_dataset.context_length)
    if config.initialization == "finance_pretrained":
        assert config.pretrained_checkpoint is not None
        load_finance_pretrained_encoders(
            model,
            config.pretrained_checkpoint,
            expected_dataset_id=dataset_id,
            expected_context_length=train_dataset.context_length,
            expected_channels=config.channels,
            expected_market_channels=config.market_channels,
        )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        _optimizer_parameter_groups(
            model,
            encoder_learning_rate=config.training.encoder_learning_rate,
            head_learning_rate=config.training.head_learning_rate,
            weight_decay=config.training.weight_decay,
        ),
        betas=(config.training.adam_beta1, config.training.adam_beta2),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loader, sampler = _full_date_loader(
        train_dataset,
        shuffle=True,
        seed=config.seed,
        num_workers=config.training.num_workers,
        minimum_group_size=config.training.objective.minimum_cross_section_size,
    )
    updates_per_epoch = math.ceil(len(loader) / config.training.dates_per_optimizer_step)
    total_update_budget = updates_per_epoch * config.training.max_epochs
    target_rank_lookup = torch.stack(
        [
            torch.from_numpy(
                cross_sectional_rank_targets(
                    train_dataset.sample_rows[f"target_{horizon}"].to_numpy(),
                    train_dataset.asof_dates,
                    minimum_cross_section_size=(
                        config.training.objective.minimum_cross_section_size
                    ),
                )
            )
            for horizon in config.horizons
        ],
        dim=1,
    )
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    optimizer.zero_grad(set_to_none=True)
    timings: list[float] = []
    rows_processed = 0
    dates_processed = 0
    successful_updates = 0
    amp_skips = 0
    epoch = 0
    while successful_updates < optimizer_updates:
        epoch += 1
        sampler.set_epoch(epoch)
        update_started = time.perf_counter()
        for date_index, batch in enumerate(loader, start=1):
            if (date_index - 1) % config.training.dates_per_optimizer_step == 0:
                update_started = time.perf_counter()
            dates_in_step = _dates_in_current_optimizer_step(
                date_index,
                total_dates=len(loader),
                configured_dates=config.training.dates_per_optimizer_step,
            )
            backward_complete_date_with_embedding_replay(
                model,
                train_dataset,
                batch,
                device=device,
                target_rank_lookup=target_rank_lookup,
                horizon_weights=config.training.objective.horizon_weights,
                epsilon=config.training.objective.epsilon,
                scale_regularization=config.training.objective.scale_regularization,
                amp_enabled=amp_enabled,
                scaler=scaler,
                physical_microbatch_size=config.training.batch_size,
                dates_in_optimizer_step=dates_in_step,
                relative_tolerance=config.training.replay_relative_tolerance,
                absolute_tolerance=config.training.replay_absolute_tolerance,
            )
            dates_processed += 1
            rows_processed += int(batch["sample_index"].numel())
            should_step = (
                date_index % config.training.dates_per_optimizer_step == 0
                or date_index == len(loader)
            )
            if not should_step:
                continue
            _, updated = _step_optimizer(
                optimizer,
                scaler,
                model.parameters(),
                max_grad_norm=config.training.max_grad_norm,
                amp_enabled=amp_enabled,
                experiment_name="finance_patch_transformer_benchmark",
            )
            if device == "cuda":
                torch.cuda.synchronize()
            if not updated:
                amp_skips += 1
                continue
            successful_updates += 1
            timings.append(time.perf_counter() - update_started)
            if successful_updates >= optimizer_updates:
                break
    measured = timings[warmup_updates:]
    seconds_per_update = float(np.mean(measured))
    return {
        "kind": "finance_transformer_complete_date_updates",
        "dataset_id": dataset_id,
        "device": device,
        "precision": "fp16" if amp_enabled else "fp32",
        "requested_updates": optimizer_updates,
        "successful_updates": successful_updates,
        "warmup_updates_excluded": warmup_updates,
        "amp_skipped_updates": amp_skips,
        "dates_processed": dates_processed,
        "rows_processed": rows_processed,
        "mean_seconds_per_update": seconds_per_update,
        "p95_seconds_per_update": float(np.percentile(measured, 95)),
        "updates_per_epoch": updates_per_epoch,
        "total_update_budget": total_update_budget,
        "projected_cell_hours": seconds_per_update * total_update_budget / 3600.0,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
        ),
        "cuda_peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if device == "cuda" else None
        ),
        "full_data_training_unchanged": True,
        "benchmark_only_stops_early": True,
    }
