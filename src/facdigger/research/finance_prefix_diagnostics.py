"""The approved C prefix: same trainer, objective, data boundaries and LR clock."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import polars as pl
import torch

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.datasets.finance_statistics import FinanceStatisticsDataset, cache_statistics
from facdigger.datasets.window import FinanceTransformerInferenceWindowDataset
from facdigger.environment import collect_environment
from facdigger.experiments.manifest import collect_git_state, sha256_json
from facdigger.models.finance_patch_transformer import build_finance_transformer_model
from facdigger.models.finance_statistics import FinanceStatisticsRanker
from facdigger.research.finance_diagnostics import (
    evaluate_diagnostic_validation,
    fixed_dates,
    score_panel,
    statistics_style_exposures,
    window_datasets,
)
from facdigger.training.common import (
    load_snapshot_inference_rows,
    load_source_provenance,
    load_training_snapshot,
)
from facdigger.training.e1_engine import select_device
from facdigger.training.finance_data import finance_data_protocol, load_finance_selection
from facdigger.training.finance_transformer_engine import train_finance_transformer
from facdigger.training.progress import append_progress
from facdigger.training.runtime import (
    TrainingControl,
    TrainingPaused,
    TrainingRuntimeConfig,
    run_lock,
    save_checkpoint,
    snapshot_checksums,
    write_json,
)


def diagnostic_inputs(snapshot: Path, checksums: Path, config: Any) -> tuple:
    if (
        config.evaluation_split != "valid"
        or config.unlock_test
        or config.initialization != "scratch"
    ):
        raise DataContractError("C requires scratch, locked holdout and validation evaluation")
    if snapshot_checksums(snapshot) != json.loads(checksums.read_text()):
        raise DataContractError("diagnostic snapshot differs from immutable checksum inventory")
    manifest, frames = load_training_snapshot(snapshot, include_features=False)
    load_source_provenance(snapshot, manifest)
    protocol = finance_data_protocol(snapshot, manifest, config)
    rows, _, _ = load_finance_selection(snapshot, manifest, sample_index=frames["sample_index"])
    labelled = {
        "F": rows.filter(pl.col("split") == "train_fit"),
        "S": rows.filter(pl.col("split") == "inner_selection"),
        "V": frames["sample_index"].filter(pl.col("split") == "valid"),
    }
    pools = {
        phase: load_snapshot_inference_rows(
            snapshot,
            manifest,
            asof_dates=part["asof_date"].unique().sort().to_list(),
        )
        for phase, part in labelled.items()
    }
    return manifest, protocol, labelled, pools


def prepare_prefix_statistics(
    snapshot: Path, checksums: Path, config: Any, output: Path, *, budget_seconds: float = 7200
) -> None:
    if not 0 < budget_seconds <= 7200:
        raise ValueError("statistics preparation is capped at two CPU allocation hours")
    if output.resolve().is_relative_to(snapshot.resolve()):
        raise ValueError("cache must be outside immutable snapshot")
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    audit = {"status": "running", "budget_seconds": budget_seconds}

    def check_stop():
        if time.monotonic() - started > budget_seconds - 60:
            raise TimeoutError("statistics preparation budget reached")

    try:
        manifest, protocol, _, pools = diagnostic_inputs(snapshot, checksums, config)
        datasets = window_datasets(snapshot, manifest, config, pools)
        for phase, dataset in datasets.items():
            check_stop()
            cache_statistics(
                dataset,
                output / phase,
                windows=tuple(config.model.statistics_windows),
                identity=protocol,
                check_stop=check_stop,
            )
        audit["status"] = "complete"
    except Exception as exc:
        audit.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        audit["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "preparation.json", audit)


def run_prefix_diagnostics(
    snapshot: Path,
    checksums: Path,
    config: Any,
    cache: Path,
    output: Path,
    *,
    candidate: str,
    repository_root: Path,
    budget_seconds: float,
) -> dict:
    if candidate not in {"finance", "statistics_linear", "statistics_mlp"}:
        raise ValueError("C has exactly three preregistered candidates")
    if not 0 < budget_seconds <= 8 * 3600:
        raise ValueError("C candidate cannot exceed the entire eight-hour allocation budget")
    if config.training.max_epochs != 10 or config.training.minimum_epochs != 6:
        raise ValueError("C must retain the original ten-epoch training clock")
    for source in (snapshot, cache):
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError("diagnostic output must be outside immutable inputs")
    git = collect_git_state(repository_root)
    if git["dirty"] is not False:
        raise DataContractError("diagnostic execution requires a committed clean code checkout")
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output / ".training.lock"):
        return _run_prefix(
            snapshot,
            checksums,
            config,
            cache,
            output,
            candidate=candidate,
            git=git,
            started=started,
            budget_seconds=budget_seconds,
        )


def _run_prefix(
    snapshot, checksums, config, cache, output, *, candidate, git, started, budget_seconds
) -> dict:
    manifest, protocol, labelled, pools = diagnostic_inputs(snapshot, checksums, config)
    identity = {
        "candidate": candidate,
        "config": config.model_dump(mode="json"),
        "data_protocol": protocol,
        "prefix_epochs": 2,
    }
    audit_path = output / "diagnostic.json"
    previous = json.loads(audit_path.read_text()) if audit_path.is_file() else None
    if previous and previous["identity"] != identity:
        raise DataContractError("diagnostic continuation identity differs")
    checkpoints = output / "checkpoints"
    last_path = checkpoints / "last.pt"
    if last_path.is_file() and previous is None:
        raise DataContractError("checkpoint has no bound diagnostic identity")
    audit = previous or {"identity": identity, "attempts": [], "holdout_used": False}
    audit["attempts"].append(
        {"git": git, "environment": collect_environment(), "budget_seconds": budget_seconds}
    )
    audit.update(status="running", training_status="running")
    write_json(audit_path, audit)
    windows = tuple(config.model.statistics_windows)
    control = TrainingControl(
        TrainingRuntimeConfig(
            checkpoint_interval_seconds=600,
            max_walltime_seconds=budget_seconds,
            shutdown_margin_seconds=180,
            handle_signals=True,
        ),
        clock=time.monotonic,
    )
    # Loading, validation and cache verification count against the same budget.
    control.started = started

    def save_observation(model, epoch):
        save_checkpoint(
            output / "observations" / f"epoch-{epoch}.pt",
            {
                "epoch": epoch,
                "identity": identity,
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            },
        )

    def progress(event):
        append_progress(output / "progress.jsonl", event, attempt=len(audit["attempts"]))
        if event["event"] == "optimizer_progress" and event["global_step"] == 100:
            groups = (
                math.ceil(
                    labelled["F"]["asof_date"].n_unique() / config.training.dates_per_optimizer_step
                )
                * 2
            )
            lower_bound = (
                time.monotonic() - started + event["elapsed_seconds"] / 100 * (groups - 100)
            )
            audit["update_only_projected_seconds"] = lower_bound
            if lower_bound >= budget_seconds - 180:
                control.request_stop("diagnostic_projection_over_budget")
        if event["event"] == "epoch_completed" and event["epoch"] == 2:
            control.request_stop("diagnostic_prefix_epoch_2")

    def check_stop():
        if time.monotonic() - started >= budget_seconds - 120 or control.reason in {
            "SIGTERM",
            "time_limit_warning",
        }:
            raise TimeoutError("C observation allocation budget/signal reached")

    def validate_last(state):
        expected = {"config": identity["config"], "data_protocol": protocol}
        if candidate != "finance":
            expected["diagnostic_model"] = candidate
        if (
            state.get("protocol_hash") != sha256_json(expected)
            or state.get("data_protocol") != protocol
            or state.get("config") != identity["config"]
        ):
            raise DataContractError("diagnostic committed checkpoint identity differs")

    try:
        cached = {
            phase: FinanceStatisticsDataset(
                cache / phase, identity=protocol, windows=windows, expected_rows=pool
            )
            for phase, pool in pools.items()
        }
        datasets = (
            window_datasets(snapshot, manifest, config, pools) if candidate == "finance" else cached
        )
        panel = fixed_dates(labelled["F"])
        audit["F_panel"] = [str(day) for day in panel]
        with control:
            last = (
                torch.load(last_path, map_location="cpu", weights_only=False)
                if last_path.is_file()
                else None
            )
            prefix_done = (
                last is not None
                and last["epoch"] == 2
                and last["progress"]["phase"] == "epoch_complete"
            )
            if last is not None:
                validate_last(last)
            if not prefix_done:
                try:
                    train_finance_transformer(
                        config,
                        train_dataset=datasets["F"],
                        train_labelled_rows=labelled["F"],
                        valid_dataset=datasets["S"],
                        valid_labelled_rows=labelled["S"],
                        data_protocol=protocol,
                        dataset_id=manifest["dataset_id"],
                        checkpoint_dir=checkpoints,
                        resume_from=last_path if last_path.is_file() else None,
                        control=control,
                        progress_callback=progress,
                        state_observer=save_observation,
                        diagnostic_model=None if candidate == "finance" else candidate,
                    )
                    raise RuntimeError("two-epoch diagnostic must pause, never finalize training")
                except TrainingPaused as exc:
                    audit["pause_reason"] = exc.reason
                last = torch.load(last_path, map_location="cpu", weights_only=False)
                validate_last(last)
            audit.update(
                training_status="paused",
                committed_epoch=last["epoch"],
                committed_phase=last["progress"]["phase"],
                global_step=last["global_step"],
                history=last["history"],
                best_epoch=last["best_epoch"],
            )
            if last["epoch"] != 2 or last["progress"]["phase"] != "epoch_complete":
                audit["status"] = "paused_before_prefix_end"
                return audit
            if last["progress"]["finished"]:
                raise DataContractError("two epochs cannot satisfy full-training completion")
            device = select_device(config.training.device)
            model = (
                build_finance_transformer_model(config, context_length=datasets["F"].context_length)
                if candidate == "finance"
                else FinanceStatisticsRanker(
                    candidate,
                    input_dim=(len(config.channels) + len(config.market_channels))
                    * (5 * len(windows) + 1),
                    horizons=tuple(config.horizons),
                )
            ).to(device)
            # A signal can arrive between the epoch commit and observer callback.
            if not (output / "observations/epoch-2.pt").exists():
                model.load_state_dict(last["model_state"], strict=True)
                save_observation(model, 2)
            fit_labels = labelled["F"].filter(pl.col("asof_date").is_in(panel))
            if candidate == "finance":
                fit = FinanceTransformerInferenceWindowDataset(
                    feature_store=datasets["F"].feature_store,
                    market_store=datasets["F"].market_store,
                    inference_index=pools["F"].filter(pl.col("asof_date").is_in(panel)),
                    channels=config.channels,
                    market_channels=config.market_channels,
                    context_length=datasets["F"].context_length,
                    primary_horizon=config.primary_horizon,
                )
            else:
                fit = cached["F"].subset(panel)
            initial = torch.load(
                output / "observations/epoch-0.pt", map_location="cpu", weights_only=False
            )
            parameter_names = set(dict(model.named_parameters()))
            audit["relative_parameter_change"] = {}
            for epoch in (0, 1, 2):
                check_stop()
                state = torch.load(
                    output / f"observations/epoch-{epoch}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                if state["identity"] != identity or state["epoch"] != epoch:
                    raise DataContractError("fixed-state observation identity differs")
                model.load_state_dict(state["model_state"], strict=True)
                numerator = sum(
                    (v - initial["model_state"][k]).float().square().sum().item()
                    for k, v in state["model_state"].items()
                    if k in parameter_names
                )
                denominator = sum(
                    v.float().square().sum().item()
                    for k, v in initial["model_state"].items()
                    if k in parameter_names
                )
                audit["relative_parameter_change"][str(epoch)] = (
                    numerator / max(denominator, 1e-30)
                ) ** 0.5
                for phase, dataset, support in (
                    ("F", fit, fit_labels),
                    ("S", datasets["S"], labelled["S"]),
                ):
                    check_stop()
                    daily, predictions = score_panel(
                        model, dataset, support, config, check_stop=check_stop
                    )
                    daily.write_parquet(output / f"epoch-{epoch}-{phase}-daily.parquet")
                    predictions.write_parquet(output / f"epoch-{epoch}-{phase}-predictions.parquet")
                    style_cache = cached["F"].subset(panel) if phase == "F" else cached["S"]
                    statistics_style_exposures(predictions, style_cache, support).write_parquet(
                        output / f"epoch-{epoch}-{phase}-styles.parquet"
                    )
            # V is opened only after the prefix ends; S chose best, epoch zero is ineligible.
            for epoch in sorted({2, int(last["best_epoch"])}):
                if epoch not in {1, 2}:
                    raise DataContractError("diagnostic best must be a completed eligible epoch")
                state = torch.load(
                    output / f"observations/epoch-{epoch}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                model.load_state_dict(state["model_state"], strict=True)
                daily, predictions = score_panel(
                    model, datasets["V"], labelled["V"], config, check_stop=check_stop
                )
                daily.write_parquet(output / f"epoch-{epoch}-V-daily.parquet")
                predictions.write_parquet(output / f"epoch-{epoch}-V-predictions.parquet")
                statistics_style_exposures(predictions, cached["V"], labelled["V"]).write_parquet(
                    output / f"epoch-{epoch}-V-styles.parquet"
                )
                check_stop()
                evaluate_diagnostic_validation(
                    snapshot,
                    manifest,
                    predictions,
                    labelled["V"],
                    config,
                    model_id=f"diagnostic-{candidate}-epoch-{epoch}",
                    checkpoint_hash=sha256_file(output / f"observations/epoch-{epoch}.pt"),
                    destination=output / f"epoch-{epoch}-V-evaluation",
                )
            audit.update(
                status="observations_complete",
                training_status="paused",
                checkpoint_sha256=sha256_file(last_path),
            )
    except TimeoutError as exc:
        audit.update(status="paused_budget", reason=str(exc))
    except Exception as exc:
        audit.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        audit["attempts"][-1]["elapsed_seconds"] = time.monotonic() - started
        write_json(audit_path, audit)
    return audit
