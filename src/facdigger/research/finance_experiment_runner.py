"""Complete F/S research cells using the existing Finance trainer and resume contract."""

from __future__ import annotations

import json
import math
import time
from contextlib import ExitStack
from pathlib import Path

import torch

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.datasets.finance_statistics import FinanceStatisticsDataset
from facdigger.environment import collect_environment
from facdigger.research.finance_diagnostics import window_datasets
from facdigger.research.finance_experiment_config import candidate_settings, cell_config
from facdigger.research.finance_experiment_plan import (
    clean_git,
    disjoint_output,
    get_cell,
    load_fold_inputs,
    read_experiment_plan,
)
from facdigger.research.finance_experiment_state import (
    CompleteStateObserver,
    execution_identity,
    file_inventory,
    import_prefix,
    validate_checkpoint,
    verify_inventory,
)
from facdigger.training.finance_transformer_engine import train_finance_transformer
from facdigger.training.progress import append_progress
from facdigger.training.runtime import (
    TrainingControl,
    TrainingPaused,
    TrainingRuntimeConfig,
    run_lock,
    save_checkpoint,
    write_json,
)


def _cell_root(output: Path, cell_id: str) -> Path:
    root = output / "cells" / cell_id
    if not root.resolve().is_relative_to(output.resolve()):
        raise DataContractError("cell output escaped the experiment directory")
    return root


def _read_cell(root: Path, plan: dict, cell: dict) -> dict:
    audit = json.loads((root / "cell.json").read_text())
    if (
        audit["identity"]["plan_hash"] != plan["plan_hash"]
        or audit["identity"]["cell"] != cell
        or audit.get("holdout_used") is not False
        or audit.get("validation_used") is not False
    ):
        raise DataContractError("cell audit differs from the frozen experiment")
    if audit["status"] == "complete":
        verify_inventory(root, audit["artifacts"])
    return audit


def _budget(runtime: TrainingRuntimeConfig, audit: dict, cumulative: float | None):
    if cumulative is None:
        return runtime
    if not math.isfinite(cumulative) or cumulative <= 0 or runtime.max_walltime_seconds is None:
        raise ValueError("cumulative budget requires finite positive per-attempt and total budgets")
    charges = [
        attempt.get("elapsed_seconds", attempt["reserved_seconds"])
        for attempt in audit["attempts"]
    ]
    if any(value is None or not math.isfinite(value) or value < 0 for value in charges):
        raise DataContractError("previous attempt has no finite resource accounting")
    charged = sum(charges)
    remaining = min(runtime.max_walltime_seconds, cumulative - charged)
    if remaining <= runtime.shutdown_margin_seconds:
        raise TrainingPaused("cumulative_budget", Path("cell.json"))
    return TrainingRuntimeConfig.model_validate({
        **runtime.model_dump(mode="json"), "max_walltime_seconds": remaining,
    })


def run_experiment_cell(
    output: Path, cell_id: str, *, repository_root: Path,
    runtime: TrainingRuntimeConfig, cache: Path | None = None,
    parent_prefix: Path | None = None, cumulative_budget_seconds: float | None = None,
) -> dict:
    """One independent writer per cell; no provider, V, holdout or production calls."""
    started = time.monotonic()
    plan = read_experiment_plan(output)
    clean_git(repository_root, plan["identity"]["code_commit"])
    cell = get_cell(plan, cell_id)
    root = _cell_root(output, cell_id)
    cache = (cache or output / "cache" / cell["fold_id"]).resolve()
    disjoint_output(root, cache)
    location = plan["identity"]["folds"][cell["fold_id"]]
    disjoint_output(output, Path(location["path"]))
    if parent_prefix is not None:
        disjoint_output(output, parent_prefix)
        parent_prefix = parent_prefix.resolve()
    config = cell_config(plan["identity"]["supervised_config"], cell["seed"])
    # Always enable update-boundary checkpoints, including on ordinary local machines.
    if runtime.checkpoint_interval_seconds is None:
        runtime = TrainingRuntimeConfig.model_validate({
            **runtime.model_dump(mode="json"), "checkpoint_interval_seconds": 600,
        })
    with run_lock(root / ".training.lock"):
        audit_path = root / "cell.json"
        previous = _read_cell(root, plan, cell) if audit_path.is_file() else None
        if previous and previous["status"] == "complete":
            if parent_prefix is not None and previous["parent_prefix"] != str(parent_prefix):
                raise DataContractError("completed cell has a different prefix parent")
            return previous
        if (output / "selection-seal.json").exists():
            raise DataContractError("sealed cells cannot be trained again")
        environment = collect_environment()
        execution = execution_identity(config, environment)
        identity = {
            "plan_hash": plan["plan_hash"], "cell": cell,
            "config": config.model_dump(mode="json"),
            "data_protocol": location["data_protocol"], "execution": execution,
        }
        # All paired cells use the same numerical environment. Host/path is not an identity.
        # Different cells may legitimately arrive together; only their short shared
        # binding waits. The full-duration per-cell writer lock remains fail-fast.
        lock_timeout = 60.0
        if runtime.max_walltime_seconds is not None:
            lock_timeout = min(lock_timeout, max(
                0.0, runtime.max_walltime_seconds - runtime.shutdown_margin_seconds
                - (time.monotonic() - started),
            ))
        with run_lock(output / ".plan.lock", timeout_seconds=lock_timeout):
            binding_path = output / "execution.json"
            if binding_path.is_file():
                if json.loads(binding_path.read_text()) != execution:
                    raise DataContractError("matrix numerical execution environment differs")
            else:
                write_json(binding_path, execution)
        if previous:
            if previous["identity"] != identity:
                raise DataContractError("cell continuation identity differs")
            if parent_prefix is not None and previous["parent_prefix"] != str(parent_prefix):
                raise DataContractError("cell prefix parent cannot change")
            parent_prefix = Path(previous["parent_prefix"]) if previous["parent_prefix"] else None
            audit = previous
        else:
            if set(p.name for p in root.iterdir()) - {".training.lock"}:
                raise DataContractError("unbound cell output is not empty")
            audit = {
                "identity": identity, "attempts": [], "holdout_used": False,
                "validation_used": False,
                "parent_prefix": str(parent_prefix) if parent_prefix else None,
            }
        try:
            effective = _budget(runtime, audit, cumulative_budget_seconds)
        except TrainingPaused:
            audit.update(status="paused", stop_reason="cumulative_budget")
            write_json(audit_path, audit)
            return audit
        audit["attempts"].append({
            "reserved_seconds": effective.max_walltime_seconds,
            "runtime": effective.model_dump(mode="json"), "environment": environment,
        })
        audit.update(status="running", phase="inputs")
        audit.pop("error", None)
        audit.pop("stop_reason", None)
        if cumulative_budget_seconds is not None:
            audit["cumulative_budget_seconds"] = cumulative_budget_seconds
        write_json(audit_path, audit)
        last = root / "checkpoints/last.pt"
        control = TrainingControl(effective)
        control.started = started
        attempt = len(audit["attempts"])

        def check_stop():
            reason = control.stop_reason()
            if reason:
                raise TrainingPaused(reason, last)

        def progress(event):
            append_progress(root / "progress.jsonl", event, attempt=attempt)
            for key in ("epoch", "global_step"):
                if key in event:
                    audit[key] = event[key]
            if event["event"] in {"checkpoint_saved", "epoch_completed", "training_started"}:
                audit["phase"] = event.get("phase", event["event"])
                write_json(audit_path, audit)

        try:
            with control:
                check_stop()
                manifest, protocol, labelled, pools = load_fold_inputs(
                    plan, cell["fold_id"], seed=cell["seed"],
                )
                check_stop()
                cached = {
                    phase: FinanceStatisticsDataset(
                        cache / phase, identity=protocol,
                        windows=tuple(config.model.statistics_windows), expected_rows=pools[phase],
                    )
                    for phase in ("F", "S")
                }
                datasets = (
                    window_datasets(Path(location["path"]), manifest, config, pools)
                    if cell["candidate"] == "finance" else cached
                )
                check_stop()
                if parent_prefix is not None:
                    import_prefix(parent_prefix, root, identity, config)
                observer = CompleteStateObserver(
                    root=root, identity=identity, config=config, datasets=datasets,
                    labelled=labelled, cached=cached, check_stop=check_stop, attempt=attempt,
                )
                state = None
                if last.is_file():
                    state = torch.load(last, map_location="cpu", weights_only=False)
                    validate_checkpoint(state, config, protocol, cell["candidate"])
                    observer.repair(state, parent=parent_prefix is not None)
                kind, dropout = candidate_settings(cell["candidate"])
                if state is None or not state["progress"]["finished"]:
                    audit["phase"] = "training"
                    write_json(audit_path, audit)
                    _, training = train_finance_transformer(
                        config, train_dataset=datasets["F"], train_labelled_rows=labelled["F"],
                        valid_dataset=datasets["S"], valid_labelled_rows=labelled["S"],
                        data_protocol=protocol, dataset_id=manifest["dataset_id"],
                        checkpoint_dir=root / "checkpoints",
                        resume_from=last if last.is_file() else None,
                        control=control, progress_callback=progress, state_observer=observer,
                        diagnostic_model=kind, statistics_dropout=dropout,
                    )
                    audit["training"] = training
                state = torch.load(last, map_location="cpu", weights_only=False)
                validate_checkpoint(state, config, protocol, cell["candidate"])
                if (
                    not state["progress"]["finished"]
                    or state["progress"]["phase"] != "epoch_complete"
                ):
                    raise DataContractError("cell did not finish its complete training policy")
                # A time-limit signal can occur after the final commit, before best export/observer.
                save_checkpoint(root / "checkpoints/best.pt", state["best_checkpoint"])
                observer.repair(state, parent=parent_prefix is not None)
                best = torch.load(
                    root / "checkpoints/best.pt", map_location="cpu", weights_only=False,
                )
                if (
                    best["epoch"] != state["best_epoch"]
                    or best["protocol_hash"] != state["protocol_hash"]
                    or best["best_selection_score"] != state["best_selection_score"]
                    or any(not torch.equal(v.cpu(), best["model_state"][k].cpu())
                           for k, v in state["best_checkpoint"]["model_state"].items())
                ):
                    raise DataContractError("best export differs from the committed S selection")
                write_json(root / "history.json", state["history"])
                artifacts = [last, root / "checkpoints/best.pt", root / "history.json"]
                for epoch in [0, *[row["epoch"] for row in state["history"]]]:
                    if not observer.completed(epoch):
                        raise DataContractError("complete epoch is missing F/S observation")
                    marker = observer.marker(epoch)
                    artifacts.append(marker)
                    artifacts.extend(
                        root / name for name in json.loads(marker.read_text())["files"]
                    )
                check_stop()
                audit.update(
                    status="complete", phase="complete", epoch=state["epoch"],
                    global_step=state["global_step"], best_epoch=state["best_epoch"],
                    best_selection_score=state["best_selection_score"],
                    artifacts=file_inventory(root, artifacts),
                )
        except TrainingPaused as exc:
            audit.update(status="paused", stop_reason=exc.reason)
        except Exception as exc:
            audit.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if last.is_file():
                committed = torch.load(last, map_location="cpu", weights_only=False)
                audit.update(
                    epoch=committed["epoch"], global_step=committed["global_step"],
                    committed_phase=committed["progress"]["phase"],
                    best_epoch=committed["best_epoch"],
                )
            audit["attempts"][-1]["elapsed_seconds"] = time.monotonic() - started
            write_json(audit_path, audit)
        return audit


def seal_experiment_selections(output: Path, *, repository_root: Path) -> dict:
    """Freeze every native S-best together; this action does not grant or read V/test."""
    plan = read_experiment_plan(output)
    clean_git(repository_root, plan["identity"]["code_commit"])
    with ExitStack() as locks:
        locks.enter_context(run_lock(output / ".plan.lock"))
        cells = plan["identity"]["cells"]
        for cell in cells:
            locks.enter_context(run_lock(_cell_root(output, cell["cell_id"]) / ".training.lock"))
        for fold_id in plan["identity"]["folds"]:
            load_fold_inputs(plan, fold_id, seed=plan["identity"]["research"]["seeds"][0])
        selections = {}
        for cell in cells:
            root = _cell_root(output, cell["cell_id"])
            if not (root / "cell.json").exists():
                raise DataContractError("cannot seal selections before every cell completes")
            audit = _read_cell(root, plan, cell)
            if audit["status"] != "complete":
                raise DataContractError("cannot seal selections before every cell completes")
            selections[cell["cell_id"]] = {
                "best_epoch": audit["best_epoch"],
                "best_checkpoint_sha256": sha256_file(root / "checkpoints/best.pt"),
                "cell_audit_sha256": sha256_file(root / "cell.json"),
            }
        seal = {
            "plan_hash": plan["plan_hash"], "selections": selections,
            "selection_rule": {
                "metric": "native S mean_rank_ic - penalty * subperiod_std",
                "penalty": plan["identity"]["supervised_config"]["training"]["objective"][
                    "selection_stability_penalty"
                ],
                "subperiods": plan["identity"]["supervised_config"]["training"]["objective"][
                    "selection_subperiods"
                ],
            },
            "validation_used": False, "holdout_used": False,
            "validation_execution_implemented": False,
        }
        path = output / "selection-seal.json"
        if path.is_file():
            if json.loads(path.read_text()) != seal:
                raise DataContractError("sealed selections changed")
        else:
            write_json(path, seal)
        return seal
