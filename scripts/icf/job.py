"""ICF allocation adapter. Training and resume logic remain in facdigger."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from slurm_runtime import current_task_ref, remaining_seconds

from facdigger.environment import collect_environment, environment_is_healthy
from facdigger.training.progress import append_progress
from facdigger.training.resources import training_hardware
from facdigger.training.runtime import (
    DatasetLocation,
    TrainingControl,
    TrainingPaused,
    load_training_runtime,
    resolve_dataset,
    run_lock,
    snapshot_checksums,
    write_json,
)


def check_environment() -> dict:
    import torch

    report = collect_environment()
    if not environment_is_healthy(report, require_model=True):
        raise RuntimeError("locked model environment is not healthy")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("ICF job requires exactly one allocated CUDA/MIG device")
    values = torch.ones((32, 32), device="cuda", dtype=torch.float16, requires_grad=True)
    result = values @ values
    result.float().mean().backward()
    torch.cuda.synchronize()
    if result[0, 0].item() != 32 or not torch.isfinite(values.grad).all().item():
        raise RuntimeError("CUDA FP16 forward/backward smoke check failed")
    report["allocated_hardware"] = training_hardware()
    report["job_id"] = os.environ.get("SLURM_JOB_ID")
    report["cpus_per_task"] = os.environ.get("SLURM_CPUS_PER_TASK")
    return report


def stage_snapshot(source: Path, root: Path, evidence: Path) -> DatasetLocation:
    inventory = snapshot_checksums(source)
    if "manifest.json" not in inventory:
        raise ValueError("staging input is not a complete snapshot")
    required = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    if shutil.disk_usage(root).free < int(required * 1.1):
        raise RuntimeError("insufficient node-local space for snapshot staging")
    dataset_id = str(json.loads((source / "manifest.json").read_text())["dataset_id"])
    # Dataset IDs are identifiers, never path expressions.
    if not re.fullmatch(r"[A-Za-z0-9_-]+", dataset_id):
        raise ValueError("invalid snapshot dataset ID")
    target = root / dataset_id
    shutil.copytree(source, target)
    if snapshot_checksums(target) != inventory:
        raise RuntimeError("snapshot transfer verification failed")
    checksums = evidence / f"{dataset_id}.checksums.json"
    write_json(checksums, inventory)
    return DatasetLocation(path=target, checksums=checksums)


def progress_signature(run_dir: Path) -> tuple[dict, list]:
    """Read committed progress under the same writer lock as the training entry."""
    matrix_path = run_dir / "matrix.json"
    lock = ".research.lock" if matrix_path.is_file() else ".training.lock"
    with run_lock(run_dir / lock):
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        completed = 0
        child = run_dir
        if matrix_path.is_file():
            stages = json.loads(matrix_path.read_text())["stages"]
            completed = sum(item["status"] == "complete" for item in stages)
            active = [item for item in stages if item["status"] != "complete" and "run_dir" in item]
            if len(active) > 1:
                raise RuntimeError("research must identify at most one unfinished child run")
            if not active:
                return manifest, [completed, 0, 0, None, None, None, None]
            child = Path(active[0]["run_dir"]).resolve()
            if not child.is_relative_to(run_dir.resolve()):
                raise RuntimeError("research child is outside its bound run")
        elif manifest.get("status") == "complete":
            completed = 1
        checkpoint_path = child / "checkpoints" / "last.pt"
        if not checkpoint_path.is_file():
            if any((child / "checkpoints").glob("*.pt")):
                raise RuntimeError("existing weights have no committed last.pt")
            return manifest, [completed, 0, 0, None, None, None, None]
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        progress = checkpoint["progress"]
        return manifest, [
            completed,
            checkpoint["epoch"],
            checkpoint["global_step"],
            progress["phase"],
            progress.get("cursor"),
            progress.get("local_cursor"),
            progress.get("market_cursor"),
        ]


def progress_advanced(before: list | None, after: list) -> bool:
    # Phase names alone can change when replaying probe/selection entry. Only
    # consumed cursors, completed epochs/stages and successful updates count.
    def position(value: list) -> tuple:
        stages, epoch, step, phase, cursor, local, market = value
        return (stages, epoch, cursor or local or 0, market or 0, phase == "epoch_complete", step)

    if before is None:
        before = [0, 0, 0, None, None, None, None]
    if position(after) < position(before) or (after[0] == before[0] and after[2] < before[2]):
        raise RuntimeError("committed training progress regressed")
    return position(after) > position(before)


def settle_attempt(
    run_dir: Path, state: dict, *, result: dict, progress: list | None, previous: bool = False
) -> None:
    active = state["unsettled_attempts"][0] if previous else state["active_attempt"]
    if result["status"] != "blocked":
        advanced = (
            active.get("phase") != "inspection"
            and progress is not None
            and progress_advanced(active["progress_before"], progress)
        )
        state["no_progress"] = 0 if advanced else state["no_progress"] + 1
    if progress is not None:
        state["progress"] = progress
    evidence = run_dir / "allocations" / f"{active['job_id']}-{active['number']}"
    write_json(evidence / "result.json", {**active, **result, "progress_after": progress})
    if previous:
        state["unsettled_attempts"].pop(0)
    else:
        state["active_attempt"] = None
    write_json(run_dir / "allocation_state.json", state)


def signal_seconds(runtime) -> int:
    if not runtime.handle_signals or runtime.shutdown_margin_seconds <= 0:
        raise ValueError("ICF requires signal handling and a positive shutdown margin")
    return math.ceil(runtime.shutdown_margin_seconds)


def load_trainer(mode: str):
    # Heavy imports happen before the final budget sample in the job step.
    if mode == "finance-transformer":
        from facdigger.training.finance_transformer import run_finance_transformer
        from facdigger.training.finance_transformer_config import load_finance_transformer_config

        return run_finance_transformer, load_finance_transformer_config
    if mode == "finance-pretrain":
        from facdigger.training.finance_pretrain import run_finance_pretraining
        from facdigger.training.finance_pretrain_config import load_finance_pretraining_config

        return run_finance_pretraining, load_finance_pretraining_config
    if mode == "transformer-run":
        from facdigger.research.transformer_config import load_transformer_comparison_config
        from facdigger.research.transformer_runner import run_transformer_comparison

        return run_transformer_comparison, load_transformer_comparison_config
    raise ValueError("unknown training mode")


def run_step(evidence: Path) -> int:
    plan = json.loads((evidence / "step.json").read_text())
    trainer, load_config = load_trainer(plan["mode"])
    config = load_config(plan["config"])
    runtime = load_training_runtime(evidence / "runtime.json")
    kwargs = {"repository_root": plan["code"], "run_dir": plan["run_dir"]}
    if plan["mode"] == "transformer-run":
        from facdigger.training.resources import load_resource_budget

        kwargs["resource_budget"] = load_resource_budget(plan["resource_budget"])
    else:
        kwargs["dataset_dir"] = Path(plan["dataset"])
    # Include srun launch, interpreter/import/configuration time, even when the
    # runtime limit is shorter than the allocation. Only this adapter knows Slurm.
    left = remaining_seconds(os.environ["SLURM_JOB_ID"])
    budget = min(left, plan["deadline_epoch_seconds"] - time.time())
    runtime = type(runtime).model_validate({**runtime.model_dump(), "max_walltime_seconds": budget})
    with TrainingControl(runtime) as control:
        write_json(evidence / "step_runtime.json", runtime.model_dump(mode="json"))
        write_json(
            evidence / "step_started.json",
            {
                "bootstrap_seconds": time.time() - plan["budget_sampled_epoch_seconds"],
                "remaining_seconds": left,
            },
        )
        try:
            trainer(config, **kwargs, control=control)
        except TrainingPaused as exc:
            print(str(exc), file=sys.stderr, flush=True)
            return 75
    return 0


def run_job() -> int:
    job_id = os.environ["SLURM_JOB_ID"]
    run_dir = Path(os.environ["FD_RUN_DIR"]).resolve()
    code = Path(os.environ["FD_CODE_ROOT"]).resolve()
    mode = os.environ["FD_MODE"]
    if mode not in {"finance-transformer", "finance-pretrain", "transformer-run"}:
        raise ValueError("unknown training mode")
    with run_lock(run_dir / ".allocation.lock"):
        startup_log = run_dir / "allocations/startup.jsonl"
        startup_log.parent.mkdir(parents=True, exist_ok=True)
        append_progress(startup_log, {"event": "allocation_startup", "job_id": job_id}, attempt=0)
        try:
            task_ref = current_task_ref()
            charged = remaining_seconds(job_id)
        except BaseException as exc:
            append_progress(startup_log, {
                "event": "allocation_startup_failed", "job_id": job_id,
                "error": f"{type(exc).__name__}: {exc}",
            }, attempt=0)
            raise
        append_progress(startup_log, {
            "event": "allocation_startup_complete", "job_id": job_id,
            "task_ref": task_ref, "remaining_seconds": charged,
        }, attempt=0)
        started = time.monotonic()
        state_path = run_dir / "allocation_state.json"
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {
                "attempts": 0,
                "charged_seconds": 0,
                "no_progress": 0,
                "progress": None,
            }
        )
        max_attempts = int(os.environ.get("FD_MAX_ATTEMPTS", "10"))
        max_seconds = int(os.environ.get("FD_MAX_TOTAL_SECONDS", "1209600"))
        max_no_progress = int(os.environ.get("FD_MAX_NO_PROGRESS", "2"))
        unsettled = state.setdefault("unsettled_attempts", [])
        if state["no_progress"] >= max_no_progress and not (
            unsettled or state.get("active_attempt")
        ):
            raise RuntimeError("repeated allocations made no training progress")
        if state["attempts"] >= max_attempts:
            raise RuntimeError(
                "allocation attempt limit reached; inspect the run before continuing"
            )
        if state["charged_seconds"] + charged > max_seconds:
            raise RuntimeError("allocation time budget exhausted")
        state["attempts"] += 1
        state["charged_seconds"] += charged
        if state.get("active_attempt") is not None:
            unsettled.append(state["active_attempt"])
        # Register before checkpoint inspection/imports too. Death during this
        # read-only phase must consume an attempt, not bypass the retry limits.
        state["active_attempt"] = {
            "number": state["attempts"],
            "job_id": job_id,
            "progress_before": None,
            "reserved_seconds": charged,
            "phase": "inspection",
        }
        write_json(state_path, state)
        evidence = run_dir / "allocations" / f"{job_id}-{state['attempts']}"
        evidence.mkdir(parents=True, exist_ok=True)
        temporary = None
        phase = "inspection"
        manifest = {}
        returncode = 1
        failure = None
        after = None
        before = None
        blocked = False
        try:
            _, before = progress_signature(run_dir)
            while unsettled:
                # At most the oldest unfinished attempt entered training; later
                # unfinished inspections never launched a child. Settle in order.
                settle_attempt(
                    run_dir,
                    state,
                    progress=before,
                    previous=True,
                    result={
                        "status": "interrupted",
                        "settled_by_job_id": job_id,
                        "reservation_retained": True,
                    },
                )
            if state["no_progress"] >= max_no_progress:
                blocked = True
                raise RuntimeError("repeated allocations made no training progress")
            state["active_attempt"].update({"progress_before": before, "phase": "environment"})
            write_json(state_path, state)
            phase = "environment"
            write_json(evidence / "environment.json", check_environment())
            minimum = int(os.environ.get("FD_MIN_OUTPUT_FREE_BYTES", "2147483648"))
            if shutil.disk_usage(run_dir).free < minimum:
                raise RuntimeError("insufficient persistent checkpoint space")
            runtime = load_training_runtime(os.environ["FD_RUNTIME"])
            if int(os.environ.get("FD_SIGNAL_SECONDS", "0")) != signal_seconds(runtime):
                raise ValueError("submitted signal lead differs from the runtime shutdown margin")
            phase = "input_staging"
            staging = os.environ.get("FD_STAGING_ROOT")
            if staging:
                Path(staging).mkdir(parents=True, exist_ok=True)
                temporary = Path(tempfile.mkdtemp(prefix=f"facdigger-{job_id}-", dir=staging))
            dataset = None
            if mode != "transformer-run":
                dataset = Path(os.environ["FD_DATASET"]).resolve()
                if temporary:
                    dataset = resolve_dataset(dataset, runtime)
                    location = stage_snapshot(dataset, temporary, evidence)
                    dataset_id = json.loads((dataset / "manifest.json").read_text())["dataset_id"]
                    runtime.dataset_overrides[dataset_id] = location
            elif runtime.fold_snapshots:
                for fold, source in list(runtime.fold_snapshots.items()):
                    dataset_id = json.loads((source.path / "manifest.json").read_text())[
                        "dataset_id"
                    ]
                    resolve_dataset(
                        source.path,
                        type(runtime)(dataset_overrides={dataset_id: source}),
                        dataset_id=dataset_id,
                    )
                    location = (
                        stage_snapshot(source.path, temporary, evidence) if temporary else source
                    )
                    runtime.fold_snapshots[fold] = location
                    runtime.dataset_overrides[dataset_id] = location
            elif temporary:
                plans_path = run_dir / "folds.json"
                if not plans_path.is_file():
                    raise ValueError(
                        "first matrix staging requires explicit prebuilt fold_snapshots"
                    )
                for plan in json.loads(plans_path.read_text()):
                    runtime.dataset_overrides[plan["dataset_id"]] = stage_snapshot(
                        Path(plan["dataset_path"]),
                        temporary,
                        evidence,
                    )
            phase = "training"
            state["active_attempt"]["phase"] = phase
            write_json(state_path, state)
            left = remaining_seconds(job_id)
            sampled = time.time()
            budget = min(left, runtime.max_walltime_seconds or left)
            runtime = type(runtime).model_validate(
                {**runtime.model_dump(), "max_walltime_seconds": budget}
            )
            write_json(evidence / "runtime.json", runtime.model_dump(mode="json"))
            write_json(
                evidence / "step.json",
                {
                    "mode": mode,
                    "config": os.environ["FD_CONFIG"],
                    "code": str(code),
                    "run_dir": str(run_dir),
                    "dataset": str(dataset) if dataset else None,
                    "resource_budget": os.environ.get("FD_RESOURCE_BUDGET"),
                    "budget_sampled_epoch_seconds": sampled,
                    "deadline_epoch_seconds": sampled + budget,
                },
            )
            result = subprocess.run(
                [
                    "srun",
                    sys.executable,
                    str(code / "scripts/icf/job.py"),
                    "--run-step",
                    str(evidence),
                ],
                cwd=code,
                check=False,
            )
            returncode = result.returncode
            manifest, after = progress_signature(run_dir)
            if returncode == 75 and (manifest.get("status") != "paused" or after[1] == 0):
                raise RuntimeError("pause exit code without a committed paused run")
            if returncode == 0 and manifest.get("status") != "complete":
                raise RuntimeError("successful exit without a completed run")
        except BaseException as exc:
            failure = {"phase": phase, "type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            elapsed = math.ceil(time.monotonic() - started)
            state["charged_seconds"] -= max(0, charged - elapsed)
            state["last_exit_code"] = returncode
            # Refund/record observable failures too. If progress is corrupt, keep
            # that error and close the attempt without pretending it advanced.
            try:
                if after is None and phase == "training":
                    _, after = progress_signature(run_dir)
                elif after is None:
                    after = before
            except Exception as exc:
                after = before
                failure = failure or {
                    "phase": "progress_read",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                raise
            finally:
                settle_attempt(
                    run_dir,
                    state,
                    progress=after,
                    result={
                        "status": "blocked"
                        if blocked
                        else "failed"
                        if failure or returncode not in {0, 75}
                        else "paused"
                        if returncode == 75
                        else "complete",
                        "exit_code": returncode,
                        "elapsed_seconds": elapsed,
                        "error": failure,
                    },
                )
                if temporary is not None:
                    shutil.rmtree(temporary)
        if returncode == 75:
            if state["no_progress"] >= max_no_progress:
                raise RuntimeError("repeated allocations made no training progress")
            if (
                os.environ.get("FD_AUTO_REQUEUE", "0") == "1"
                and manifest.get("stop_reason") in {"walltime_budget", "time_limit_warning"}
                and state["attempts"] < max_attempts
                and state["charged_seconds"] + charged <= max_seconds
            ):
                # All result/ledger writes and scratch cleanup precede requeue,
                # which may terminate this batch process before returning.
                subprocess.run(["scontrol", "requeue", task_ref], check=True)
        return returncode


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--check-environment"]:
        print(json.dumps(check_environment(), ensure_ascii=False, indent=2))
    elif len(args) == 2 and args[0] == "--signal-seconds":
        print(signal_seconds(load_training_runtime(args[1])))
    elif len(args) == 2 and args[0] == "--run-step":
        raise SystemExit(run_step(Path(args[1])))
    elif args:
        raise SystemExit(
            "Usage: job.py [--check-environment | --signal-seconds RUNTIME | --run-step EVIDENCE]"
        )
    else:
        raise SystemExit(run_job())
