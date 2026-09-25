"""ICF allocation adapter. Training and resume logic remain in facdigger."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from facdigger.environment import collect_environment, environment_is_healthy
from facdigger.training.resources import training_hardware
from facdigger.training.runtime import (
    DatasetLocation,
    load_training_runtime,
    resolve_dataset,
    run_lock,
    snapshot_checksums,
    write_json,
)


def parse_time_left(value: str) -> int:
    match = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)", value.strip())
    if match is None:
        raise ValueError(f"cannot determine finite Slurm time remaining: {value!r}")
    days, hours, minutes, seconds = (int(item or 0) for item in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError("invalid Slurm time remaining")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def remaining_seconds(job_id: str) -> int:
    result = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%L"],
        check=True,
        text=True,
        capture_output=True,
    )
    return parse_time_left(result.stdout)


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
    import torch

    manifest = json.loads((run_dir / "manifest.json").read_text())
    if (run_dir / "matrix.json").is_file():
        stages = json.loads((run_dir / "matrix.json").read_text())["stages"]
        active = [item for item in stages if item["status"] == "paused"]
        if len(active) != 1:
            raise RuntimeError("paused research must identify one paused child run")
        child = Path(active[0]["run_dir"])
        completed = sum(item["status"] == "complete" for item in stages)
    else:
        child = run_dir
        completed = 0
    checkpoint = torch.load(
        child / "checkpoints" / "last.pt", map_location="cpu", weights_only=False
    )
    progress = checkpoint["progress"]
    signature = [
        completed,
        checkpoint["epoch"],
        checkpoint["global_step"],
        progress["phase"],
        progress.get("cursor"),
        progress.get("local_cursor"),
        progress.get("market_cursor"),
    ]
    return manifest, signature


def run_job() -> int:
    job_id = os.environ["SLURM_JOB_ID"]
    run_dir = Path(os.environ["FD_RUN_DIR"]).resolve()
    code = Path(os.environ["FD_CODE_ROOT"]).resolve()
    mode = os.environ["FD_MODE"]
    if mode not in {"finance-transformer", "finance-pretrain", "transformer-run"}:
        raise ValueError("unknown training mode")
    with run_lock(run_dir / ".allocation.lock"):
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
        if state["attempts"] >= int(os.environ.get("FD_MAX_ATTEMPTS", "10")):
            raise RuntimeError(
                "allocation attempt limit reached; inspect the run before continuing"
            )
        charged = remaining_seconds(job_id)
        if state["charged_seconds"] + charged > int(
            os.environ.get("FD_MAX_TOTAL_SECONDS", "1209600")
        ):
            raise RuntimeError("allocation time budget exhausted")
        started = time.monotonic()
        # Charge the whole allocation first: hard kills cannot evade the total budget.
        state["attempts"] += 1
        state["charged_seconds"] += charged
        write_json(state_path, state)
        evidence = run_dir / "allocations" / f"{job_id}-{state['attempts']}"
        evidence.mkdir(parents=True)
        write_json(evidence / "environment.json", check_environment())
        minimum = int(os.environ.get("FD_MIN_OUTPUT_FREE_BYTES", "2147483648"))
        if shutil.disk_usage(run_dir).free < minimum:
            raise RuntimeError("insufficient persistent checkpoint space")
        runtime = load_training_runtime(os.environ["FD_RUNTIME"])
        if not runtime.handle_signals:
            raise ValueError("ICF signal forwarding requires runtime.handle_signals=true")
        staging = os.environ.get("FD_STAGING_ROOT")
        temporary = None
        try:
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
                    # Check the frozen transfer inventory before generating a new one.
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
            left = remaining_seconds(job_id)
            budget = min(left, runtime.max_walltime_seconds or left)
            runtime = type(runtime).model_validate(
                {**runtime.model_dump(), "max_walltime_seconds": budget}
            )
            runtime_file = evidence / "runtime.json"
            write_json(runtime_file, runtime.model_dump(mode="json"))
            command = [
                "srun",
                sys.executable,
                "-m",
                "facdigger",
                "research" if mode == "transformer-run" else "train",
                mode,
                "--config",
                os.environ["FD_CONFIG"],
                "--run-dir",
                str(run_dir),
                "--runtime",
                str(runtime_file),
            ]
            if dataset is not None:
                command.extend(["--dataset", str(dataset)])
            if mode == "transformer-run" and os.environ.get("FD_RESOURCE_BUDGET"):
                command.extend(["--resource-budget", os.environ["FD_RESOURCE_BUDGET"]])
            result = subprocess.run(command, cwd=code, check=False)
            state["charged_seconds"] -= max(0, charged - int(time.monotonic() - started))
            state["last_exit_code"] = result.returncode
            if result.returncode == 75:
                manifest, signature = progress_signature(run_dir)
                if manifest["status"] != "paused":
                    raise RuntimeError("pause exit code without a committed paused run")
                state["no_progress"] = (
                    state["no_progress"] + 1 if signature == state["progress"] else 0
                )
                state["progress"] = signature
                write_json(state_path, state)
                if state["no_progress"] >= int(os.environ.get("FD_MAX_NO_PROGRESS", "2")):
                    raise RuntimeError("repeated allocations made no training progress")
                if (
                    os.environ.get("FD_AUTO_REQUEUE", "0") == "1"
                    and manifest.get("stop_reason")
                    in {
                        "walltime_budget",
                        "time_limit_warning",
                    }
                    and state["attempts"] < int(os.environ.get("FD_MAX_ATTEMPTS", "10"))
                ):
                    subprocess.run(["scontrol", "requeue", job_id], check=True)
            write_json(state_path, state)
            return result.returncode
        finally:
            if temporary is not None:
                shutil.rmtree(temporary)


if __name__ == "__main__":
    if sys.argv[1:] == ["--check-environment"]:
        print(json.dumps(check_environment(), ensure_ascii=False, indent=2))
    elif sys.argv[1:]:
        raise SystemExit("Usage: job.py [--check-environment]")
    else:
        raise SystemExit(run_job())
