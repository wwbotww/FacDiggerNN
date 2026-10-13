"""School allocation adapter for a frozen intervention task list; no scientific overrides."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from slurm_runtime import current_task_ref, remaining_seconds


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def progress(root: Path) -> list:
    path = root / "cell.json"
    cell = json.loads(path.read_text()) if path.exists() else {}
    return [
        cell.get("epoch", 0),
        cell.get("global_step", 0),
        len(list((root / "chunks").glob("*/complete.json"))),
        len(list((root / "observations").glob("epoch-*.json"))),
    ]


def can_requeue(ledger: dict, audit: dict, spec: dict) -> bool:
    return (
        audit.get("status") == "paused"
        and audit.get("stop_reason") in {"walltime_budget", "time_limit_warning"}
        and ledger["no_progress"] < spec["max_no_progress"]
        and ledger["attempts"] < spec["max_attempts"]
        and ledger["charged_seconds"]
        + spec["shutdown_margin_seconds"]
        + spec["exit_reserve_seconds"]
        < spec["cumulative_budget_seconds"]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    os.umask(0o077)
    spec = json.loads(args.deployment.read_text())
    task_ref = current_task_ref()
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    task = spec["tasks"][index]
    root = Path(task["output"]) / "cells" / task["cell_id"]
    ops = args.deployment.resolve().parent
    ledger_path = ops / "ledgers" / f"task-{index}.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = ops / "allocations" / f"{task_ref}-{os.environ.get('SLURM_RESTART_COUNT', '0')}.json"
    with (ledger_path.with_suffix(".lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = (
            json.loads(ledger_path.read_text())
            if ledger_path.exists()
            else {
                "attempts": 0,
                "no_progress": 0,
                "charged_seconds": 0,
                "task": task,
            }
        )
        if ledger["task"] != task or receipt.exists():
            raise RuntimeError("allocation identity changed or duplicate writer")
        if (
            ledger["attempts"] >= spec["max_attempts"]
            or ledger["no_progress"] >= spec["max_no_progress"]
        ):
            raise RuntimeError("bounded allocation retry limit reached")
        remaining = remaining_seconds(os.environ["SLURM_JOB_ID"])
        reserved = min(remaining, spec["cumulative_budget_seconds"] - ledger["charged_seconds"])
        if reserved <= spec["shutdown_margin_seconds"] + spec["exit_reserve_seconds"]:
            raise RuntimeError("no safe allocation budget remains")
        ledger["attempts"] += 1
        ledger["charged_seconds"] += reserved  # Conservative charge survives abrupt termination.
        save(ledger_path, ledger)
        record = {
            "task_ref": task_ref,
            "task": task,
            "phase": "imports",
            "status": "starting",
            "code_commit": spec["code_commit"],
            "reserved_seconds": reserved,
            "holdout_used": False,
            "validation_used": False,
        }
        save(receipt, record)
        before, rc = None, None
        try:
            from job import check_environment

            from facdigger.data.snapshots import sha256_file
            from facdigger.research.finance_experiment_config import cell_config
            from facdigger.research.finance_experiment_plan import clean_git, read_experiment_plan
            from facdigger.research.finance_experiment_runner import run_experiment_cell
            from facdigger.research.finance_experiment_state import execution_identity
            from facdigger.training.runtime import TrainingRuntimeConfig, load_training_runtime

            code = Path(__file__).resolve().parents[2]
            clean_git(code, spec["code_commit"])
            admission_path = Path(spec["admission"])
            if sha256_file(admission_path) != spec["admission_sha256"]:
                raise RuntimeError("GPU admission receipt changed")
            admission = json.loads(admission_path.read_text())
            if admission["code_commit"] != spec["code_commit"]:
                raise RuntimeError("GPU admission belongs to another scientific commit")
            plan = read_experiment_plan(Path(task["output"]))
            if plan["plan_hash"] != task["plan_hash"]:
                raise RuntimeError("task list differs from frozen science")
            cell = next(c for c in plan["identity"]["cells"] if c["cell_id"] == task["cell_id"])
            config = cell_config(plan["identity"]["supervised_config"], cell["seed"])
            if execution_identity(config, check_environment()) != admission["execution"]:
                raise RuntimeError("allocation differs from admitted numerical environment")
            runtime = load_training_runtime(Path(spec["runtime"]))
            if (
                not runtime.handle_signals
                or runtime.shutdown_margin_seconds != spec["shutdown_margin_seconds"]
            ):
                raise RuntimeError("allocation and signal/shutdown margins differ")
            before = progress(root)
            left = min(
                remaining_seconds(os.environ["SLURM_JOB_ID"]),
                reserved - (time.monotonic() - started),
            )
            budget = left - spec["exit_reserve_seconds"]
            if budget <= runtime.shutdown_margin_seconds:
                raise RuntimeError("startup exhausted the safe work interval")
            runtime = TrainingRuntimeConfig.model_validate(
                {
                    **runtime.model_dump(),
                    "max_walltime_seconds": budget,
                }
            )
            record.update(
                status="running",
                phase="training",
                budget_seconds=budget,
                startup_seconds=time.monotonic() - started,
                progress_before=before,
            )
            save(receipt, record)
            audit = run_experiment_cell(
                Path(task["output"]),
                task["cell_id"],
                repository_root=code,
                runtime=runtime,
                cache=Path(task["cache"]),
                cumulative_budget_seconds=spec["cumulative_budget_seconds"],
            )
            rc = 0 if audit["status"] == "complete" else 75 if audit["status"] == "paused" else 1
            record.update(
                status=audit["status"], exit_code=rc, stop_reason=audit.get("stop_reason")
            )
        except BaseException as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            elapsed = time.monotonic() - started
            ledger["charged_seconds"] -= max(0, reserved - elapsed)
            after = progress(root) if before is not None else None
            ledger["no_progress"] = ledger["no_progress"] + 1 if after == before else 0
            record.update(elapsed_seconds=elapsed, progress_after=after)
            save(ledger_path, ledger)
            save(receipt, record)
        if rc == 75 and can_requeue(ledger, audit, spec):
            record["requeue_requested"] = task_ref
            save(receipt, record)
            # This call may terminate the batch immediately. All durable records precede it.
            subprocess.run(["scontrol", "requeue", task_ref], check=True, timeout=20)
        return rc


if __name__ == "__main__":
    raise SystemExit(main())
