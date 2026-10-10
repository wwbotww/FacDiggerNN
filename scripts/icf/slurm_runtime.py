"""Current-allocation identity and finite budgets, shared by ICF launchers only."""

from __future__ import annotations

import math
import os
import re
import subprocess
import time


def current_task_ref() -> str:
    """Use an explicit array element even when its raw ID equals the array ID."""
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[1-9][0-9]*", job_id):
        raise ValueError("a positive SLURM_JOB_ID is required")
    array_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_id is None and task_id is None:
        return job_id
    if (
        array_id is None or not re.fullmatch(r"[1-9][0-9]*", array_id)
        or task_id is None or not re.fullmatch(r"0|[1-9][0-9]*", task_id)
    ):
        raise ValueError("Slurm array identity requires both a valid array ID and task index")
    return f"{array_id}_{task_id}"


def parse_time_left(value: str) -> int:
    match = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)", value.strip())
    if match is None:
        raise ValueError(f"cannot determine finite Slurm time remaining: {value!r}")
    days, hours, minutes, seconds = (int(item or 0) for item in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError("invalid Slurm time remaining")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def remaining_seconds(job_id: str) -> int:
    """Read only this allocation's budget; never infer one from sibling jobs."""
    if job_id != os.environ.get("SLURM_JOB_ID"):
        raise ValueError("remaining time must belong to the current SLURM_JOB_ID")
    task_ref = current_task_ref()
    started = time.monotonic()
    result = subprocess.run(
        ["squeue", "-h", "-r", "-j", task_ref, "-o", "%A|%i|%L"],
        check=True, text=True, capture_output=True, timeout=15,
    )
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1:
        raise ValueError(f"expected one Slurm record for {task_ref}, received {len(rows)}")
    fields = [field.strip() for field in rows[0].split("|")]
    if len(fields) != 3 or fields[:2] != [job_id, task_ref]:
        raise ValueError(f"Slurm record identity differs from allocation {job_id}/{task_ref}")
    # The scheduler sampled before the response arrived. Charge the whole query
    # conservatively; waiting for Slurm must never extend the training deadline.
    elapsed = math.ceil(time.monotonic() - started)
    return max(0, parse_time_left(fields[2]) - elapsed)
