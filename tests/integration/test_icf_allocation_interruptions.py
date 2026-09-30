"""Real batch-process deaths with fake Slurm; no GPU or scheduler required."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

CHILD = """
import importlib.util
import os
import signal
from pathlib import Path

spec = importlib.util.spec_from_file_location('icf_job', os.environ['ICF_JOB_SCRIPT'])
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)
job.remaining_seconds = lambda _: 3600

def environment():
    Path(os.environ['ICF_READY']).write_text('ready')
    signal.pause()

job.check_environment = environment
if os.environ['ICF_KILL_PHASE'] == 'inspection':
    job.progress_signature = lambda _: environment()
job.run_job()
"""


@pytest.mark.skipif(os.name != "posix", reason="POSIX process death")
@pytest.mark.parametrize("kill_phase", ["inspection", "environment"])
def test_repeated_kills_before_first_checkpoint_are_bounded(tmp_path, kill_phase):
    repository = Path(__file__).resolve().parents[2]
    run = tmp_path / "run"
    ready = tmp_path / "ready"
    env = {
        **os.environ,
        "PYTHONPATH": str(repository / "src"),
        "FD_RUN_DIR": str(run),
        "FD_CODE_ROOT": str(repository),
        "FD_MODE": "finance-transformer",
        "FD_MAX_ATTEMPTS": "10",
        "FD_MAX_TOTAL_SECONDS": "100000",
        "FD_MAX_NO_PROGRESS": "2",
        "ICF_JOB_SCRIPT": str(repository / "scripts/icf/job.py"),
        "ICF_READY": str(ready),
        "ICF_KILL_PHASE": kill_phase,
    }
    for attempt in (1, 2):
        env["SLURM_JOB_ID"] = str(attempt)
        with subprocess.Popen(
            [sys.executable, "-c", CHILD],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as process:
            try:
                deadline = time.monotonic() + 30
                while not ready.exists() and process.poll() is None:
                    if time.monotonic() >= deadline:
                        pytest.fail("allocation did not reserve its attempt")
                    time.sleep(0.02)
                assert ready.exists()
                process.kill()
                _, error = process.communicate(timeout=10)
                assert process.returncode == -9, error
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
        ready.unlink()
    env["SLURM_JOB_ID"] = "3"
    env["ICF_KILL_PHASE"] = "environment"
    stopped = subprocess.run(
        [sys.executable, "-c", CHILD],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert stopped.returncode != 0
    assert "no training progress" in stopped.stderr
    assert not ready.exists()
    state = json.loads((run / "allocation_state.json").read_text())
    assert state["attempts"] == 3 and state["no_progress"] == 2
    assert 7200 <= state["charged_seconds"] < 10800 and state["active_attempt"] is None
    assert state["unsettled_attempts"] == []
