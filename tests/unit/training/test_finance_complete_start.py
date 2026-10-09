from __future__ import annotations

import errno
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/icf/finance_complete_start.py"
spec = importlib.util.spec_from_file_location("complete_start", SCRIPT)
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


def fixture(tmp_path):
    root = tmp_path / "cells/wf/seed-42/finance"
    root.mkdir(parents=True)
    (root / ".training.lock").touch()
    now, retries = [0.0], []
    kwargs = {
        "output": tmp_path, "cell_id": "wf/seed-42/finance", "on_retry": retries.append,
        "clock": lambda: now[0], "sleep": lambda delay: now.__setitem__(0, now[0] + delay),
    }
    return root, kwargs, now, retries


def contention(path, cause=None):
    raise RuntimeError(f"training run already has a writer: {path}") from (
        cause if cause is not None else BlockingIOError(errno.EAGAIN, "busy")
    )


def test_frozen_entry_retries_only_before_state_and_resamples_the_call(tmp_path):
    root, kwargs, now, retries = fixture(tmp_path)
    samples = []

    def run():
        samples.append(100 - now[0])
        if len(samples) <= 2:
            contention(tmp_path / ".plan.lock")
        (root / "cell.json").write_text('{"status":"paused"}')
        return 75

    assert startup.run_with_startup_retry(run, **kwargs) == 75
    assert samples == [100, 98, 96]
    assert len(retries) == 2 and (root / ".training.lock").exists()


@pytest.mark.parametrize("artifact", ["cell.json", "checkpoints", "progress.jsonl", "parent"])
def test_any_committed_or_unknown_cell_state_forbids_automatic_retry(tmp_path, artifact):
    root, kwargs, _, retries = fixture(tmp_path)
    (root / artifact).touch()
    with pytest.raises(RuntimeError):
        startup.run_with_startup_retry(lambda: contention(tmp_path / ".plan.lock"), **kwargs)
    assert retries == []


@pytest.mark.parametrize("failure", ["cell_lock", "other_plan", "wrong_cause", "other_error"])
def test_duplicate_writers_and_unrelated_failures_are_not_retried(tmp_path, failure):
    root, kwargs, _, retries = fixture(tmp_path)

    def run():
        if failure == "cell_lock":
            contention(root / ".training.lock")
        if failure == "other_plan":
            contention(tmp_path / "another/.plan.lock")
        if failure == "wrong_cause":
            contention(tmp_path / ".plan.lock", ValueError("different failure"))
        raise ValueError("bad data")

    with pytest.raises((RuntimeError, ValueError)):
        startup.run_with_startup_retry(run, **kwargs)
    assert retries == []


def test_startup_wait_has_both_time_and_retry_limits(tmp_path):
    _, kwargs, now, retries = fixture(tmp_path)
    def run():
        contention(tmp_path / ".plan.lock")

    with pytest.raises(RuntimeError):
        startup.run_with_startup_retry(run, timeout_seconds=0.5, **kwargs)
    assert now[0] == 0.5 and len(retries) == 1
    now[0] = 0
    retries.clear()
    with pytest.raises(RuntimeError):
        startup.run_with_startup_retry(run, max_retries=2, **kwargs)
    assert now[0] == 4 and len(retries) == 2


def test_cancel_during_wait_is_not_retried(tmp_path):
    _, kwargs, _, retries = fixture(tmp_path)

    def cancel(_):
        raise KeyboardInterrupt

    kwargs["sleep"] = cancel
    with pytest.raises(KeyboardInterrupt):
        startup.run_with_startup_retry(lambda: contention(tmp_path / ".plan.lock"), **kwargs)
    assert len(retries) == 1
