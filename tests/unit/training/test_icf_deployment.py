from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from facdigger.training.runtime import snapshot_checksums, write_json

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/icf/job.py"
spec = importlib.util.spec_from_file_location("icf_job", SCRIPT)
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


@pytest.mark.parametrize(
    "value,seconds", [("04:00:00", 14400), ("2-00:00:00", 172800), ("05:30", 330), ("00:00", 0)]
)
def test_slurm_remaining_time(value, seconds):
    assert job.parse_time_left(value) == seconds


@pytest.mark.parametrize("value", ["UNLIMITED", "", "INVALID", "0:90", "1:00\n2:00"])
def test_unknown_time_is_not_an_unlimited_training_budget(value):
    with pytest.raises(ValueError):
        job.parse_time_left(value)


def test_staging_preserves_input_and_checks_all_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write_json(source / "manifest.json", {"dataset_id": "dataset"})
    (source / "features.parquet").write_bytes(b"content")
    inventory = snapshot_checksums(source)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    staged = job.stage_snapshot(source, scratch, tmp_path / "evidence")
    assert snapshot_checksums(staged.path) == inventory
    assert snapshot_checksums(source) == inventory
    assert json.loads(staged.checksums.read_text()) == inventory


@pytest.mark.parametrize(
    "code,reason,requeue",
    [(0, None, False), (1, None, False), (75, "SIGTERM", False), (75, "walltime_budget", True)],
)
def test_job_uses_remaining_budget_and_only_requeues_committed_time_pause(
    tmp_path,
    monkeypatch,
    code,
    reason,
    requeue,
):
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text("handle_signals: true\nshutdown_margin_seconds: 300\n")
    run = tmp_path / "run"
    for key, value in {
        "SLURM_JOB_ID": "42",
        "FD_RUN_DIR": str(run),
        "FD_CODE_ROOT": str(tmp_path),
        "FD_MODE": "finance-transformer",
        "FD_CONFIG": "config.yaml",
        "FD_RUNTIME": str(runtime),
        "FD_DATASET": str(tmp_path / "snapshot"),
        "FD_AUTO_REQUEUE": "1",
        "FD_MIN_OUTPUT_FREE_BYTES": "0",
        "FD_MAX_ATTEMPTS": "3",
        "FD_MAX_TOTAL_SECONDS": "10000",
        "FD_SIGNAL_SECONDS": "300",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("FD_STAGING_ROOT", raising=False)
    remaining = iter([3600, 1800])
    monkeypatch.setattr(job, "remaining_seconds", lambda _: next(remaining))
    monkeypatch.setattr(job, "check_environment", lambda: {"test": True})
    monkeypatch.setattr(
        job,
        "progress_signature",
        lambda _: (
            {"status": "complete" if code == 0 else "paused", "stop_reason": reason},
            [0, 1, 10, "train", 40, None, None],
        ),
    )
    calls = []

    def call(args, **kwargs):
        calls.append(args)
        if args[0] == "srun":
            assert args[-2] == "--run-step"
            generated = Path(args[-1]) / "runtime.json"
            assert json.loads(generated.read_text())["max_walltime_seconds"] == 1800
            assert json.loads((generated.parent / "step.json").read_text())["run_dir"] == str(run)
            return SimpleNamespace(returncode=code)
        assert args == ["scontrol", "requeue", "42"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(job.subprocess, "run", call)
    assert job.run_job() == code
    assert (len(calls) == 2) == requeue
    assert json.loads((run / "allocation_state.json").read_text())["attempts"] == 1


def test_hard_killed_attempt_is_settled_before_another_allocation(tmp_path, monkeypatch):
    run = tmp_path / "run"
    signature = [0, 1, 10, "train", 40, None, None]
    write_json(
        run / "allocation_state.json",
        {
            "attempts": 2,
            "charged_seconds": 7200,
            "no_progress": 1,
            "progress": signature,
            "active_attempt": {
                "number": 2,
                "job_id": "41",
                "progress_before": signature,
                "reserved_seconds": 3600,
            },
        },
    )
    for key, value in {
        "SLURM_JOB_ID": "42",
        "FD_RUN_DIR": str(run),
        "FD_CODE_ROOT": str(tmp_path),
        "FD_MODE": "finance-transformer",
        "FD_MAX_NO_PROGRESS": "2",
        "FD_MAX_ATTEMPTS": "10",
        "FD_MAX_TOTAL_SECONDS": "100000",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(job, "progress_signature", lambda _: ({"status": "running"}, signature))
    monkeypatch.setattr(job, "remaining_seconds", lambda _: 3600)
    monkeypatch.setattr(job.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(job, "check_environment", lambda: pytest.fail("must stop before loading"))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="no training progress"):
            job.run_job()
    state = json.loads((run / "allocation_state.json").read_text())
    assert state["attempts"] == 3
    assert state["charged_seconds"] == 7200
    assert state["no_progress"] == 2
    assert state["active_attempt"] is None


def configure_job(tmp_path, monkeypatch):
    run = tmp_path / "run"
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text("handle_signals: true\nshutdown_margin_seconds: 300\n")
    for key, value in {
        "SLURM_JOB_ID": "42",
        "FD_RUN_DIR": str(run),
        "FD_CODE_ROOT": str(tmp_path),
        "FD_MODE": "finance-transformer",
        "FD_CONFIG": "config.yaml",
        "FD_RUNTIME": str(runtime),
        "FD_DATASET": str(tmp_path / "snapshot"),
        "FD_SIGNAL_SECONDS": "300",
        "FD_AUTO_REQUEUE": "1",
        "FD_MIN_OUTPUT_FREE_BYTES": "0",
        "FD_MAX_ATTEMPTS": "10",
        "FD_MAX_TOTAL_SECONDS": "100000",
        "FD_MAX_NO_PROGRESS": "2",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("FD_STAGING_ROOT", raising=False)
    monkeypatch.setattr(job, "remaining_seconds", lambda _: 3600)
    monkeypatch.setattr(job, "check_environment", lambda: {"test": True})
    snapshot = {"manifest": {"status": "running"}, "signature": [0, 1, 10, "train", 40, None, None]}
    monkeypatch.setattr(
        job, "progress_signature", lambda _: (snapshot["manifest"], snapshot["signature"])
    )
    return run, snapshot


def test_early_failure_refunds_reservation_and_records_error(tmp_path, monkeypatch):
    run, _ = configure_job(tmp_path, monkeypatch)
    now = [100.0]
    monkeypatch.setattr(job.time, "monotonic", lambda: now[0])

    def fail():
        now[0] = 110
        raise ValueError("environment mismatch")

    monkeypatch.setattr(job, "check_environment", fail)
    with pytest.raises(ValueError, match="environment mismatch"):
        job.run_job()
    state = json.loads((run / "allocation_state.json").read_text())
    result = json.loads((run / "allocations/42-1/result.json").read_text())
    assert state["charged_seconds"] == 10
    assert state["attempts"] == 1 and state["active_attempt"] is None
    assert result["status"] == "failed"
    assert result["error"]["phase"] == "environment"


def test_requeue_cannot_precede_durable_settlement(tmp_path, monkeypatch):
    run, snapshot = configure_job(tmp_path, monkeypatch)

    def call(args, **kwargs):
        if args[0] == "srun":
            snapshot["signature"] = [0, 1, 11, "train", 44, None, None]
            snapshot["manifest"] = {"status": "paused", "stop_reason": "walltime_budget"}
            return SimpleNamespace(returncode=75)
        assert args == ["scontrol", "requeue", "42"]
        state = json.loads((run / "allocation_state.json").read_text())
        assert state["active_attempt"] is None and state["no_progress"] == 0
        assert json.loads((run / "allocations/42-1/result.json").read_text())["status"] == "paused"
        # A real requeue can terminate the old batch before scontrol returns.
        raise SystemExit(0)

    monkeypatch.setattr(job.subprocess, "run", call)
    with pytest.raises(SystemExit):
        job.run_job()


@pytest.mark.parametrize("phase", ["market", "probe"])
def test_phase_name_change_alone_is_not_committed_progress(phase):
    before = [0, 1, 10, "probe", None, 8, 2]
    assert not job.progress_advanced(before, [0, 1, 10, phase, None, 8, 2])
    assert job.progress_advanced(before, [0, 1, 10, "epoch_complete", None, 8, 2])
    assert job.progress_advanced(before, [1, 0, 0, None, None, None, None])
    with pytest.raises(RuntimeError, match="regressed"):
        job.progress_advanced(before, [0, 2, 9, "local", None, 1, 0])


def test_running_matrix_and_missing_initial_checkpoint_can_be_observed(tmp_path):
    torch = pytest.importorskip("torch")
    root = tmp_path / "matrix"
    child = root / "runs/fold/pretrain"
    write_json(root / "manifest.json", {"status": "running"})
    write_json(
        root / "matrix.json",
        {
            "stages": [
                {"status": "complete", "run_dir": str(root / "previous")},
                {"status": "running", "run_dir": str(child)},
            ]
        },
    )
    assert job.progress_signature(root)[1] == [1, 0, 0, None, None, None, None]
    (child / "checkpoints").mkdir(parents=True)
    torch.save(
        {
            "epoch": 2,
            "global_step": 11,
            "progress": {
                "phase": "local",
                "local_cursor": 3,
                "market_cursor": 0,
            },
        },
        child / "checkpoints/last.pt",
    )
    assert job.progress_signature(root)[1] == [1, 2, 11, "local", None, 3, 0]
    from facdigger.training.runtime import run_lock

    with (
        run_lock(root / ".research.lock"),
        pytest.raises(RuntimeError, match="already has a writer"),
    ):
        job.progress_signature(root)


@pytest.mark.parametrize("delay,expected", [(120, 80), (201, None)])
def test_job_step_deducts_bootstrap_from_shorter_runtime_budget(
    tmp_path, monkeypatch, delay, expected
):
    from facdigger.training.runtime import TrainingRuntimeConfig

    now = [1000.0]
    monkeypatch.setattr(job.time, "time", lambda: now[0])
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setattr(job, "remaining_seconds", lambda _: 3000)
    write_json(
        tmp_path / "runtime.json",
        TrainingRuntimeConfig(
            max_walltime_seconds=200,
            shutdown_margin_seconds=10,
            handle_signals=True,
        ).model_dump(mode="json"),
    )
    write_json(
        tmp_path / "step.json",
        {
            "mode": "finance-transformer",
            "config": "config.yaml",
            "code": str(tmp_path),
            "run_dir": str(tmp_path / "run"),
            "dataset": str(tmp_path / "snapshot"),
            "budget_sampled_epoch_seconds": 1000,
            "deadline_epoch_seconds": 1200,
        },
    )
    called = []

    def trainer(config, **kwargs):
        called.append(kwargs["control"].config.max_walltime_seconds)

    def imports(mode):
        now[0] += delay
        return trainer, lambda _: {}

    monkeypatch.setattr(job, "load_trainer", imports)
    if expected is None:
        with pytest.raises(ValueError):
            job.run_step(tmp_path)
        assert not called
    else:
        assert job.run_step(tmp_path) == 0
        assert called == [expected]
        assert (
            json.loads((tmp_path / "step_started.json").read_text())["bootstrap_seconds"] == delay
        )


def test_signal_lead_uses_runtime_margin():
    from facdigger.training.runtime import TrainingRuntimeConfig

    assert (
        job.signal_seconds(
            TrainingRuntimeConfig(handle_signals=True, shutdown_margin_seconds=450.1)
        )
        == 451
    )
    with pytest.raises(ValueError, match="positive shutdown margin"):
        job.signal_seconds(TrainingRuntimeConfig(handle_signals=True))


def test_changed_signal_margin_fails_before_starting_training(tmp_path, monkeypatch):
    run, _ = configure_job(tmp_path, monkeypatch)
    monkeypatch.setenv("FD_SIGNAL_SECONDS", "299")
    monkeypatch.setattr(job.subprocess, "run", lambda *a, **k: pytest.fail("must not launch srun"))
    with pytest.raises(ValueError, match="signal lead differs"):
        job.run_job()
    assert json.loads((run / "allocations/42-1/result.json").read_text())["status"] == "failed"


def test_submit_script_derives_signal_from_runtime(tmp_path):
    repository = SCRIPT.parents[2]
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text("handle_signals: true\nshutdown_margin_seconds: 450.1\n")
    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    stub = tmp_path / "sbatch"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    stub.chmod(0o700)
    env_file = tmp_path / "resources.env"
    fields = {
        "FD_CODE_ROOT": str(repository),
        "FD_PYTHON": sys.executable,
        "FD_LOG_ROOT": str(tmp_path / "logs"),
        "FD_RUN_DIR": str(tmp_path / "run"),
        "FD_CONFIG": str(config),
        "FD_RUNTIME": str(runtime),
        "FD_PARTITION": "Teaching",
        "FD_ACCOUNT": "teaching",
        "FD_QOS": "teaching",
        "FD_GRES": "gpu:h200_1g.18gb:1",
        "FD_CPUS": "4",
        "FD_MEMORY": "32G",
        "FD_TIME": "12:00:00",
    }
    env_file.write_text("".join(f"{key}={shlex.quote(value)}\n" for key, value in fields.items()))
    result = subprocess.run(
        ["bash", str(SCRIPT.with_name("submit.sh")), str(env_file), "--test-only"],
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "PYTHONPATH": str(repository / "src"),
        },
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--signal=USR1@451" in result.stdout.splitlines()
    assert "--test-only" in result.stdout.splitlines()
