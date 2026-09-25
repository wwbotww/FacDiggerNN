from __future__ import annotations

import importlib.util
import json
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
            {"status": "paused", "stop_reason": reason},
            [0, 1, 10, "train", 40, None, None],
        ),
    )
    calls = []

    def call(args, **kwargs):
        calls.append(args)
        if args[0] == "srun":
            generated = Path(args[args.index("--runtime") + 1])
            assert json.loads(generated.read_text())["max_walltime_seconds"] == 1800
            assert "--run-dir" in args
            return SimpleNamespace(returncode=code)
        assert args == ["scontrol", "requeue", "42"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(job.subprocess, "run", call)
    assert job.run_job() == code
    assert (len(calls) == 2) == requeue
    assert json.loads((run / "allocation_state.json").read_text())["attempts"] == 1
