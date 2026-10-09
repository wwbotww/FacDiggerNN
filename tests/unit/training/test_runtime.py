from __future__ import annotations

import json
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.training.runtime import (
    TrainingControl,
    TrainingRuntimeConfig,
    load_training_runtime,
    resolve_dataset,
    run_lock,
    save_checkpoint,
    snapshot_checksums,
    write_json,
)


def test_runtime_budget_does_not_reset_between_stages():
    now = [10.0]
    config = TrainingRuntimeConfig(max_walltime_seconds=100, shutdown_margin_seconds=20)
    control = TrainingControl(config, clock=lambda: now[0])
    now[0] += 79
    assert control.stop_reason() is None
    now[0] += 1
    assert control.stop_reason() == "walltime_budget"
    control.request_stop("SIGTERM")
    assert control.stop_reason() == "SIGTERM"
    with pytest.raises(ValueError, match="margin"):
        TrainingRuntimeConfig(max_walltime_seconds=10, shutdown_margin_seconds=10)


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="POSIX signals")
def test_signal_handlers_are_scoped_and_term_overrides_requeue_reason():
    original = signal.getsignal(signal.SIGUSR1)
    with TrainingControl(TrainingRuntimeConfig(handle_signals=True)) as control:
        signal.raise_signal(signal.SIGUSR1)
        assert control.reason == "time_limit_warning"
        signal.raise_signal(signal.SIGTERM)
        assert control.reason == "SIGTERM"
    assert signal.getsignal(signal.SIGUSR1) == original


def test_run_lock_rejects_second_writer_and_releases_on_exception(tmp_path):
    path = tmp_path / "run.lock"
    with pytest.raises(LookupError), run_lock(path):
        with pytest.raises(RuntimeError, match="already has a writer"), run_lock(path):
            pass
        raise LookupError
    with run_lock(path):
        assert path.exists()


def test_short_shared_lock_serializes_concurrent_writers_without_losing_updates(tmp_path):
    path = tmp_path / "shared.lock"
    value = tmp_path / "counter"
    value.write_text("0")
    start = Barrier(8)

    def increment(_):
        start.wait(timeout=5)
        with run_lock(path, timeout_seconds=5):
            previous = int(value.read_text())
            time.sleep(0.01)
            value.write_text(str(previous + 1))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(increment, range(8)))
    assert value.read_text() == "8"
    assert path.is_file()


def test_shared_lock_timeout_does_not_enter_or_steal_the_existing_lock(tmp_path):
    path = tmp_path / "shared.lock"
    with run_lock(path):
        inode = path.stat().st_ino
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="already has a writer"):
            with run_lock(path, timeout_seconds=0.1):
                pytest.fail("occupied critical section entered")
        assert time.monotonic() - started >= 0.09
        assert path.stat().st_ino == inode
        with pytest.raises(RuntimeError, match="already has a writer"), run_lock(path):
            pass
    with run_lock(path):
        pass


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan")])
def test_shared_lock_rejects_unbounded_or_negative_waits(tmp_path, timeout):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        with run_lock(tmp_path / "lock", timeout_seconds=timeout):
            pass
    assert not (tmp_path / "lock").exists()


def test_windows_lock_waits_for_contention_and_releases(tmp_path, monkeypatch):
    import errno
    import sys
    from types import SimpleNamespace

    from facdigger.training import runtime

    clock = [0.0]
    calls = []

    def locking(fd, mode, count):
        calls.append(mode)
        if len(calls) <= 2:
            raise OSError(errno.EACCES, "lock busy")

    monkeypatch.setattr(runtime, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(runtime, "time", SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda delay: clock.__setitem__(0, clock[0] + delay),
    ))
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(
        locking=locking, LK_NBLCK=1, LK_UNLCK=2,
    ))
    with pytest.raises(LookupError), run_lock(tmp_path / "lock", timeout_seconds=1):
        raise LookupError
    assert calls == [1, 1, 1, 2] and clock[0] == pytest.approx(0.1)


def test_relocation_checks_data_files_not_only_manifest(tmp_path):
    source = tmp_path / "snapshot"
    source.mkdir()
    write_json(source / "manifest.json", {"dataset_id": "fixture"})
    (source / "features.parquet").write_bytes(b"original")
    checksums = tmp_path / "checksums.json"
    write_json(checksums, snapshot_checksums(source))
    runtime_file = tmp_path / "runtime.yaml"
    runtime_file.write_text(
        "dataset_overrides:\n  fixture:\n    path: snapshot\n    checksums: checksums.json\n"
    )
    runtime = load_training_runtime(runtime_file)
    digest = sha256_file(source / "manifest.json")
    assert (
        resolve_dataset(tmp_path / "offline", runtime, dataset_id="fixture", manifest_hash=digest)
        == source
    )
    (source / "features.parquet").write_bytes(b"corrupted")
    with pytest.raises(DataContractError, match="checksums"):
        resolve_dataset(source, runtime, dataset_id="fixture", manifest_hash=digest)
    assert json.loads(checksums.read_text())["manifest.json"] == digest


def test_interrupted_serialization_preserves_the_committed_checkpoint(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    path = tmp_path / "last.pt"
    save_checkpoint(path, {"step": 12})

    def interrupted(payload, stream):
        stream.write(b"partial write")
        raise OSError("storage write interrupted")

    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        save_checkpoint(path, {"step": 13})
    assert torch.load(path, weights_only=False) == {"step": 12}


@pytest.mark.parametrize(
    "field", ["checkpoint_interval_seconds", "max_walltime_seconds", "shutdown_margin_seconds"]
)
def test_runtime_rejects_nonfinite_time(field):
    with pytest.raises(ValueError):
        TrainingRuntimeConfig(**{field: float("inf")})
