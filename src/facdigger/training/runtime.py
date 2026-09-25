"""Optional, scheduler-independent controls for offline training runs."""

from __future__ import annotations

import copy
import errno
import json
import os
import signal
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from facdigger.data.config import StrictModel
from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file


class DatasetLocation(StrictModel):
    path: Path
    checksums: Path


class TrainingRuntimeConfig(StrictModel):
    checkpoint_interval_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_walltime_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    shutdown_margin_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    handle_signals: bool = False
    dataset_overrides: dict[str, DatasetLocation] = Field(default_factory=dict)
    fold_snapshots: dict[str, DatasetLocation] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_budget(self) -> TrainingRuntimeConfig:
        if self.max_walltime_seconds is not None and (
            self.shutdown_margin_seconds >= self.max_walltime_seconds
        ):
            raise ValueError("shutdown margin must be smaller than walltime budget")
        return self


def load_training_runtime(path: str | Path | None) -> TrainingRuntimeConfig:
    if path is None:
        return TrainingRuntimeConfig()
    source = Path(path).resolve()
    config = TrainingRuntimeConfig.model_validate(
        yaml.safe_load(source.read_text(encoding="utf-8"))
    )
    for location in [*config.dataset_overrides.values(), *config.fold_snapshots.values()]:
        location.path = (source.parent / location.path).resolve()
        location.checksums = (source.parent / location.checksums).resolve()
    return config


def snapshot_checksums(root: Path) -> dict[str, str]:
    """Inventory an immutable snapshot, including all its actual data files."""
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def write_snapshot_checksums(root: Path, output: Path) -> None:
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("checksum inventory must be outside the immutable snapshot")
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError("snapshot manifest is missing")
    write_json(output, snapshot_checksums(root))


def resolve_dataset(
    path: Path,
    runtime: TrainingRuntimeConfig,
    *,
    dataset_id: str | None = None,
    manifest_hash: str | None = None,
) -> Path:
    if dataset_id is None:
        dataset_id = str(
            json.loads((path / "manifest.json").read_text(encoding="utf-8"))["dataset_id"]
        )
    override = runtime.dataset_overrides.get(dataset_id)
    result = override.path if override else path
    if override:
        expected = json.loads(override.checksums.read_text(encoding="utf-8"))
        if not isinstance(expected, dict) or "manifest.json" not in expected:
            raise DataContractError("snapshot checksums must include manifest.json")
        if snapshot_checksums(result) != expected:
            raise DataContractError("relocated snapshot files differ from checksums")
        if manifest_hash is None and (path / "manifest.json").is_file():
            manifest_hash = sha256_file(path / "manifest.json")
    manifest = result / "manifest.json"
    if manifest_hash is not None and sha256_file(manifest) != manifest_hash:
        raise DataContractError("snapshot manifest differs from bound training input")
    if json.loads(manifest.read_text(encoding="utf-8"))["dataset_id"] != dataset_id:
        raise DataContractError("snapshot dataset_id does not match")
    return result.resolve()


def _sync_directory(path: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            try:
                os.fsync(descriptor)
            except OSError as exc:
                # Some local/network filesystems do not implement directory fsync.
                # File fsync + atomic rename still protect against process crashes.
                if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
                    raise
        finally:
            os.close(descriptor)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _sync_directory(path.parent)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _sync_directory(path.parent)


def checkpoint_copy(value: Any) -> Any:
    """Detach a selected checkpoint from live tensors without using training RNG."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: checkpoint_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [checkpoint_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(checkpoint_copy(item) for item in value)
    return copy.deepcopy(value)


@contextmanager
def run_lock(path: Path) -> Iterator[None]:
    """Hold an OS lock; never unlink it or steal it based on a remote PID."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            if path.stat().st_size == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"training run already has a writer: {path}") from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"training run already has a writer: {path}") from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class TrainingPaused(Exception):
    def __init__(self, reason: str, checkpoint: Path) -> None:
        self.reason = reason
        self.checkpoint = checkpoint
        super().__init__(f"training paused ({reason}); resume from {checkpoint}")


class TrainingControl:
    def __init__(
        self,
        config: TrainingRuntimeConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or TrainingRuntimeConfig()
        self.clock = clock
        self.started = clock()
        self.last_saved = self.started
        self.reason: str | None = None
        self._handlers: dict[int, Any] = {}

    @property
    def enabled(self) -> bool:
        return bool(
            self.config.checkpoint_interval_seconds
            or self.config.max_walltime_seconds
            or self.config.handle_signals
        )

    def __enter__(self) -> TrainingControl:
        if self.config.handle_signals:
            if not hasattr(signal, "SIGUSR1"):
                raise ValueError("signal controls require SIGUSR1; use walltime on this platform")
            for sig in (signal.SIGTERM, signal.SIGUSR1):
                self._handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._signal)
        return self

    def __exit__(self, *args: Any) -> None:
        for sig, handler in self._handlers.items():
            signal.signal(sig, handler)
        self._handlers.clear()

    def _signal(self, signum: int, frame: Any) -> None:
        # SIGTERM must never be mistaken for permission to requeue a cancelled job.
        self.request_stop("time_limit_warning" if signum == signal.SIGUSR1 else "SIGTERM")

    def request_stop(self, reason: str = "requested") -> None:
        if self.reason is None or reason == "SIGTERM":
            self.reason = reason

    def stop_reason(self) -> str | None:
        budget = self.config.max_walltime_seconds
        if (
            self.reason is None
            and budget is not None
            and (self.clock() - self.started >= budget - self.config.shutdown_margin_seconds)
        ):
            self.reason = "walltime_budget"
        return self.reason

    def checkpoint_due(self) -> bool:
        interval = self.config.checkpoint_interval_seconds
        return bool(
            self.stop_reason()
            or (interval is not None and self.clock() - self.last_saved >= interval)
        )

    def saved(self) -> None:
        self.last_saved = self.clock()

    def raise_if_stopping(self, checkpoint: Path) -> None:
        reason = self.stop_reason()
        if reason is not None:
            if not checkpoint.is_file():
                raise RuntimeError("cannot pause without a committed checkpoint")
            raise TrainingPaused(reason, checkpoint)


def remaining_loader(loader: Any, sampler: Any, cursor: int) -> Any:
    """Recreate an iterator without consuming training RNG a second time.

    The sampler skips indices, not dataset reads. Only num_workers=0 is supported
    for mid-epoch recovery; prefetch/worker RNG is deliberately not guessed.
    """
    import torch

    sampler.start_batch = cursor
    if cursor:
        state = torch.get_rng_state()
        try:
            return iter(loader)
        finally:
            torch.set_rng_state(state)
    return iter(loader)
