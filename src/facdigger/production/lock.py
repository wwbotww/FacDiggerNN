"""Kernel-released single-process lock for the Linux production container."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path


class ProductionAlreadyRunning(RuntimeError):
    """Raised when another process owns the production service lock."""


class ProductionLock:
    def __init__(self, state_database: str | Path) -> None:
        database = Path(state_database).resolve()
        self.path = database.with_suffix(database.suffix + ".lock")
        self._descriptor: int | None = None

    def __enter__(self) -> ProductionLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise ProductionAlreadyRunning(
                f"production lock is owned by another process: {self.path}"
            ) from exc
        self._descriptor = descriptor
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode())
        os.fsync(descriptor)
        return self

    def __exit__(self, *_: object) -> None:
        if self._descriptor is None:
            return
        fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        os.close(self._descriptor)
        self._descriptor = None
