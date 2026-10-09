"""Bounded startup recovery for a matrix pinned to the older non-waiting plan lock.

This adapter calls the unchanged scientific entry again only if it has not written
any cell state. New matrices should use the native bounded plan-lock wait instead.
"""

from __future__ import annotations

import math
import time
from pathlib import Path


def run_with_startup_retry(
    run, *, output: Path, cell_id: str, on_retry,
    timeout_seconds: float = 120, max_retries: int = 30,
    clock=time.monotonic, sleep=time.sleep,
):
    """Retry only confirmed plan-lock contention before any data/model state exists."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or max_retries < 0:
        raise ValueError("startup retries require finite positive time and a nonnegative count")
    root = (output / "cells" / cell_id).resolve()
    cells_root = (output / "cells").resolve()
    if not root.is_relative_to(cells_root) or root == cells_root:
        raise ValueError("cell must be inside the experiment cells directory")
    expected = f"training run already has a writer: {output / '.plan.lock'}"
    started = clock()
    retries = 0
    while True:
        try:
            # The caller re-samples actual allocation time before EVERY invocation.
            return run()
        except RuntimeError as exc:
            if (
                str(exc) != expected
                or not isinstance(exc.__cause__, BlockingIOError)
                or not root.is_dir()
                or set(p.name for p in root.iterdir()) != {".training.lock"}
            ):
                raise
            remaining = timeout_seconds - (clock() - started)
            if remaining <= 0 or retries >= max_retries:
                raise
            retries += 1
            delay = min(2.0, remaining)
            on_retry({
                "event": "plan_lock_startup_retry", "retry": retries,
                "elapsed_seconds": clock() - started, "sleep_seconds": delay,
            })
            sleep(delay)
            if clock() - started >= timeout_seconds:
                raise
