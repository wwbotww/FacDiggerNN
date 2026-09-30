"""Observational phase timing for the finance trainers, never recovery state."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from facdigger.training.resources import process_memory


def append_progress(path: Path, event: dict[str, Any], *, attempt: int) -> None:
    payload = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "process_id": os.getpid(),
        "attempt": attempt,
        **event,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


class TrainingProgress:
    def __init__(
        self,
        callback: Callable[[dict[str, Any]], None] | None,
        *,
        clock: Callable[[], float] = time.perf_counter,
        interval_seconds: float = 60,
    ) -> None:
        self.callback = callback
        self.clock = clock
        self.interval = interval_seconds

    @contextmanager
    def phase(self, name: str, **metadata: Any) -> Iterator[Callable[..., None]]:
        started = self.clock()
        last_report: float | None = None
        values: dict[str, Any] = {}

        def emit(event: str, **extra: Any) -> None:
            if self.callback is not None:
                self.callback(
                    {
                        "event": event,
                        "phase": name,
                        "phase_elapsed_seconds": self.clock() - started,
                        **metadata,
                        **values,
                        **extra,
                        **process_memory(),
                    }
                )

        def update(**progress: Any) -> None:
            nonlocal last_report
            changed_phase = progress.get("subphase") != values.get("subphase")
            values.update(progress)
            if self.callback is None:
                return
            now = self.clock()
            if last_report is None or changed_phase or now - last_report >= self.interval:
                emit("phase_progress")
                last_report = now

        emit("phase_started")
        try:
            yield update
        except BaseException as exc:
            emit("phase_interrupted", error_type=type(exc).__name__)
            raise
        else:
            emit("phase_completed")
