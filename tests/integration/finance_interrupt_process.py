"""Child process used by actual SIGKILL/SIGTERM recovery tests."""

from __future__ import annotations

import json
import signal
import sys
import threading
from pathlib import Path

from test_finance_interruptions import _run

from facdigger.training.runtime import TrainingControl, TrainingPaused, TrainingRuntimeConfig


def main():
    kind, mechanism, destination = sys.argv[1:]
    root = Path(destination)
    root.mkdir(parents=True)
    # SIGKILL leaves running state; no Python exception handler can update it.
    (root / "manifest.json").write_text(json.dumps({"status": "running"}))
    with TrainingControl(
        TrainingRuntimeConfig(
            checkpoint_interval_seconds=1e-9,
            handle_signals=True,
        )
    ) as control:

        def interrupt(event):
            if event["event"] == "checkpoint_saved" and event["global_step"] == 1:
                if mechanism == "kill":
                    (root / "ready").write_text("committed")
                    threading.Event().wait(60)
                    raise RuntimeError("parent did not kill the test process")
                signal.raise_signal(signal.SIGTERM)

        try:
            _run(kind, root / "checkpoints", control=control, callback=interrupt)
        except TrainingPaused as exc:
            (root / "manifest.json").write_text(
                json.dumps({"status": "paused", "reason": exc.reason})
            )
            return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
