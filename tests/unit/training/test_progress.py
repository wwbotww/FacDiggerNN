from __future__ import annotations

import pytest

from facdigger.training.progress import TrainingProgress
from facdigger.training.runtime import TrainingPaused


def test_phase_progress_is_rate_limited_but_reports_subphase_changes(monkeypatch):
    monkeypatch.setattr("facdigger.training.progress.process_memory", lambda: {})
    now = [0.0]
    events = []
    observer = TrainingProgress(events.append, clock=lambda: now[0])
    with observer.phase("probe", epoch=3) as update:
        update(subphase="fit", completed=1, total=10)
        now[0] = 30
        update(subphase="fit", completed=2, total=10)
        now[0] = 60
        update(subphase="fit", completed=3, total=10)
        update(subphase="selection", completed=1, total=5)
        update(subphase="selection", completed=5, total=5)
    assert [event["event"] for event in events] == [
        "phase_started",
        "phase_progress",
        "phase_progress",
        "phase_progress",
        "phase_completed",
    ]
    assert events[-1]["completed"] == 5
    assert events[-1]["phase_elapsed_seconds"] == 60
    assert all(event["epoch"] == 3 for event in events)


def test_interruption_never_reports_a_completed_phase(tmp_path):
    events = []
    observer = TrainingProgress(events.append)
    with pytest.raises(TrainingPaused), observer.phase("selection"):
        raise TrainingPaused("walltime_budget", tmp_path / "last.pt")
    assert [event["event"] for event in events] == ["phase_started", "phase_interrupted"]
    assert events[-1]["error_type"] == "TrainingPaused"


def test_disabled_observation_does_not_collect_resources(monkeypatch):
    monkeypatch.setattr(
        "facdigger.training.progress.process_memory", lambda: pytest.fail("observer is disabled")
    )
    with TrainingProgress(None).phase("train") as update:
        update(completed=1)
