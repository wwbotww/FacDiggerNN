from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def adapter(monkeypatch):
    directory = Path(__file__).resolve().parents[3] / "scripts/icf"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "icf_interventions", directory / "finance_interventions.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "reason,allowed",
    [
        ("walltime_budget", True),
        ("time_limit_warning", True),
        ("SIGTERM", False),
        ("cumulative_budget", False),
        ("corruption", False),
    ],
)
def test_requeue_is_limited_to_safely_paused_time_limits(adapter, reason, allowed):
    ledger = {"attempts": 1, "no_progress": 0, "charged_seconds": 3600}
    spec = {
        "max_no_progress": 2,
        "max_attempts": 6,
        "shutdown_margin_seconds": 900,
        "exit_reserve_seconds": 300,
        "cumulative_budget_seconds": 518400,
    }
    audit = {"status": "paused", "stop_reason": reason}
    assert adapter.can_requeue(ledger, audit, spec) is allowed
    for key, value in (("attempts", 6), ("no_progress", 2), ("charged_seconds", 517201)):
        assert not adapter.can_requeue({**ledger, key: value}, audit, spec)
    assert not adapter.can_requeue(ledger, {**audit, "status": "failed"}, spec)


def test_committed_observation_chunks_count_as_progress_before_first_update(adapter, tmp_path):
    before = adapter.progress(tmp_path)
    directory = tmp_path / "chunks/epoch-0-F-0000"
    directory.mkdir(parents=True)
    (directory / "scores.tmp").write_text("unfinished")
    assert adapter.progress(tmp_path) == before
    (directory / "complete.json").write_text(json.dumps({"files": {}}))
    assert adapter.progress(tmp_path) != before
