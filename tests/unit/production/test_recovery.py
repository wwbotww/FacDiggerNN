"""Operator resume is an audited scheduling transition, never a second producer."""

import sqlite3
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from test_config_state import _config

from facdigger.data.contracts import DataContractError
from facdigger.production import recovery
from facdigger.production.lock import ProductionAlreadyRunning, ProductionLock
from facdigger.production.state import ProductionState

DAY = date(2026, 9, 23)
NOW = datetime(2026, 9, 24, 8, tzinfo=timezone.utc)
CUTOFF = datetime(2026, 9, 24, 13, 30, tzinfo=timezone.utc)


@pytest.fixture
def resume_case(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(recovery, "load_model_release", lambda _: SimpleNamespace(
        release_id=config.model.release_id,
    ))
    monkeypatch.setattr(recovery, "load_current_revision", lambda _: None)
    with ProductionState(config.state_database) as state:
        record = state.put(DAY, config.model.release_id, "blocked", attempts=3,
                           error="DataContractError: identity changed",
                           quality_reference={"security_ids": ["A", "B"]})
    return config, record


def _resume(case, *, now=NOW, **kwargs):
    config, record = case
    return recovery.resume_blocked_production(
        config, target_date=DAY, expected_updated_at=record.updated_at,
        reason="Verified repair", now_provider=lambda: now, **kwargs,
    )


def test_resume_is_durable_idempotent_and_keeps_previous_failure(resume_case):
    config, previous = resume_case
    assert _resume(resume_case)["action"] == "resume_scheduled"
    assert _resume(resume_case, now=NOW + timedelta(minutes=1))["action"] == "already_scheduled"
    with ProductionState(config.state_database) as state:
        record = state.get(DAY)
        assert record.status == "waiting_data"
        assert record.attempts == previous.attempts
        assert record.error == previous.error
        assert record.quality_reference == previous.quality_reference
        assert record.next_retry_at == NOW
    with sqlite3.connect(config.state_database) as db:
        rows = db.execute("SELECT * FROM production_resume_events").fetchall()
    assert len(rows) == 1 and rows[0][4:] == (previous.error, 3, "Verified repair")


def test_hard_exit_after_requeue_is_idempotent_after_restart(resume_case):
    config, record = resume_case
    script = """
import os, sys
from datetime import date, datetime
from facdigger.production.state import ProductionState
path, expected, observed, cutoff = sys.argv[1:]
with ProductionState(path) as state:
    state.schedule_resume(date(2026, 9, 23), '1' * 64, expected_updated_at=expected,
                          reason='Verified repair', observed=datetime.fromisoformat(observed),
                          cutoff_at=datetime.fromisoformat(cutoff))
os._exit(73)
"""
    result = subprocess.run([sys.executable, "-c", script, str(config.state_database),
                             record.updated_at, NOW.isoformat(), CUTOFF.isoformat()],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 73, result.stderr
    assert _resume(resume_case)["action"] == "already_scheduled"
    with sqlite3.connect(config.state_database) as db:
        assert db.execute("SELECT COUNT(*) FROM production_resume_events").fetchone()[0] == 1


@pytest.mark.parametrize("status", ["running", "published", "expired", "waiting_data"])
def test_resume_rejects_other_states_without_resetting_them(resume_case, status):
    config, _ = resume_case
    with ProductionState(config.state_database) as state:
        record = state.put(DAY, config.model.release_id, status, attempts=8)
    with pytest.raises((ValueError, DataContractError)):
        _resume((config, record))
    with ProductionState(config.state_database) as state:
        assert state.get(DAY) == record


@pytest.mark.parametrize("now", [CUTOFF, CUTOFF + timedelta(days=1),
                                 datetime(2026, 9, 23, 22, tzinfo=timezone.utc)])
def test_resume_never_moves_the_original_window(resume_case, now):
    config, record = resume_case
    with pytest.raises(DataContractError, match="original open publication window"):
        _resume(resume_case, now=now)
    with ProductionState(config.state_database) as state:
        assert state.get(DAY) == record


def test_resume_holds_same_lock_as_the_service(resume_case):
    with ProductionLock(resume_case[0].state_database):
        with pytest.raises(ProductionAlreadyRunning):
            _resume(resume_case)


def test_resume_compare_and_set_and_release_pin(resume_case):
    config, record = resume_case
    with ProductionState(config.state_database) as state:
        state.put(DAY, config.model.release_id, "blocked", attempts=4, error="a new failure")
    with pytest.raises(DataContractError, match="changed since inspection"):
        _resume(resume_case)
    config = config.model_copy(update={"model": config.model.model_copy(
        update={"release_id": "2" * 64},
    )})
    with pytest.raises(DataContractError, match="fixed release"):
        _resume((config, record))


def test_audit_insert_failure_rolls_back_requeue(resume_case):
    config, record = resume_case
    with sqlite3.connect(config.state_database) as db:
        db.execute("""CREATE TRIGGER abort_resume BEFORE INSERT ON production_resume_events
                      BEGIN SELECT RAISE(ABORT, 'injected audit failure'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="injected audit failure"):
        _resume(resume_case)
    with ProductionState(config.state_database) as state:
        assert state.get(DAY) == record


@pytest.mark.parametrize("damage", ["source", "release", "delivery"])
def test_resume_does_not_unblock_corrupt_inputs(resume_case, monkeypatch, damage):
    config, record = resume_case
    if damage == "delivery":
        path = config.factor_batch.output_root / "broken"
        path.mkdir(parents=True)
        (path / "manifest.json").write_text("not JSON")
    else:
        def broken(*_):
            raise DataContractError("corrupt artifact")
        monkeypatch.setattr(recovery, "load_current_revision" if damage == "source"
                            else "load_model_release", broken)
    with pytest.raises(DataContractError):
        _resume(resume_case)
    with ProductionState(config.state_database) as state:
        assert state.get(DAY) == record


def test_slow_preflight_crossing_cutoff_cannot_requeue(resume_case):
    config, record = resume_case
    clock = iter([NOW, CUTOFF])
    with pytest.raises(ValueError, match="original cutoff"):
        recovery.resume_blocked_production(config, target_date=DAY,
                                          expected_updated_at=record.updated_at, reason="test",
                                          now_provider=lambda: next(clock))
    with ProductionState(config.state_database) as state:
        assert state.get(DAY) == record
