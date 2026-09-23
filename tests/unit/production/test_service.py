from datetime import datetime, timedelta, timezone

import pytest

from facdigger.production.config import ProductionServiceConfig
from facdigger.production.service import production_health
from facdigger.production.state import ProductionState


def _config(tmp_path) -> ProductionServiceConfig:
    return ProductionServiceConfig.model_validate(
        {
            "data": {
                "provider_config": tmp_path / "provider.yaml",
                "store_root": tmp_path / "store",
                "bootstrap_source": tmp_path / "bronze",
            },
            "model": {
                "release_root": tmp_path / "releases",
                "release_id": "1" * 64,
            },
            "inference": {"output_root": tmp_path / "inference"},
            "factor_batch": {"output_root": tmp_path / "factors"},
            "state_database": tmp_path / "production.sqlite3",
            "poll_seconds": 60,
        }
    )


def test_health_requires_recent_heartbeat(tmp_path) -> None:
    config = _config(tmp_path)
    now = datetime(2026, 8, 13, 0, 0, tzinfo=timezone.utc)

    assert production_health(config, now=now)["healthy"] is False
    with ProductionState(config.state_database) as state:
        state.heartbeat(at=now - timedelta(seconds=120), phase="sleeping")
    assert production_health(config, now=now)["healthy"] is True
    with ProductionState(config.state_database) as state:
        state.heartbeat(at=now - timedelta(seconds=181), phase="sleeping")
    assert production_health(config, now=now)["healthy"] is False


def test_failed_day_does_not_make_live_service_unhealthy(tmp_path):
    from datetime import date

    config = _config(tmp_path)
    now = datetime(2026, 8, 18, 14, tzinfo=timezone.utc)
    with ProductionState(config.state_database) as state:
        state.put(date(2026, 8, 17), config.model.release_id, "expired", attempts=4,
                  quality_report={"status": "insufficient"})
        state.heartbeat(at=now, phase="sleeping")
    health = production_health(config, now=now)
    assert health["healthy"] is True
    assert health["production"] == {
        "target_date": "2026-08-17", "status": "expired", "quality": "insufficient",
    }


def test_service_continues_after_waiting_and_blocked_ticks(tmp_path, monkeypatch):
    import threading

    from facdigger.production.runner import ProductionTickResult
    from facdigger.production.service import serve_production

    config = _config(tmp_path)
    stopped = threading.Event()
    observed = []

    def tick(config, stopped, now):
        action = ["waiting_data", "blocked", "expired"][len(observed)]
        observed.append(action)
        if len(observed) == 3:
            stopped.set()
        return ProductionTickResult(action, now.date(), "due", 1)

    monkeypatch.setattr("facdigger.production.service._run_tick_with_heartbeat", tick)
    monkeypatch.setattr("facdigger.production.service._heartbeat_wait", lambda *args: None)
    serve_production(config, stop_event=stopped)
    assert observed == ["waiting_data", "blocked", "expired"]


@pytest.mark.parametrize("start,jump", [
    (datetime(2026, 9, 23, 22, 30, tzinfo=timezone.utc), 3600),  # First 19:00 NY attempt.
    (datetime(2026, 9, 24, 0, 0, tzinfo=timezone.utc), 1800),  # Retry window.
    (datetime(2026, 9, 24, 13, 0, tzinfo=timezone.utc), 3600),  # Past market open.
])
def test_wait_rechecks_absolute_time_after_suspend(tmp_path, monkeypatch, start, jump):
    from facdigger.production import service

    observed = [start]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return observed[0].astimezone(tz)

    class Stop:
        calls = 0

        def is_set(self):
            return False

        def wait(self, seconds):
            self.calls += 1
            observed[0] += timedelta(seconds=seconds + (jump if self.calls == 1 else 0))

    stop = Stop()
    monkeypatch.setattr(service, "datetime", Clock)
    service._heartbeat_wait(_config(tmp_path), stop, 1800, "not_due")
    assert stop.calls == 1
