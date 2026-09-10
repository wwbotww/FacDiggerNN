from datetime import date, datetime, timezone

import pytest

from facdigger.production.config import ProductionServiceConfig
from facdigger.production.lock import ProductionAlreadyRunning, ProductionLock
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
        }
    )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("inference", "output_root", "data/snapshots/production"),
        ("data", "store_root", "data/walk_forward_snapshots/production"),
        (None, "state_database", "data/snapshots/production.sqlite3"),
    ],
)
def test_config_rejects_generated_paths_inside_training_snapshots(
    tmp_path,
    section,
    field,
    value,
) -> None:
    base = _config(tmp_path).model_dump()
    if section is None:
        base[field] = value
    else:
        base[section][field] = value
    with pytest.raises(ValueError, match="must not be inside"):
        ProductionServiceConfig.model_validate(base)


def test_config_rejects_unordered_minimums_and_release_placeholder(tmp_path) -> None:
    base = _config(tmp_path).model_dump()
    base["inference"]["minimum_candidate_rows"] = 10
    base["inference"]["minimum_eligible_rows"] = 20
    with pytest.raises(ValueError, match="cannot exceed"):
        ProductionServiceConfig.model_validate(base)

    base = _config(tmp_path).model_dump()
    base["model"]["release_id"] = "0" * 64
    with pytest.raises(ValueError, match="placeholder"):
        ProductionServiceConfig.model_validate(base)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("schedule", "first_attempt", "18:00:00", "first_attempt"),
        ("schedule", "retry_minutes", 15, "retry_minutes"),
        ("data", "revision_sessions", 9, "revision_sessions"),
        ("inference", "retention_sessions", 9, "retention"),
    ],
)
def test_config_freezes_approved_daily_protocol(
    tmp_path,
    section,
    field,
    value,
    message,
) -> None:
    payload = _config(tmp_path).model_dump()
    payload[section][field] = value

    with pytest.raises(ValueError, match=message):
        ProductionServiceConfig.model_validate(payload)


def test_state_pins_release_and_persists_heartbeat(tmp_path) -> None:
    state_path = tmp_path / "production.sqlite3"
    target = date(2026, 8, 12)
    with ProductionState(state_path) as state:
        state.put(target, "1" * 64, "running", attempts=1)
        with pytest.raises(ValueError, match="cannot change"):
            state.put(target, "2" * 64, "running", attempts=2)
        observed = datetime(2026, 8, 13, tzinfo=timezone.utc)
        state.heartbeat(at=observed, phase="sleeping", detail="not_due")
        assert state.service_health() == {
            "heartbeat_at": observed.isoformat(),
            "phase": "sleeping",
            "detail": "not_due",
        }


def test_single_process_lock_is_exclusive(tmp_path) -> None:
    state_path = tmp_path / "production.sqlite3"
    with ProductionLock(state_path):
        with pytest.raises(ProductionAlreadyRunning):
            with ProductionLock(state_path):
                pass
    with ProductionLock(state_path):
        pass


def test_quality_reference_survives_restarts_and_cannot_shrink(tmp_path) -> None:
    target = date(2026, 8, 17)
    path = tmp_path / "state.sqlite3"
    reference = {"security_ids": ["A", "B"], "target_date": target.isoformat()}
    with ProductionState(path) as state:
        state.put(target, "1" * 64, "running", attempts=1, quality_reference=reference)
    with ProductionState(path) as state:
        record = state.put(target, "1" * 64, "waiting_data", attempts=2)
        assert record.quality_reference == reference
        with pytest.raises(ValueError, match="reference cannot change"):
            state.put(target, "1" * 64, "running", attempts=3,
                      quality_reference={**reference, "security_ids": ["A"]})


def test_existing_state_schema_is_extended_without_losing_published_dates(tmp_path) -> None:
    import sqlite3

    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE production_runs (
            target_date TEXT PRIMARY KEY, release_id TEXT NOT NULL, status TEXT NOT NULL,
            attempts INTEGER NOT NULL, next_retry_at TEXT, snapshot_id TEXT,
            delivery_id TEXT, error TEXT, updated_at TEXT NOT NULL)""")
        connection.execute("INSERT INTO production_runs VALUES (?,?,?,?,?,?,?,?,?)", (
            "2026-08-17", "1" * 64, "published", 2, None, "snapshot", "delivery", None, "old",
        ))
    for _ in range(2):
        with ProductionState(path) as state:
            row = state.get(date(2026, 8, 17))
            assert row.status == "published" and row.delivery_id == "delivery"
            assert row.quality_reference is None and row.quality_report is None


def test_repeated_readiness_alarm_is_deduplicated_and_recovery_logged(tmp_path, caplog):
    import json
    import logging

    caplog.set_level(logging.INFO, logger="facdigger.production.state")
    day = date(2026, 8, 17)
    quality = {"status": "insufficient", "violations": ["missing_data"]}
    with ProductionState(tmp_path / "state.sqlite3") as state:
        for attempt in (1, 2):
            state.put(day, "1" * 64, "running", attempts=attempt)
            state.put(day, "1" * 64, "waiting_data", attempts=attempt, quality_report=quality,
                      error="TargetSessionIncomplete: changed counts")
        state.put(day, "1" * 64, "published", attempts=3,
                  quality_report={"status": "ready", "violations": []})
    notices = [json.loads(record.message) for record in caplog.records]
    assert [notice["status"] for notice in notices] == ["waiting_data", "published"]


@pytest.mark.parametrize(("field", "value"), [
    ("max_computational_missing_fraction", 1),
    ("max_delivery_unscorable_fraction", -0.1),
    ("minimum_delivery_eligible_rows", 0),
])
def test_quality_configuration_rejects_invalid_tolerances(tmp_path, field, value):
    payload = _config(tmp_path).model_dump()
    payload["quality"][field] = value
    with pytest.raises(ValueError):
        ProductionServiceConfig.model_validate(payload)
