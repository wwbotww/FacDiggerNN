"""Portable container-owned scheduler for the daily production transaction."""

from __future__ import annotations

import signal
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from facdigger.data.session_store import load_current_revision, prune_source_revisions
from facdigger.production.calendar import production_window
from facdigger.production.config import ProductionServiceConfig
from facdigger.production.lock import ProductionLock
from facdigger.production.retention import prune_inference_snapshots
from facdigger.production.runner import run_production_tick
from facdigger.production.state import ProductionState


def production_health(
    config: ProductionServiceConfig,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    observed = now or datetime.now(timezone.utc)
    with ProductionState(config.state_database) as state:
        health = state.service_health()
        latest = state.latest()
    if health is None:
        return {"healthy": False, "reason": "service has no heartbeat"}
    heartbeat = datetime.fromisoformat(str(health["heartbeat_at"]))
    maximum_age = timedelta(seconds=max(config.poll_seconds * 3, 180))
    age = observed.astimezone(timezone.utc) - heartbeat.astimezone(timezone.utc)
    return {
        **health,
        "healthy": timedelta(0) <= age <= maximum_age,
        "age_seconds": age.total_seconds(),
        "maximum_age_seconds": maximum_age.total_seconds(),
        # Data readiness is not container liveness. A skipped date must not
        # create restart loops or imply that yesterday's factor is usable.
        "production": (
            {
                "target_date": latest.target_date.isoformat(),
                "status": latest.status,
                "quality": (latest.quality_report or {}).get("status"),
            }
            if latest else None
        ),
    }


def _seconds_until_next_decision(
    config: ProductionServiceConfig,
    now: datetime,
) -> float:
    window = production_window(now, config.schedule)
    if window.phase == "not_due":
        return max((window.first_attempt_at - now).total_seconds(), 1.0)
    with ProductionState(config.state_database) as state:
        record = state.get(window.target_date)
    if (
        record is not None
        and record.status == "waiting_data"
        and record.next_retry_at is not None
    ):
        return max((record.next_retry_at - now).total_seconds(), 1.0)
    return float(config.poll_seconds)


def _heartbeat_wait(
    config: ProductionServiceConfig,
    stopped: threading.Event,
    delay_seconds: float,
    detail: str,
) -> None:
    deadline = datetime.now(timezone.utc) + timedelta(seconds=max(delay_seconds, 0.0))
    interval = float(config.poll_seconds)
    while not stopped.is_set():
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            return
        stopped.wait(min(interval, remaining))
        if not stopped.is_set():
            with ProductionState(config.state_database) as state:
                state.heartbeat(phase="sleeping", detail=detail)


def _run_tick_with_heartbeat(
    config: ProductionServiceConfig,
    stopped: threading.Event,
    now: datetime,
) -> Any:
    finished = threading.Event()

    def heartbeat() -> None:
        while not finished.wait(config.poll_seconds):
            if stopped.is_set():
                return
            with ProductionState(config.state_database) as state:
                state.heartbeat(phase="running_tick")

    worker = threading.Thread(target=heartbeat, name="production-heartbeat", daemon=True)
    worker.start()
    try:
        return run_production_tick(config, now=now)
    finally:
        finished.set()
        worker.join(timeout=1)


def serve_production(
    config: ProductionServiceConfig,
    *,
    stop_event: threading.Event | None = None,
) -> None:
    """Poll forever inside one container; schedule decisions use New York time."""

    stopped = stop_event or threading.Event()

    def request_stop(*_: object) -> None:
        stopped.set()

    if stop_event is None:
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
    with ProductionLock(config.state_database):
        while not stopped.is_set():
            now = datetime.now(timezone.utc)
            with ProductionState(config.state_database) as state:
                state.heartbeat(at=now, phase="running_tick")
            try:
                result = _run_tick_with_heartbeat(config, stopped, now)
                detail = result.action
                if result.action not in {"not_due", "retry_wait"}:
                    prune_inference_snapshots(
                        config.inference.output_root,
                        keep_sessions=config.inference.retention_sessions,
                    )
                if (
                    result.action not in {"not_due", "retry_wait"}
                    and (config.data.store_root / "CURRENT").is_file()
                ):
                    current = load_current_revision(config.data.store_root)
                    parent = (current.manifest.get("production_revision") or {}).get(
                        "parent_revision_id"
                    )
                    keep = {current.revision_id}
                    if isinstance(parent, str) and parent:
                        keep.add(parent)
                    prune_source_revisions(config.data.store_root, keep=keep)
            except Exception as exc:
                detail = f"service_error:{type(exc).__name__}:{exc}"
            with ProductionState(config.state_database) as state:
                state.heartbeat(
                    phase="sleeping",
                    detail=detail,
                )
            _heartbeat_wait(
                config,
                stopped,
                _seconds_until_next_decision(config, datetime.now(timezone.utc)),
                detail,
            )
        with ProductionState(config.state_database) as state:
            state.heartbeat(phase="stopped")
