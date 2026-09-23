"""Small transactional ledger for idempotent daily production ticks."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

ProductionStatus = Literal["waiting_data", "running", "published", "expired", "blocked"]


@dataclass(frozen=True)
class ProductionRecord:
    target_date: date
    release_id: str
    status: ProductionStatus
    attempts: int
    next_retry_at: datetime | None
    snapshot_id: str | None
    delivery_id: str | None
    error: str | None
    quality_reference: dict[str, Any] | None = None
    quality_report: dict[str, Any] | None = None


class ProductionState:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS production_runs (
                target_date TEXT PRIMARY KEY,
                release_id TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                next_retry_at TEXT,
                snapshot_id TEXT,
                delivery_id TEXT,
                error TEXT,
                quality_reference TEXT,
                quality_report TEXT,
                last_notice TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        # Extend existing local ledgers without discarding their pinned releases
        # or published dates. Serialize migrations with the heartbeat connection.
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            columns = {
                row[1] for row in self._connection.execute("PRAGMA table_info(production_runs)")
            }
            for column in ("quality_reference", "quality_report", "last_notice"):
                if column not in columns:
                    self._connection.execute(
                        f"ALTER TABLE production_runs ADD COLUMN {column} TEXT"
                    )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS service_state (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                heartbeat_at TEXT NOT NULL,
                phase TEXT NOT NULL,
                detail TEXT
            )
            """
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ProductionState:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get(self, target: date) -> ProductionRecord | None:
        row = self._connection.execute(
            """SELECT target_date, release_id, status, attempts, next_retry_at,
                      snapshot_id, delivery_id, error, quality_reference, quality_report
               FROM production_runs WHERE target_date = ?""",
            (target.isoformat(),),
        ).fetchone()
        if row is None:
            return None
        retry = datetime.fromisoformat(row[4]) if row[4] else None
        return ProductionRecord(
            target_date=date.fromisoformat(row[0]),
            release_id=row[1],
            status=row[2],
            attempts=int(row[3]),
            next_retry_at=retry,
            snapshot_id=row[5],
            delivery_id=row[6],
            error=row[7],
            quality_reference=json.loads(row[8]) if row[8] else None,
            quality_report=json.loads(row[9]) if row[9] else None,
        )

    def put(
        self,
        target: date,
        release_id: str,
        status: ProductionStatus,
        *,
        attempts: int,
        next_retry_at: datetime | None = None,
        snapshot_id: str | None = None,
        delivery_id: str | None = None,
        error: str | None = None,
        quality_reference: dict[str, Any] | None = None,
        quality_report: dict[str, Any] | None = None,
    ) -> ProductionRecord:
        now = datetime.now(timezone.utc).isoformat()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.get(target)
            if existing is not None and existing.release_id != release_id:
                raise ValueError(
                    "release_id cannot change after the target date has entered production"
                )
            if existing is not None and existing.quality_reference is not None:
                if (
                    quality_reference is not None
                    and quality_reference != existing.quality_reference
                ):
                    raise ValueError("quality reference cannot change within a target date")
                quality_reference = existing.quality_reference
            if quality_report is None and existing is not None:
                quality_report = existing.quality_report
            if status == "running" and error is None and existing is not None:
                error = existing.error  # A retry/crash must not erase the last failure.
            self._connection.execute(
                """
                INSERT INTO production_runs (
                    target_date, release_id, status, attempts, next_retry_at,
                    snapshot_id, delivery_id, error, quality_reference, quality_report, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(target_date) DO UPDATE SET
                    status=excluded.status,
                    attempts=excluded.attempts,
                    next_retry_at=excluded.next_retry_at,
                    snapshot_id=excluded.snapshot_id,
                    delivery_id=excluded.delivery_id,
                    error=excluded.error,
                    quality_reference=excluded.quality_reference,
                    quality_report=excluded.quality_report,
                    updated_at=excluded.updated_at
                """,
                (
                    target.isoformat(),
                    release_id,
                    status,
                    attempts,
                    next_retry_at.isoformat() if next_retry_at else None,
                    snapshot_id,
                    delivery_id,
                    error,
                    json.dumps(quality_reference, sort_keys=True) if quality_reference else None,
                    json.dumps(quality_report, sort_keys=True) if quality_report else None,
                    now,
                ),
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        result = self.get(target)
        assert result is not None
        self._report_transition(result)
        return result

    def _report_transition(self, record: ProductionRecord) -> None:
        if record.status == "running":
            return
        quality = record.quality_report or {}
        # Ordinary state comparison, not another semantic hash. Attempts, clocks
        # and changing counts do not emit the same operational alarm repeatedly.
        notice = json.dumps({
            "status": record.status,
            "quality": quality.get("status"),
            "violations": quality.get("violations", []),
            "error_type": (record.error or "").split(":", 1)[0],
        }, sort_keys=True)
        changed = self._connection.execute(
            "UPDATE production_runs SET last_notice = ? "
            "WHERE target_date = ? AND (last_notice IS NULL OR last_notice != ?)",
            (notice, record.target_date.isoformat(), notice),
        ).rowcount
        if not changed:
            return
        degraded = quality.get("status") == "degraded"
        level = logging.INFO if record.status == "published" and not degraded else logging.WARNING
        logger.log(level, json.dumps({
            "event": "production_readiness",
            "target_date": record.target_date.isoformat(),
            "status": record.status,
            "quality": quality.get("status"),
            "violations": quality.get("violations", []),
            "attempts": record.attempts,
            "next_retry_at": record.next_retry_at.isoformat() if record.next_retry_at else None,
            "computation": quality.get("computation"),
            "delivery": quality.get("delivery"),
            "error": record.error,
        }, sort_keys=True))

    def latest(self) -> ProductionRecord | None:
        row = self._connection.execute(
            "SELECT target_date FROM production_runs ORDER BY target_date DESC LIMIT 1"
        ).fetchone()
        return None if row is None else self.get(date.fromisoformat(row[0]))

    def heartbeat(
        self,
        *,
        at: datetime | None = None,
        phase: str,
        detail: str | None = None,
    ) -> None:
        observed = at or datetime.now(timezone.utc)
        if observed.tzinfo is None:
            raise ValueError("heartbeat timestamp must be timezone-aware")
        self._connection.execute(
            """
            INSERT INTO service_state (singleton, heartbeat_at, phase, detail)
            VALUES (1, ?, ?, ?)
            ON CONFLICT(singleton) DO UPDATE SET
                heartbeat_at=excluded.heartbeat_at,
                phase=excluded.phase,
                detail=excluded.detail
            """,
            (observed.astimezone(timezone.utc).isoformat(), phase, detail),
        )

    def service_health(self) -> dict[str, str | None] | None:
        row = self._connection.execute(
            "SELECT heartbeat_at, phase, detail FROM service_state WHERE singleton = 1"
        ).fetchone()
        if row is None:
            return None
        return {"heartbeat_at": row[0], "phase": row[1], "detail": row[2]}
