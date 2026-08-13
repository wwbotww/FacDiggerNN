"""Small transactional ledger for idempotent daily production ticks."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

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
                updated_at TEXT NOT NULL
            )
            """
        )
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
                      snapshot_id, delivery_id, error
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
    ) -> ProductionRecord:
        now = datetime.now(timezone.utc).isoformat()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.get(target)
            if existing is not None and existing.release_id != release_id:
                raise ValueError(
                    "release_id cannot change after the target date has entered production"
                )
            self._connection.execute(
                """
                INSERT INTO production_runs (
                    target_date, release_id, status, attempts, next_retry_at,
                    snapshot_id, delivery_id, error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(target_date) DO UPDATE SET
                    status=excluded.status,
                    attempts=excluded.attempts,
                    next_retry_at=excluded.next_retry_at,
                    snapshot_id=excluded.snapshot_id,
                    delivery_id=excluded.delivery_id,
                    error=excluded.error,
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
                    now,
                ),
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        result = self.get(target)
        assert result is not None
        return result

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
