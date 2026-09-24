"""Explicit operator recovery; never fetch, score, or create a new delivery."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timezone
from typing import Any

from facdigger.data.contracts import DataContractError
from facdigger.data.session_store import load_current_revision
from facdigger.inference.releases import load_model_release
from facdigger.production.calendar import target_production_window
from facdigger.production.config import ProductionServiceConfig
from facdigger.production.lock import ProductionLock
from facdigger.production.publication import recover_publication
from facdigger.production.state import ProductionState


def resume_blocked_production(
    config: ProductionServiceConfig, *, target_date: date, expected_updated_at: str,
    reason: str, now_provider: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Reconcile original publication first, then explicitly requeue an open target."""
    if not reason.strip() or not expected_updated_at.strip():
        raise ValueError("resume requires a reason and expected_updated_at")
    clock = now_provider or (lambda: datetime.now(timezone.utc))
    with ProductionLock(config.state_database), ProductionState(config.state_database) as state:
        observed = clock()
        window = target_production_window(target_date, observed, config.schedule)
        record = state.get(target_date)
        if record is None or record.release_id != config.model.release_id:
            raise DataContractError("resume target or fixed release does not match the ledger")
        if record.status == "published":
            raise DataContractError("published production cannot be resumed")
        if record.status not in {"blocked", "waiting_data"}:
            raise DataContractError("only an inspected blocked record can be resumed")
        if record.status == "blocked" and record.updated_at != expected_updated_at:
            raise DataContractError("blocked record changed since inspection")
        # FD-04 may restore an on-time original after the cutoff. This is not
        # permission to create another batch, nor to promote staging output.
        recovered = recover_publication(config, state, window, record, observed=observed)
        if recovered is not None:
            return {"action": "already_published", "target_date": str(target_date),
                    "delivery_id": recovered.delivery_id, "attempts": recovered.attempts}
        if window.phase != "open":
            raise DataContractError("resume requires the original open publication window")
        release = load_model_release(config.model.release_root / config.model.release_id)
        if release.release_id != record.release_id:
            raise DataContractError("resume release artifact does not match the pinned release")
        load_current_revision(config.data.store_root)  # Corrupt inputs do not get requeued.
        resumed, repeated = state.schedule_resume(
            target_date, record.release_id, expected_updated_at=expected_updated_at,
            reason=reason, observed=clock(), cutoff_at=window.cutoff_at,
        )
        return {"action": "already_scheduled" if repeated else "resume_scheduled",
                "target_date": str(target_date), "attempts": resumed.attempts,
                "next_retry_at": resumed.next_retry_at.isoformat(),
                "cutoff_at": window.cutoff_at.isoformat(), "previous_error": resumed.error,
                "release_id": resumed.release_id}
