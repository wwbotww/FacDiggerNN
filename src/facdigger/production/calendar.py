"""New-York-time scheduling decisions for one daily production target."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal
from zoneinfo import ZoneInfo

from facdigger.data.market_calendar import (
    next_regular_session,
    previous_regular_session,
    regular_session,
)
from facdigger.production.config import ProductionScheduleConfig

NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class ProductionWindow:
    target_date: date
    first_attempt_at: datetime
    cutoff_at: datetime
    phase: Literal["not_due", "open", "expired"]


def production_window(
    now: datetime,
    schedule: ProductionScheduleConfig,
) -> ProductionWindow:
    """Return the most recent target whose production window could contain ``now``."""

    if now.tzinfo is None:
        raise ValueError("production clock must be timezone-aware")
    local = now.astimezone(NEW_YORK)
    today = local.date()
    today_session = regular_session(today)
    if today_session is not None and now >= today_session.open_utc:
        target = today
    else:
        target = previous_regular_session(today)
    first = datetime.combine(target, schedule.first_attempt, tzinfo=NEW_YORK)
    execution_session = regular_session(next_regular_session(target))
    assert execution_session is not None
    cutoff = execution_session.open_utc.astimezone(NEW_YORK)
    phase: Literal["not_due", "open", "expired"]
    if local < first:
        phase = "not_due"
    elif local >= cutoff:
        phase = "expired"
    else:
        phase = "open"
    return ProductionWindow(target, first, cutoff, phase)
