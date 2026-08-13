"""New-York-time scheduling decisions for one daily production target."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal
from zoneinfo import ZoneInfo

from facdigger.data.providers.eodhd.market_calendar import (
    next_regular_session,
    previous_regular_session,
    regular_session_open,
    regular_sessions,
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
    today_is_session = bool(regular_sessions(today, today))
    if today_is_session:
        today_first = datetime.combine(today, schedule.first_attempt, tzinfo=NEW_YORK)
        today_open = regular_session_open(today)
        if local >= today_first:
            target = today
        elif local < today_open:
            target = previous_regular_session(today)
        else:
            return ProductionWindow(
                today,
                today_first,
                regular_session_open(next_regular_session(today)),
                "not_due",
            )
    else:
        target = previous_regular_session(today)
    first = datetime.combine(target, schedule.first_attempt, tzinfo=NEW_YORK)
    cutoff = regular_session_open(next_regular_session(target))
    phase: Literal["not_due", "open", "expired"]
    if local < first:
        phase = "not_due"
    elif local >= cutoff:
        phase = "expired"
    else:
        phase = "open"
    return ProductionWindow(target, first, cutoff, phase)
