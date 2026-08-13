from datetime import datetime
from zoneinfo import ZoneInfo

from facdigger.production.calendar import production_window
from facdigger.production.config import ProductionScheduleConfig

NY = ZoneInfo("America/New_York")


def test_regular_day_moves_from_previous_cutoff_to_same_day_waiting_window() -> None:
    schedule = ProductionScheduleConfig()

    before_open = production_window(datetime(2026, 8, 17, 8, 0, tzinfo=NY), schedule)
    after_open = production_window(datetime(2026, 8, 17, 10, 0, tzinfo=NY), schedule)
    due = production_window(datetime(2026, 8, 17, 19, 0, tzinfo=NY), schedule)

    assert before_open.target_date.isoformat() == "2026-08-14"
    assert before_open.phase == "open"
    assert after_open.target_date.isoformat() == "2026-08-17"
    assert after_open.phase == "not_due"
    assert due.target_date.isoformat() == "2026-08-17"
    assert due.phase == "open"


def test_weekend_keeps_friday_window_open_until_monday_session_open() -> None:
    window = production_window(
        datetime(2026, 8, 16, 12, 0, tzinfo=NY),
        ProductionScheduleConfig(),
    )

    assert window.target_date.isoformat() == "2026-08-14"
    assert window.phase == "open"
    assert window.cutoff_at.isoformat() == "2026-08-17T09:30:00-04:00"
