"""以固定事实验证时区、休市和提前收盘，不能由待测库生成预期值。"""

import json
from datetime import date
from pathlib import Path

import pytest

from facdigger.data.market_calendar import (
    next_regular_session,
    previous_regular_session,
    regular_session,
    regular_sessions,
    shift_regular_session,
)


def test_shared_session_facts() -> None:
    fixture = Path(__file__).parents[2] / "fixtures" / "us_equities_sessions.json"
    for row in json.loads(fixture.read_text())["sessions"]:
        day = date.fromisoformat(row["date"])
        session = regular_session(day)
        if row["open"] is None:
            assert session is None
            assert regular_sessions(day, day) == []
        else:
            assert session is not None
            assert session.session_date == day
            assert session.open_utc.isoformat() == row["open"]
            assert session.close_utc.isoformat() == row["close"]
            assert regular_sessions(day, day) == [day]
        assert previous_regular_session(day).isoformat() == row["previous"]
        assert next_regular_session(day).isoformat() == row["next"]


def test_explicit_historical_range_and_offsets() -> None:
    assert regular_sessions(date(2000, 1, 1), date(2000, 1, 4)) == [
        date(2000, 1, 3), date(2000, 1, 4)
    ]
    assert shift_regular_session(date(2026, 11, 25), 2) == date(2026, 11, 30)
    assert shift_regular_session(date(2026, 11, 30), -2) == date(2026, 11, 25)
    assert shift_regular_session(date(2026, 11, 25), 0) == date(2026, 11, 25)
    with pytest.raises(ValueError, match="not a regular"):
        shift_regular_session(date(2026, 11, 26), 0)
    with pytest.raises(ValueError, match="start"):
        regular_sessions(date(2026, 11, 27), date(2026, 11, 25))
