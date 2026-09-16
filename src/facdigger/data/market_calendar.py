"""常规美股交易时段；第三方日历类型只存在于本模块内部。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from importlib.metadata import version
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import polars as pl

if TYPE_CHECKING:
    from exchange_calendars.exchange_calendar import ExchangeCalendar

CALENDAR_NAME = "US_EQUITIES_REGULAR"
CALENDAR_VERSION = "exchange_calendars:4.13.2:XNYS"
MARKET_TIMEZONE = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class MarketSession:
    """一个交易日的实际常规时段，不包含盘前盘后。"""

    session_date: date
    open_utc: datetime
    close_utc: datetime


@lru_cache(maxsize=32)
def _calendar(first_year: int, last_year: int) -> ExchangeCalendar:
    # 相邻年度为前后日查询留出空间，避免库的默认窗口依赖机器当前日期。
    import exchange_calendars

    if version("exchange_calendars") != "4.13.2":
        raise ValueError("exchange_calendars version does not match calendar provenance")
    return exchange_calendars.get_calendar(
        "XNYS", start=f"{first_year - 1:04d}-01-01", end=f"{last_year + 1:04d}-12-31"
    )


def regular_sessions(start: date, end: date) -> list[date]:
    """返回闭区间交易日，提前收盘日仍计入。"""
    if start > end:
        raise ValueError("calendar start must not be after end")
    calendar = _calendar(start.year, end.year)
    return [value.date() for value in calendar.sessions_in_range(start, end)]


def regular_session(day: date) -> MarketSession | None:
    """返回 UTC 开收盘；休市返回空值，不推断成工作日。"""
    calendar = _calendar(day.year, day.year)
    if not calendar.is_session(day):
        return None
    return MarketSession(
        day,
        calendar.session_open(day).to_pydatetime().astimezone(timezone.utc),
        calendar.session_close(day).to_pydatetime().astimezone(timezone.utc),
    )


def previous_regular_session(day: date) -> date:
    """返回严格早于输入日期的最近交易日。"""
    calendar = _calendar(day.year, day.year)
    return calendar.date_to_session(day - timedelta(days=1), direction="previous").date()


def next_regular_session(day: date) -> date:
    """返回严格晚于输入日期的首个交易日。"""
    calendar = _calendar(day.year, day.year)
    return calendar.date_to_session(day + timedelta(days=1), direction="next").date()


def shift_regular_session(day: date, offset: int) -> date:
    """按交易日位移；零位移要求输入本身是交易日。"""
    if offset == 0 and regular_session(day) is None:
        raise ValueError(f"not a regular US-equity session: {day.isoformat()}")
    step = next_regular_session if offset > 0 else previous_regular_session
    for _ in range(abs(offset)):
        day = step(day)
    return day


def regular_session_frame(start: date, end: date) -> pl.DataFrame:
    """将共同日历转为现有数据管线需要的交易日表。"""
    return pl.DataFrame(
        {"trade_date": regular_sessions(start, end)}, schema={"trade_date": pl.Date}
    )
