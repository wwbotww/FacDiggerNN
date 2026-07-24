"""Deterministic regular-session calendar for the US equity exchanges.

NYSE, Nasdaq and NYSE American share the regular full-day closure calendar used
by the EODHD US universe.  Half days remain sessions because EOD daily bars are
expected on those dates.
"""

from __future__ import annotations

import calendar as calendar_module
from datetime import date, timedelta

import polars as pl

CALENDAR_NAME = "US_EQUITIES_REGULAR"
CALENDAR_VERSION = "2026.1"

# Unscheduled full-day closures which cannot be expressed as recurring rules.
_SPECIAL_CLOSURES = {
    date(2001, 9, 11),
    date(2001, 9, 12),
    date(2001, 9, 13),
    date(2001, 9, 14),
    date(2004, 6, 11),  # President Reagan national day of mourning
    date(2007, 1, 2),  # President Ford national day of mourning
    date(2012, 10, 29),  # Hurricane Sandy
    date(2012, 10, 30),
    date(2018, 12, 5),  # President George H. W. Bush national day of mourning
    date(2025, 1, 9),  # President Carter national day of mourning
}


def _observed(day: date) -> date:
    if day.weekday() == calendar_module.SATURDAY:
        return day - timedelta(days=1)
    if day.weekday() == calendar_module.SUNDAY:
        return day + timedelta(days=1)
    return day


def _new_year_closure(year: int) -> date | None:
    """NYSE does not move a Saturday New Year closure to the preceding Friday."""

    day = date(year, 1, 1)
    if day.weekday() == calendar_module.SATURDAY:
        return None
    if day.weekday() == calendar_module.SUNDAY:
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (occurrence - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year, month, calendar_module.monthrange(year, month)[1])
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter_sunday(year: int) -> date:
    """Return Gregorian Easter using the Meeus/Jones/Butcher algorithm."""

    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def full_day_closures(year: int) -> set[date]:
    closures = {
        _nth_weekday(year, 1, calendar_module.MONDAY, 3),  # MLK Day
        _nth_weekday(year, 2, calendar_module.MONDAY, 3),  # Washington's Birthday
        _easter_sunday(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, calendar_module.MONDAY),  # Memorial Day
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, calendar_module.MONDAY, 1),  # Labor Day
        _nth_weekday(year, 11, calendar_module.THURSDAY, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),
    }
    new_year = _new_year_closure(year)
    if new_year is not None:
        closures.add(new_year)
    if year >= 2022:
        closures.add(_observed(date(year, 6, 19)))  # Juneteenth
    closures.update(day for day in _SPECIAL_CLOSURES if day.year == year)
    return closures


def regular_sessions(start: date, end: date) -> list[date]:
    """Return inclusive regular sessions for US equities."""

    if start > end:
        raise ValueError("calendar start must not be after end")
    closures: set[date] = set()
    # Include adjacent years because observed New Year can fall in December.
    for year in range(start.year - 1, end.year + 2):
        closures.update(full_day_closures(year))
    result: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in closures:
            result.append(current)
        current += timedelta(days=1)
    return result


def regular_session_frame(start: date, end: date) -> pl.DataFrame:
    return pl.DataFrame(
        {"trade_date": regular_sessions(start, end)},
        schema={"trade_date": pl.Date},
    )
