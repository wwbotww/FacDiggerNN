from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from facdigger.data.contracts import DataContractError
from facdigger.data.providers.eodhd.client import DailyCallBudget
from facdigger.data.providers.eodhd.config import EODHDConfig
from facdigger.data.providers.eodhd.daily import (
    DailyDataNotReady,
    fetch_daily_revision,
    require_fresh_daily_requests,
)


def _config(tmp_path: Path) -> EODHDConfig:
    return EODHDConfig.model_validate(
        {
            "allow_demo_token": False,
            "universe": {
                "mode": "historical_liquid",
                "max_symbols": 1000,
            },
            "cache_dir": tmp_path / "cache",
            "state_dir": tmp_path / "state",
            "refresh": True,
            "cache_ttl_hours": 0,
            "max_calls_per_day": 1000,
            "delisting_imputation": {"enabled": True},
        }
    )


class DailyClient:
    def __init__(self, tmp_path: Path, *, target_available: bool = True) -> None:
        self.request_log: list[dict[str, Any]] = []
        self.budget = DailyCallBudget(tmp_path / "budget.json", 1000)
        self.target_available = target_available

    def get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        call_cost: int = 1,
    ) -> Any:
        self.request_log.append(
            {
                "path": path,
                "params": params or {},
                "cache_hit": False,
                "call_cost": call_cost,
            }
        )
        if path.startswith("exchange-symbol-list/"):
            return (
                [
                    {
                        "Code": "AAA",
                        "Exchange": "NASDAQ",
                        "Type": "Common Stock",
                        "Isin": "US0000000001",
                    }
                ]
                if (params or {}).get("delisted") == 0
                else []
            )
        if path.startswith("eod-bulk-last-day/"):
            day = str((params or {})["date"])
            if not self.target_available and day == "2026-08-12":
                return []
            return [
                {
                    "code": "AAA",
                    "date": day,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "adjusted_close": 10.5,
                    "volume": 1000.0,
                }
            ]
        raise AssertionError(path)


def test_daily_revision_uses_one_weighted_bulk_request_per_session(tmp_path) -> None:
    client = DailyClient(tmp_path)
    revision = fetch_daily_revision(
        client,  # type: ignore[arg-type]
        _config(tmp_path),
        revision_start=date(2026, 8, 10),
        target_date=date(2026, 8, 12),
    )

    assert revision.bars.height == 3
    assert revision.bars["trade_date"].max() == date(2026, 8, 12)
    bulk = [row for row in revision.request_log if row["path"].startswith("eod-bulk")]
    assert [row["call_cost"] for row in bulk] == [100, 100, 100]
    require_fresh_daily_requests(revision)


def test_daily_revision_retries_when_target_bulk_is_not_ready(tmp_path) -> None:
    with pytest.raises(DailyDataNotReady, match="not ready"):
        fetch_daily_revision(
            DailyClient(tmp_path, target_available=False),  # type: ignore[arg-type]
            _config(tmp_path),
            revision_start=date(2026, 8, 10),
            target_date=date(2026, 8, 12),
        )


def test_daily_publication_rejects_cached_eod_requests(tmp_path) -> None:
    client = DailyClient(tmp_path)
    revision = fetch_daily_revision(
        client,  # type: ignore[arg-type]
        _config(tmp_path),
        revision_start=date(2026, 8, 10),
        target_date=date(2026, 8, 12),
    )
    client.request_log[-1]["cache_hit"] = True
    changed = replace(revision, request_log=tuple(client.request_log))

    with pytest.raises(DataContractError, match="fresh"):
        require_fresh_daily_requests(changed)
