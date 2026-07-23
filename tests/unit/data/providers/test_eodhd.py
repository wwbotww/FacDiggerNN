from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from facdigger.data.contracts import validate_bars, validate_corporate_actions, validate_universe
from facdigger.data.providers.eodhd.client import (
    DailyCallBudget,
    EODHDBudgetError,
    EODHDClient,
)
from facdigger.data.providers.eodhd.config import EODHDConfig
from facdigger.data.providers.eodhd.mapper import (
    build_imputed_delistings,
    build_metadata_index,
    build_universe,
    map_corporate_actions,
    map_eod_bars,
    parse_split_ratio,
)
from facdigger.data.providers.eodhd.provider import EODHDProvider
from facdigger.data.providers.eodhd.universe import (
    discover_historical_symbols,
    select_top_liquid_symbols,
)
from facdigger.labels.forward_return import build_forward_excess_return_labels

EOD_ROWS = [
    {
        "date": f"2025-01-{day:02d}",
        "open": 100.0 + day,
        "high": 102.0 + day,
        "low": 99.0 + day,
        "close": 101.0 + day,
        "adjusted_close": (101.0 + day) * 0.98,
        "volume": 1_000_000 + day,
    }
    for day in range(1, 29)
]


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers: dict[str, str] = {}

    def json(self) -> Any:
        return self.payload


class FakeTransport:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, *, params: dict[str, Any], timeout: float) -> FakeResponse:
        self.calls.append((url, params))
        return FakeResponse(self.payload)


def make_client(tmp_path: Path, transport: FakeTransport, limit: int = 20) -> EODHDClient:
    return EODHDClient(
        base_url="https://example.invalid/api",
        api_token="super-secret",
        cache_dir=tmp_path / "cache",
        budget=DailyCallBudget(tmp_path / "state" / "budget.json", limit),
        transport=transport,
        max_retries=0,
        cache_ttl_hours=24,
    )


def test_client_cache_omits_token_and_preserves_budget(tmp_path: Path) -> None:
    transport = FakeTransport(EOD_ROWS[:1])
    client = make_client(tmp_path, transport)

    first = client.get_json("eod/AAPL.US", {"from": "2025-01-01"})
    second = client.get_json("eod/AAPL.US", {"from": "2025-01-01"})

    assert first == second
    assert len(transport.calls) == 1
    assert client.budget.status()["network_attempts"] == 1
    cache_text = next((tmp_path / "cache").glob("*.json")).read_text(encoding="utf-8")
    assert "super-secret" not in cache_text
    assert json.loads(cache_text)["request"]["path"] == "eod/AAPL.US"
    assert [entry["cache_hit"] for entry in client.request_log] == [False, True]


def test_client_stops_before_exceeding_daily_budget(tmp_path: Path) -> None:
    transport = FakeTransport(EOD_ROWS[:1])
    client = make_client(tmp_path, transport, limit=1)
    client.get_json("eod/AAPL.US")
    with pytest.raises(EODHDBudgetError, match="exhausted"):
        client.get_json("eod/TSLA.US")
    assert len(transport.calls) == 1


def test_client_budget_accounts_for_weighted_bulk_calls(tmp_path: Path) -> None:
    transport = FakeTransport(EOD_ROWS[:1])
    client = make_client(tmp_path, transport, limit=100)

    client.get_json("eod-bulk-last-day/US", call_cost=100)

    assert client.budget.status()["api_calls"] == 100
    assert client.budget.status()["network_attempts"] == 1
    with pytest.raises(EODHDBudgetError, match=r"100\+1>100"):
        client.get_json("eod/MSFT.US")


def test_eodhd_mapping_satisfies_standard_contracts() -> None:
    metadata = build_metadata_index(
        [
            {
                "Code": "AAPL",
                "Name": "Apple Inc",
                "Exchange": "NASDAQ",
                "Currency": "USD",
                "Type": "Common Stock",
                "Isin": "US0378331005",
            }
        ],
        "US",
    )
    bars = map_eod_bars(
        {"AAPL.US": EOD_ROWS},
        metadata,
        source_revision="test-v1",
        ingested_at=datetime(2025, 2, 1, tzinfo=timezone.utc),
    )
    universe = build_universe(bars, min_listed_sessions=20, min_price=1.0, min_adv20_usd=1_000_000)

    assert validate_bars(bars).height == len(EOD_ROWS)
    assert validate_universe(universe).height == len(EOD_ROWS)
    assert bars["security_id"][0] == "eodhd:isin:US0378331005"
    assert bars["adj_factor"][0] == pytest.approx(0.98)
    assert universe["exchange"][0] == "XNAS"
    assert universe["eligible"].sum() == 9


def test_corporate_action_mapping_has_explicit_factor_semantics() -> None:
    metadata = build_metadata_index([], "US")
    actions = map_corporate_actions(
        dividends_by_symbol={
            "AAPL.US": [
                {
                    "date": "2025-02-07",
                    "declarationDate": "2025-01-30",
                    "value": 0.25,
                    "currency": "USD",
                }
            ]
        },
        splits_by_symbol={"AAPL.US": [{"date": "2020-08-31", "split": "4/1"}]},
        metadata=metadata,
        source_revision="test-v1",
    )

    assert actions is not None
    validate_corporate_actions(actions)
    split = actions.filter(actions["action_type"] == "split").row(0, named=True)
    assert split["price_factor"] == pytest.approx(0.25)
    assert split["volume_factor"] == pytest.approx(4.0)
    assert parse_split_ratio("1/5") == pytest.approx((5.0, 0.2))


class FakeEODHDClient:
    def __init__(self, tmp_path: Path) -> None:
        self.request_log: list[dict[str, Any]] = []
        self.budget = DailyCallBudget(tmp_path / "fake-budget.json", 20)

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self.request_log.append({"path": path, "params": params or {}, "cache_hit": False})
        if path.startswith("eod/"):
            return EOD_ROWS
        raise AssertionError(f"unexpected request: {path}")


def test_provider_writes_only_standard_boundary_files(tmp_path: Path) -> None:
    config = EODHDConfig.model_validate(
        {
            "symbols": ["AAPL.US"],
            "start": "2025-01-01",
            "end": "2025-01-31",
            "output_dir": tmp_path / "bronze",
            "cache_dir": tmp_path / "cache",
            "state_dir": tmp_path / "state",
            "metadata_overrides": [
                {
                    "provider_symbol": "AAPL.US",
                    "isin": "US0378331005",
                    "exchange": "NASDAQ",
                }
            ],
        }
    )
    result = EODHDProvider(config, client=FakeEODHDClient(tmp_path)).ingest()

    assert set(result.files) == {"bars", "universe", "manifest"}
    assert result.files["bars"].is_file()
    assert result.files["universe"].is_file()
    assert result.manifest["delistings"]["emitted"] is False
    manifest_text = result.files["manifest"].read_text(encoding="utf-8")
    assert "super-secret" not in manifest_text


def test_top_liquid_selection_filters_exchange_type_and_ranks_deterministically() -> None:
    metadata = [
        {"Code": "AAA", "Exchange": "NASDAQ", "Type": "Common Stock"},
        {"Code": "BBB", "Exchange": "NYSE", "Type": "Common Stock"},
        {"Code": "CCC", "Exchange": "PINK", "Type": "Common Stock"},
        {"Code": "ETF1", "Exchange": "NASDAQ", "Type": "ETF"},
    ]
    bulk = [
        {"code": "AAA", "date": "2026-07-21", "close": 20, "avgvol_200d": 200_000},
        {"code": "BBB", "date": "2026-07-21", "close": 50, "avgvol_200d": 100_000},
        {"code": "CCC", "date": "2026-07-21", "close": 100, "avgvol_200d": 999_999},
        {"code": "ETF1", "date": "2026-07-21", "close": 100, "avgvol_200d": 999_999},
    ]
    selection = EODHDConfig.model_validate(
        {
            "allow_demo_token": False,
            "universe": {
                "mode": "top_liquid",
                "max_symbols": 2,
                "min_price": 5,
                "min_avg_volume_200d": 100_000,
            },
        }
    ).universe

    symbols, audit = select_top_liquid_symbols(metadata, bulk, exchange_code="US", config=selection)

    assert symbols == ["BBB.US", "AAA.US"]
    assert audit["selected_count"] == 2
    assert audit["research_ready"] is False


def test_top_liquid_config_disallows_demo_fallback_and_explicit_symbols() -> None:
    with pytest.raises(ValueError, match="allow_demo_token=false"):
        EODHDConfig.model_validate({"universe": {"mode": "top_liquid", "max_symbols": 10}})
    with pytest.raises(ValueError, match="symbols must be empty"):
        EODHDConfig.model_validate(
            {
                "symbols": ["AAPL.US"],
                "allow_demo_token": False,
                "universe": {"mode": "top_liquid", "max_symbols": 10},
            }
        )


def test_historical_discovery_includes_active_and_delisted_without_current_ranking() -> None:
    active = [
        {"Code": "AAA", "Exchange": "NASDAQ", "Type": "Common Stock"},
        {"Code": "ETF1", "Exchange": "NASDAQ", "Type": "ETF"},
    ]
    delisted = [
        {"Code": "OLD", "Exchange": "NYSE", "Type": "Common Stock"},
        {"Code": "OTC", "Exchange": "PINK", "Type": "Common Stock"},
    ]
    selection = EODHDConfig.model_validate(
        {
            "allow_demo_token": False,
            "universe": {"mode": "historical_liquid", "max_symbols": 1000},
            "delisting_imputation": {"enabled": True},
        }
    ).universe

    symbols, metadata, audit = discover_historical_symbols(
        active,
        delisted,
        exchange_code="US",
        config=selection,
    )

    assert symbols == ["AAA.US", "OLD.US"]
    assert {row["Code"]: row["_is_delisted"] for row in metadata} == {
        "AAA": False,
        "OLD": True,
    }
    assert audit["selection_uses_current_liquidity"] is False
    assert audit["daily_max_symbols"] == 1000


def test_dynamic_universe_grid_and_delisting_imputation_are_auditable() -> None:
    metadata = build_metadata_index(
        [
            {
                "Code": "AAA",
                "Exchange": "NASDAQ",
                "Type": "Common Stock",
                "_is_delisted": False,
            },
            {
                "Code": "OLD",
                "Exchange": "NYSE",
                "Type": "Common Stock",
                "_is_delisted": True,
            },
        ],
        "US",
    )
    active_rows = [row for index, row in enumerate(EOD_ROWS) if index != 9]
    bars = map_eod_bars(
        {"AAA.US": active_rows, "OLD.US": EOD_ROWS[:24]},
        metadata,
        source_revision="test-historical",
        ingested_at=datetime(2025, 2, 1, tzinfo=timezone.utc),
    )
    universe = build_universe(
        bars,
        min_listed_sessions=1,
        min_price=1.0,
        min_adv20_usd=1.0,
        max_daily_symbols=1,
    )
    delistings = build_imputed_delistings(
        bars,
        universe,
        exchange_returns={"XNAS": -0.55, "XNYS": -0.30, "XASE": -0.30},
        default_return=-0.50,
        source_revision="test-historical",
    )

    assert universe.group_by("trade_date").agg(pl.col("eligible").sum())[
        "eligible"
    ].max() == 1
    halted = universe.filter(
        (pl.col("security_id") == "eodhd:symbol:AAA.US")
        & (pl.col("trade_date").dt.day() == 10)
    )
    assert halted["is_halted"].to_list() == [True]
    assert delistings is not None
    assert delistings["delisting_return"].to_list() == [-0.30]
    assert delistings["is_imputed"].to_list() == [True]
    assert delistings["imputation_method"].to_list() == ["exchange_penalty:XNYS"]

    labels = build_forward_excess_return_labels(
        bars,
        universe,
        delistings=delistings,
        horizon=5,
    )
    crossing = labels.filter(
        (pl.col("security_id") == "eodhd:symbol:OLD.US")
        & (pl.col("asof_date").dt.day() == 20)
    ).row(0, named=True)
    assert crossing["label_end"].day == 25
    assert crossing["crosses_delisting"] is True
    assert crossing["raw_return"] is not None


class FakeDiscoveryClient:
    def __init__(self, tmp_path: Path) -> None:
        self.request_log: list[dict[str, Any]] = []
        self.budget = DailyCallBudget(tmp_path / "discovery-budget.json", 1000)

    def get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        call_cost: int = 1,
    ) -> Any:
        self.budget.reserve(call_cost)
        self.request_log.append({"path": path, "params": params or {}, "call_cost": call_cost})
        if path.startswith("exchange-symbol-list/"):
            return [
                {
                    "Code": "AAA",
                    "Name": "Alpha",
                    "Exchange": "NASDAQ",
                    "Currency": "USD",
                    "Type": "Common Stock",
                    "Isin": "US0000000001",
                }
            ]
        if path.startswith("eod-bulk-last-day/"):
            return [
                {
                    "code": "AAA",
                    "date": "2025-01-31",
                    "close": 100,
                    "avgvol_200d": 1_000_000,
                }
            ]
        if path == "eod/AAA.US":
            return EOD_ROWS
        raise AssertionError(f"unexpected request: {path}")


def test_provider_discovers_paid_top_liquid_universe(tmp_path: Path) -> None:
    config = EODHDConfig.model_validate(
        {
            "allow_demo_token": False,
            "universe": {"mode": "top_liquid", "max_symbols": 1},
            "start": "2025-01-01",
            "end": "2025-01-31",
            "output_dir": tmp_path / "bronze",
            "cache_dir": tmp_path / "cache",
            "state_dir": tmp_path / "state",
        }
    )
    client = FakeDiscoveryClient(tmp_path)

    result = EODHDProvider(config, client=client).ingest()

    assert result.manifest["symbols"] == ["AAA.US"]
    assert result.manifest["selection"]["selected_count"] == 1
    assert result.manifest["selection"]["research_ready"] is False
    assert result.manifest["budget"]["api_calls"] == 102
    assert pl.read_parquet(result.files["bars"])["security_id"][0] == ("eodhd:isin:US0000000001")
