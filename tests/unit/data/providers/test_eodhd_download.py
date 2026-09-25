from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from facdigger.cli import app
from facdigger.data.providers.eodhd.client import (
    DailyCallBudget,
    EODHDBudgetError,
    EODHDClient,
    EODHDError,
)
from facdigger.data.providers.eodhd.config import EODHDConfig
from facdigger.data.providers.eodhd.provider import EODHDProvider

NOW = datetime(2026, 9, 25, tzinfo=timezone.utc)


class Response:
    status_code = 200
    headers = {}

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class Transport:
    def __init__(self):
        self.usage = {
            "apiRequestsDate": "2026-09-25",
            "dailyRateLimit": 100,
            "apiRequests": 0,
            "apiKey": "fake-secret",
            "email": "private@example.invalid",
        }
        self.calls = []
        self.history_status = 200

    def get(self, url, *, params, timeout):
        path = url.split("/api/")[1]
        self.calls.append(path)
        if path == "user":
            return Response(dict(self.usage))
        self.usage["apiRequests"] += 1
        if path.startswith("exchange-symbol-list/"):
            symbol = "OLD" if params["delisted"] else "NEW"
            return Response([{"Code": symbol, "Exchange": "NYSE", "Type": "Common Stock"}])
        response = Response([])
        response.status_code = self.history_status
        return response


def client_for(tmp_path, transport, monkeypatch):
    clock = [0.0]
    sleeps = []

    def sleep(delay):
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr("facdigger.data.providers.eodhd.client.time.monotonic", lambda: clock[0])
    client = EODHDClient(
        base_url="https://example.invalid/api",
        api_token="fake-secret",
        cache_dir=tmp_path / "cache",
        budget=DailyCallBudget(tmp_path / "budget.json", 100, now=lambda: NOW),
        transport=transport,
        now=lambda: NOW,
        sleep=sleep,
        max_retries=1,
    )
    return client, sleeps


def test_account_is_live_free_and_never_caches_identity(tmp_path, monkeypatch):
    transport = Transport()
    client, _ = client_for(tmp_path, transport, monkeypatch)
    assert client.account_usage()["used_today"] == 0
    transport.usage["apiRequests"] = 7
    assert client.account_usage()["remaining_today"] == 93
    assert client.budget.status()["remaining"] == 100
    assert transport.calls == ["user", "user"]
    assert not client.cache_dir.exists()
    serialized = json.dumps([client.account_usage(), client.request_log])
    assert "fake-secret" not in serialized and "private@" not in serialized


@pytest.mark.parametrize("value", [-1, 1.5, True, None, "bad"])
def test_invalid_account_quota_fails_before_paid_request(tmp_path, monkeypatch, value):
    transport = Transport()
    transport.usage["apiRequests"] = value
    client, _ = client_for(tmp_path, transport, monkeypatch)
    client.enable_download_guard(reserve_calls=5, requests_per_minute=60)
    with pytest.raises(EODHDError, match="quota fields"):
        client.get_json("eod/NEW.US")
    assert transport.calls == ["user"]


def test_lazy_midnight_reset_and_future_date(tmp_path, monkeypatch):
    transport = Transport()
    transport.usage.update(apiRequestsDate="2026-09-24", apiRequests=100)
    client, _ = client_for(tmp_path, transport, monkeypatch)
    assert client.account_usage()["remaining_today"] == 100
    transport.usage["apiRequestsDate"] = "2026-09-26"
    with pytest.raises(EODHDError, match="quota fields"):
        client.account_usage()


def test_reserve_stops_paid_requests_but_allows_cached_reads(tmp_path, monkeypatch):
    transport = Transport()
    client, sleeps = client_for(tmp_path, transport, monkeypatch)
    client.enable_download_guard(reserve_calls=99, requests_per_minute=60)
    assert client.get_json("eod/NEW.US") == []
    assert client.get_json("eod/NEW.US") == []
    with pytest.raises(EODHDBudgetError, match="reserve reached"):
        client.get_json("eod/OLD.US")
    assert transport.calls == ["user", "eod/NEW.US", "user"]
    assert sleeps == [1.0, 1.0]
    assert client.budget.status()["api_calls"] == 1


def test_retry_rechecks_live_quota_and_cannot_spend_reserved_calls(tmp_path, monkeypatch):
    transport = Transport()
    transport.history_status = 429
    client, _ = client_for(tmp_path, transport, monkeypatch)
    client.enable_download_guard(reserve_calls=99, requests_per_minute=60)
    with pytest.raises(EODHDBudgetError, match="reserve reached"):
        client.get_json("eod/NEW.US")
    assert transport.calls == ["user", "eod/NEW.US", "user"]
    assert client.budget.status()["api_calls"] == 1
    with pytest.raises(ValueError, match="zero API-call cost"):
        client.get_json("eod/NEW.US", call_cost=0)


def test_plan_counts_all_candidates_not_daily_top_n_and_reuses_cache(tmp_path, monkeypatch):
    transport = Transport()
    client, _ = client_for(tmp_path, transport, monkeypatch)
    config = EODHDConfig.model_validate(
        {
            "universe": {"mode": "historical_liquid", "max_symbols": 1},
            "allow_demo_token": False,
            "delisting_imputation": {"enabled": True},
            "start": "2010-01-01",
            "end": "2025-12-31",
            "include_corporate_actions": True,
        }
    )
    provider = EODHDProvider(config, client)
    provider.configure_download(reserve_calls=95, requests_per_minute=60)
    report = provider.plan_historical_download(reserve_calls=95)
    assert report["selection"]["candidate_count"] == 2
    assert report["history_requests"] == 6
    assert report["minimum_remaining_api_calls"] == 6
    assert report["maximum_calls_with_retries"] == 18
    assert report["available_download_calls_today"] == 3
    assert report["minimum_download_fits_today"] is False
    assert set(transport.calls) == {"user", "exchange-symbol-list/US"}
    client.get_json(
        "eod/NEW.US",
        {
            "from": "2010-01-01",
            "to": "2025-12-31",
            "period": "d",
            "order": "a",
        },
    )
    transport.calls.clear()
    report = provider.plan_historical_download(reserve_calls=95)
    assert report["cached_history_requests"] == 1
    assert report["minimum_remaining_api_calls"] == 5
    assert transport.calls == ["user"]
    assert report["account"]["extra_calls_included"] is False


def test_ingest_cli_keeps_old_behavior_and_only_enables_guard_explicitly(tmp_path, monkeypatch):
    transport = Transport()
    client, _ = client_for(tmp_path, transport, monkeypatch)
    provider = EODHDProvider(EODHDConfig(symbols=["NEW.US"]), client)
    config_path = tmp_path / "provider.yaml"
    config_path.write_text("provider: eodhd\nsymbols: [NEW.US]\n")
    monkeypatch.setattr(
        "facdigger.data.providers.registry.provider_from_config", lambda _: provider
    )
    observed = []

    def ingest():
        client.get_json(f"eod/SYMBOL{len(observed)}.US")
        observed.append(list(transport.calls))
        return SimpleNamespace(
            provider="eodhd", output_dir=tmp_path, files={}, manifest={"warnings": []}
        )

    monkeypatch.setattr(provider, "ingest", ingest)
    runner = CliRunner()
    args = ["data", "ingest", "--config", str(config_path)]
    legacy = runner.invoke(app, args)
    assert legacy.exit_code == 0, legacy.output
    assert observed[0] == ["eod/SYMBOL0.US"]
    guarded = runner.invoke(
        app, [*args, "--reserve-api-calls", "95", "--requests-per-minute", "60"]
    )
    assert guarded.exit_code == 0, guarded.output
    assert observed[1] == ["eod/SYMBOL0.US", "user", "eod/SYMBOL1.US"]
    invalid = runner.invoke(app, [*args, "--requests-per-minute", "60"])
    assert invalid.exit_code == 1 and "requires --reserve-api-calls" in invalid.output
    assert len(observed) == 2
