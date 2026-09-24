"""Fresh fake HTTP -> adjustment recovery -> real finance inference and delivery."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from types import SimpleNamespace

import factor_fixtures
import polars as pl
import pytest
import yaml
from test_finance_factor_delivery import finance_delivery  # noqa: F401

from facdigger.data.providers.eodhd.client import DailyCallBudget, EODHDClient
from facdigger.data.providers.eodhd.config import EODHDConfig
from facdigger.data.providers.eodhd.daily import fetch_daily_revision
from facdigger.data.providers.eodhd.provider import EODHDProvider
from facdigger.data.session_store import initialize_production_store, load_current_revision
from facdigger.data.snapshots import sha256_file
from facdigger.inference import factor_batch
from facdigger.production import runner
from facdigger.production.calendar import NEW_YORK
from facdigger.production.config import ProductionServiceConfig
from facdigger.production.state import ProductionState


@pytest.mark.parametrize("scenario", ["price", "identity_elsewhere", "identity_delivery"])
def test_backfill_failure_local_quarantine_restart_and_real_delivery(
    request, tmp_path, monkeypatch, scenario,
):
    _, _, release_dir, release, _ = request.getfixturevalue("finance_delivery")
    days = factor_fixtures.sessions(165)
    target = days[-1]
    isolated_index = 20 if scenario == "identity_elsewhere" else 0
    clock = SimpleNamespace(now=datetime.combine(target, time(20), tzinfo=NEW_YORK))

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.now.astimezone(tz)

    class Transport:
        revised = False
        interrupt = True
        calls = 0

        def get(self, url, params, timeout):
            self.calls += 1
            path = url.removeprefix("https://fixture.invalid/api/")
            if path.startswith("exchange-symbol-list/"):
                data = [] if params["delisted"] else [
                    {"Code": f"S{i}", "Exchange": "NASDAQ", "Type": "Common Stock",
                     "Isin": ("US9000000000" if self.revised and scenario != "price"
                              and i == isolated_index else f"US{i:010d}")} for i in range(21)
                ]
            else:
                if path.startswith("eod/") and self.interrupt:
                    self.interrupt = False
                    raise ConnectionError("synthetic interruption during targeted history")
                if path.startswith("eod-bulk"):
                    requested = [day for day in days if str(day) == params["date"]]
                    indices = range(21)
                else:
                    requested = [
                        day for day in days if params["from"] <= str(day) <= params["to"]
                    ]
                    indices = [int(path.removeprefix("eod/S").removesuffix(".US"))]
                data = []
                for day in requested:
                    for index in indices:
                        close = 20.0 + index + days.index(day) * 0.02
                        adjusted = close * (0.5 if self.revised else 1.0)
                        if (self.revised and scenario == "price"
                                and path.startswith("eod/") and index == 0):
                            if day == days[-25]:
                                adjusted *= 0.02  # Outside the ten-day bulk window.
                        data.append({"code": f"S{index}", "date": str(day), "open": close,
                                     "high": close + 1, "low": close - 1, "close": close,
                                     "adjusted_close": adjusted, "volume": 1_000_000 + index})
            return SimpleNamespace(status_code=200, json=lambda: data)

    transport = Transport()
    provider = EODHDConfig.model_validate({
        "allow_demo_token": False, "refresh": True, "cache_ttl_hours": 0,
        "base_url": "https://fixture.invalid/api",
        "universe": {"mode": "historical_liquid", "max_symbols": 1000},
        "cache_dir": tmp_path / "cache", "state_dir": tmp_path / "api-state",
        "min_listed_sessions": 1, "min_price": 1, "min_adv20_usd": 1,
        "delisting_imputation": {"enabled": True},
    })
    client = EODHDClient(
        base_url="https://fixture.invalid/api", api_token="test-only-not-a-secret",
        cache_dir=provider.cache_dir, budget=DailyCallBudget(tmp_path / "budget.json", 100000),
        refresh=True, cache_ttl_hours=0, max_retries=0, transport=transport,
    )
    initial = fetch_daily_revision(
        client, provider, revision_start=days[0], target_date=days[-2],
    )
    store = tmp_path / "production-source"
    current = initialize_production_store(initial, provider, store, history_sessions=40)
    initial_hash = sha256_file(current.root / "bars_daily.parquet")
    provider_path = tmp_path / "provider.yaml"
    provider_path.write_text(yaml.safe_dump(provider.model_dump(mode="json")))
    config = ProductionServiceConfig.model_validate({
        "data": {"provider_config": provider_path, "store_root": store,
                 "bootstrap_source": tmp_path / "unused"},
        "model": {"release_root": release_dir.parent, "release_id": release.release_id},
        "inference": {"output_root": tmp_path / "production-snapshots",
                      "minimum_candidate_rows": 21, "minimum_eligible_rows": 20},
        "factor_batch": {"output_root": tmp_path / "production-factors", "delivery": {
            "targets": [{"instrument_id": f"S{i}"} for i in range(5)],
            "identities": [{"instrument_id": f"S{i}", "security_id": f"eodhd:isin:US{i:010d}",
                            "valid_from": target, "valid_to": target,
                            "evidence": "Synthetic test identity"} for i in range(5)],
        }},
        "state_database": tmp_path / "production.sqlite3",
    })
    transport.revised = True
    monkeypatch.setattr(runner, "EODHDProvider", lambda cfg: EODHDProvider(cfg, client=client))
    monkeypatch.setattr(factor_batch, "datetime", Clock)
    first = runner.run_production_tick(config, now=clock.now, now_provider=lambda: clock.now)
    assert first.action == "waiting_data" and "ConnectionError" in first.error
    assert load_current_revision(store).revision_id == current.revision_id
    clock.now += timedelta(minutes=30)
    result = runner.run_production_tick(config, now=clock.now, now_provider=lambda: clock.now)
    if scenario == "identity_delivery":
        assert result.action == "blocked" and "delivery identity is unresolved" in result.error
        assert not config.factor_batch.output_root.exists()
        return
    assert result.action == "published", result.error
    assert result.quality["status"] == "degraded"
    assert result.quality["computation"]["candidate_rows"] == 21
    assert result.quality["computation"]["eligible_rows"] == 20
    assert result.quality["computation"]["missing_fraction"] == 1 / 21
    assert result.quality["unscorable"][0]["reason"] == (
        "source_quality_quarantined" if scenario == "price" else "unresolved_security_identity"
    )
    assert result.quality["violations"] == []
    updated = load_current_revision(store)
    assert updated.manifest["quarantines"][0]["reasons"] == [
        "extreme_adjusted_return" if scenario == "price" else "identity_change_pending",
    ]
    assert (f"S{isolated_index}.US"
            not in updated.manifest["daily_update"]["backfilled_provider_symbols"])
    assert pl.read_parquet(updated.root / "bars_daily.parquet").filter(
        pl.col("symbol") == f"S{isolated_index}"
    ).is_empty()
    assert sha256_file(current.root / "bars_daily.parquet") == initial_hash
    bundle = config.factor_batch.output_root / result.delivery_id
    manifest = factor_batch.load_factor_batch(bundle)
    rows = pl.read_parquet(bundle / "factors.parquet")
    assert rows.height == 5 and rows["eligible"].sum() == (4 if scenario == "price" else 5)
    assert rows.filter(~pl.col("eligible"))["score"].to_list() == (
        [None] if scenario == "price" else []
    )
    assert manifest.source.kind == "signal_inference"
    calls = transport.calls
    assert runner.run_production_tick(
        config, now=clock.now, now_provider=lambda: clock.now,
    ).delivery_id == result.delivery_id
    assert transport.calls == calls
    with ProductionState(config.state_database) as state:
        assert state.get(target).attempts == 2
        assert state.get(target).quality_reference["security_ids"] == [
            f"eodhd:isin:US{i:010d}" for i in range(21)
        ]
