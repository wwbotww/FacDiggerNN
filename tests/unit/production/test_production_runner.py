from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from facdigger.production.config import ProductionServiceConfig
from facdigger.production.runner import run_production_tick
from facdigger.production.state import ProductionState

NY = ZoneInfo("America/New_York")


def _config(tmp_path) -> ProductionServiceConfig:
    return ProductionServiceConfig.model_validate(
        {
            "data": {
                "provider_config": tmp_path / "provider.yaml",
                "store_root": tmp_path / "store",
                "bootstrap_source": tmp_path / "bronze",
            },
            "model": {
                "release_root": tmp_path / "releases",
                "release_id": "1" * 64,
            },
            "inference": {"output_root": tmp_path / "inference"},
            "factor_batch": {"output_root": tmp_path / "factors"},
            "state_database": tmp_path / "production.sqlite3",
        }
    )


def test_tick_does_not_touch_release_or_data_before_first_attempt(tmp_path) -> None:
    config = _config(tmp_path)
    result = run_production_tick(
        config,
        now=datetime(2026, 8, 17, 10, 0, tzinfo=NY),
    )

    assert result.action == "not_due"
    assert result.target_date.isoformat() == "2026-08-17"


def test_expired_target_is_terminal_and_never_falls_back(tmp_path) -> None:
    config = _config(tmp_path)
    target_time = datetime(2026, 8, 17, 9, 30, tzinfo=NY)
    with ProductionState(config.state_database) as state:
        state.put(
            date(2026, 8, 14),
            config.model.release_id,
            "waiting_data",
            attempts=3,
        )

    result = run_production_tick(config, now=target_time)

    assert result.action == "expired"
    assert result.target_date.isoformat() == "2026-08-14"
    assert result.delivery_id is None


def test_published_target_is_idempotent_without_loading_release(tmp_path) -> None:
    config = _config(tmp_path)
    target = datetime(2026, 8, 17, 20, 0, tzinfo=NY)
    with ProductionState(config.state_database) as state:
        state.put(
            target.date(),
            config.model.release_id,
            "published",
            attempts=1,
            snapshot_id="snapshot",
            delivery_id="delivery",
        )

    result = run_production_tick(config, now=target)

    assert result.action == "already_published"
    assert result.snapshot_id == "snapshot"
    assert result.delivery_id == "delivery"


def test_target_cannot_switch_release_after_entering_production(tmp_path) -> None:
    config = _config(tmp_path)
    target = datetime(2026, 8, 17, 20, 0, tzinfo=NY)
    with ProductionState(config.state_database) as state:
        state.put(target.date(), "2" * 64, "waiting_data", attempts=1)

    with pytest.raises(Exception, match="release_id"):
        run_production_tick(config, now=target)


def test_transient_provider_failure_waits_exactly_thirty_minutes(
    tmp_path,
    monkeypatch,
) -> None:
    from facdigger.data.providers.eodhd.daily import DailyDataNotReady

    config = _config(tmp_path)
    observed = datetime(2026, 8, 17, 20, 0, tzinfo=NY)
    monkeypatch.setattr(
        "facdigger.production.runner._load_fixed_release",
        lambda _: (_ for _ in ()).throw(DailyDataNotReady("not ready")),
    )

    result = run_production_tick(config, now=observed, now_provider=lambda: observed)

    assert result.action == "waiting_data"
    assert result.next_retry_at == datetime(2026, 8, 17, 20, 30, tzinfo=NY)
    assert result.attempts == 1

    waiting = run_production_tick(
        config,
        now=datetime(2026, 8, 17, 20, 10, tzinfo=NY),
    )
    assert waiting.action == "retry_wait"
    assert waiting.attempts == 1
