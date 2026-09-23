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


def test_live_bootstrap_fetches_full_warmup_before_current_session(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from facdigger.data.market_calendar import regular_sessions
    from facdigger.data.providers.eodhd.config import EODHDConfig
    from facdigger.production import runner

    config = _config(tmp_path)
    provider = EODHDConfig(
        allow_demo_token=False,
        universe={"mode": "historical_liquid", "max_symbols": 1000},
        delisting_imputation={"enabled": True},
        min_listed_sessions=252, refresh=True, cache_ttl_hours=0,
    )
    release = SimpleNamespace(feature_contract=SimpleNamespace(context_length=512))
    monkeypatch.setattr(runner, "_load_fixed_release", lambda _: (tmp_path, release))
    monkeypatch.setattr(runner, "load_eodhd_config", lambda _: provider)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 20, 16, tzinfo=NY).astimezone(tz)

    monkeypatch.setattr(runner, "datetime", Clock)
    client, revision, stored = object(), object(), object()
    calls = {}

    def make_provider(initial):
        calls["initial"] = initial
        return SimpleNamespace(client=lambda: client)

    def fetch(actual_client, initial, **dates):
        assert actual_client is client and initial is calls["initial"]
        calls.update(dates)
        return revision

    def initialize(actual_revision, actual_config, root, *, history_sessions):
        assert actual_revision is revision and actual_config is provider
        assert root == config.data.store_root and history_sessions == 532
        return stored

    monkeypatch.setattr(runner, "EODHDProvider", make_provider)
    monkeypatch.setattr(runner, "fetch_daily_revision", fetch)
    monkeypatch.setattr(runner, "initialize_production_store", initialize)
    assert runner.bootstrap_store(config, live=True) is stored
    assert calls["target_date"] == date(2026, 9, 18)
    assert len(regular_sessions(calls["revision_start"], calls["target_date"])) == 784
    assert calls["initial"].universe.max_symbols == 1000
    assert not calls["initial"].refresh and calls["initial"].cache_ttl_hours == 24
    # Only the initial history may use recent raw cache. Daily settings stay fresh.
    assert provider.refresh and provider.cache_ttl_hours == 0


def test_live_bootstrap_rejects_existing_store_before_any_provider_access(tmp_path, monkeypatch):
    from facdigger.data.contracts import DataContractError
    from facdigger.production import runner

    config = _config(tmp_path)
    config.data.store_root.mkdir()
    marker = config.data.store_root / "CURRENT"
    marker.write_text("original-revision\n")
    monkeypatch.setattr(runner, "_load_fixed_release", lambda _: (tmp_path, object()))

    def no_provider_access(_):
        pytest.fail("existing production source must be rejected before provider access")

    monkeypatch.setattr(runner, "load_eodhd_config", no_provider_access)
    with pytest.raises(DataContractError, match="empty production store_root"):
        runner.bootstrap_store(config, live=True)
    assert marker.read_text() == "original-revision\n"


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


@pytest.fixture
def daily_tick(tmp_path, monkeypatch):
    """Real daily ledger/gates, isolated source IO and model execution."""
    from types import SimpleNamespace

    import polars as pl

    from facdigger.data.market_calendar import regular_sessions
    from facdigger.inference.factor_batch import build_factor_frame
    from tests.unit.production.test_quality import _delivery

    days = regular_sessions(date(2026, 6, 1), date(2026, 8, 17))
    history = pl.DataFrame([
        {"security_id": f"sec-{i}", "symbol": f"S{i}", "trade_date": day, "eligible": True}
        for day in days for i in range(20)
    ])
    source = tmp_path / "source"
    source.mkdir()
    history.filter(pl.col("trade_date") < days[-1]).write_parquet(
        source / "universe_daily.parquet"
    )
    current = SimpleNamespace(
        root=source, revision_id="fixture", manifest={"resolved_end": days[-2].isoformat()},
    )
    controls = SimpleNamespace(
        fetches=0, source_missing={}, window_missing={}, published=[], latest_candidates=None,
        history=history, current=current,
    )
    config = _config(tmp_path).model_dump()
    config["factor_batch"]["delivery"] = _delivery().model_dump()
    config["inference"].update(minimum_candidate_rows=20, minimum_eligible_rows=18)
    config = ProductionServiceConfig.model_validate(config)
    monkeypatch.setattr("facdigger.production.runner._load_fixed_release", lambda _: (
        tmp_path / "release", SimpleNamespace(feature_contract=SimpleNamespace(context_length=20)),
    ))
    monkeypatch.setattr("facdigger.production.runner.load_current_revision", lambda _: current)
    monkeypatch.setattr("facdigger.production.runner.load_eodhd_config", lambda _: (
        SimpleNamespace(refresh=True, cache_ttl_hours=0)
    ))
    monkeypatch.setattr("facdigger.production.runner.EODHDProvider", lambda _: (
        SimpleNamespace(client=lambda: object())
    ))

    def fetch(*args, **kwargs):
        controls.fetches += 1
        return SimpleNamespace(backfilled_provider_symbols=())

    def publish(*args, **kwargs):
        missing = controls.source_missing.get(controls.fetches, set())
        frame = history.with_columns(
            ~((pl.col("trade_date") == days[-1]) & pl.col("security_id").is_in(list(missing)))
            .alias("eligible")
        )
        frame.write_parquet(source / "universe_daily.parquet")
        frame.filter(pl.col("eligible")).select("security_id", "trade_date").write_parquet(
            source / "bars_daily.parquet"
        )
        current.manifest["resolved_end"] = days[-1].isoformat()
        return current

    def load_snapshot(*args, **kwargs):
        frame = pl.read_parquet(source / "universe_daily.parquet").filter(
            pl.col("trade_date") == days[-1]
        ).select("security_id", "symbol", pl.col("trade_date").alias("asof_date"), "eligible")
        window_missing = controls.window_missing.get(controls.fetches, set())
        frame = frame.with_columns(
            (pl.col("eligible") & ~pl.col("security_id").is_in(list(window_missing)))
            .alias("eligible")
        ).sort("asof_date", "security_id")
        controls.latest_candidates = frame
        return {}, {"delivery_universe": frame}

    def infer(*args, **kwargs):
        kwargs["before_publish"]()
        from facdigger.inference.delivery import resolve_delivery

        candidates = controls.latest_candidates
        scores = candidates.filter(pl.col("eligible")).select(
            "security_id", "asof_date", pl.lit(1.0).alias("score"),
        )
        selection = resolve_delivery(candidates, kwargs["delivery"])
        controls.published.append(build_factor_frame(
            selection.candidates, selection.project_scores(scores),
        ))
        return tmp_path / "bundle", {
            "time": {"maximum_asof_date": days[-1].isoformat()}, "delivery_id": "delivery",
        }

    monkeypatch.setattr("facdigger.production.runner.fetch_daily_revision", fetch)
    monkeypatch.setattr("facdigger.production.runner.require_fresh_daily_requests", lambda _: None)
    monkeypatch.setattr("facdigger.production.runner.publish_daily_source_revision", publish)
    monkeypatch.setattr("facdigger.production.runner.build_inference_snapshot", lambda *a, **kw: (
        tmp_path / "snapshot", {
            "snapshot_id": "snapshot", "feature_contract": {"feature_set": "price_volume_v1"},
        },
    ))
    monkeypatch.setattr("facdigger.production.runner.load_inference_snapshot", load_snapshot)
    monkeypatch.setattr("facdigger.production.runner.run_signal_inference", infer)
    return config, controls


def _tick(config, minute=0):
    observed = datetime(2026, 8, 17, 20, minute, tzinfo=NY)
    return run_production_tick(config, now=observed, now_provider=lambda: observed)


def test_partial_data_publishes_explicit_null_without_old_factor(daily_tick):
    import polars as pl

    config, controls = daily_tick
    controls.source_missing[1] = {"sec-0"}
    result = _tick(config)
    assert result.action == "published", result.error
    assert result.quality["status"] == "degraded"
    assert result.quality["unscorable"][0]["reason"] == "missing_target_bar"
    frame = controls.published[0]
    assert frame.height == 5 and frame["eligible"].sum() == 4
    assert frame.filter(~pl.col("eligible"))["score"].to_list() == [None]
    assert frame["asof_date"].unique().to_list() == [date(2026, 8, 17)]
    assert _tick(config, 30).action == "already_published"
    assert controls.fetches == 1


def test_retry_refetches_after_source_commit_and_inference_shortfall(daily_tick):
    config, controls = daily_tick
    controls.window_missing[1] = {"sec-0", "sec-1"}
    first = _tick(config)
    assert first.action == "waiting_data", first.error
    assert controls.current.manifest["resolved_end"] == "2026-08-17"
    assert first.quality["stage"] == "inference"
    assert _tick(config, 10).action == "retry_wait"
    second = _tick(config, 30)
    assert second.action == "published", second.error
    assert second.quality["status"] == "ready"
    assert controls.fetches == 2
    assert second.quality["computation"]["reference_eligible_rows"] == 20


def test_retries_keep_original_denominator_when_membership_is_reduced(daily_tick):
    config, controls = daily_tick
    controls.source_missing = {1: {"sec-18", "sec-19"}, 2: {"sec-18", "sec-19"}}
    first = _tick(config)
    second = _tick(config, 30)
    assert first.action == second.action == "waiting_data"
    assert second.quality["computation"]["reference_eligible_rows"] == 20
    assert second.quality["computation"]["missing_fraction"] == 0.1
    assert controls.fetches == 2 and not controls.published
    observed = datetime(2026, 8, 18, 9, 30, tzinfo=NY)
    expired = run_production_tick(config, now=observed, now_provider=lambda: observed)
    assert expired.action == "expired"
    assert expired.quality["status"] == "insufficient"
    assert not controls.published


def test_model_failure_is_not_reclassified_as_missing_input(daily_tick, monkeypatch):
    from facdigger.data.contracts import DataContractError

    config, controls = daily_tick
    def fail(*args, **kwargs):
        raise DataContractError("eligible score is NaN")

    monkeypatch.setattr("facdigger.production.runner.run_signal_inference", fail)
    result = _tick(config)
    assert result.action == "blocked"
    assert result.quality["status"] == "ready"
    assert "NaN" in result.error
    assert not controls.published


@pytest.mark.parametrize("interrupted_after_commit", [False, True])
def test_reduced_day_cannot_become_next_days_smaller_reference(
    daily_tick, monkeypatch, interrupted_after_commit,
):
    from facdigger.data.providers.eodhd.daily import DailyDataNotReady

    config, controls = daily_tick
    controls.source_missing[1] = {"sec-18", "sec-19"}
    if interrupted_after_commit:
        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt("process stopped after source commit")

        monkeypatch.setattr("facdigger.production.runner.assess_daily_quality", interrupted)
        with pytest.raises(KeyboardInterrupt):
            _tick(config)
    else:
        assert _tick(config).action == "waiting_data"

    def unavailable(*args, **kwargs):
        raise DailyDataNotReady("next day is not ready yet")

    monkeypatch.setattr("facdigger.production.runner.fetch_daily_revision", unavailable)
    observed = datetime(2026, 8, 18, 20, 0, tzinfo=NY)
    result = run_production_tick(config, now=observed, now_provider=lambda: observed)
    assert result.action == "waiting_data"
    with ProductionState(config.state_database) as state:
        reference = state.get(observed.date()).quality_reference
    assert reference["target_date"] == "2026-08-18"
    assert reference["reference_date"] == "2026-08-14"
    assert len(reference["security_ids"]) == 20


def test_expiry_retains_actual_failure_and_is_idempotent(tmp_path, monkeypatch, caplog):
    import json
    import logging
    from datetime import timedelta

    from facdigger.data.providers.eodhd.daily import DailyDataNotReady

    config = _config(tmp_path)
    observed = datetime(2026, 9, 22, 20, tzinfo=NY)
    caplog.set_level(logging.INFO, logger="facdigger.production.runner")

    def fail(_):
        raise DailyDataNotReady("MGN source quality unavailable")

    monkeypatch.setattr("facdigger.production.runner._load_fixed_release", fail)
    for minutes in [0, 30]:
        clock = observed + timedelta(minutes=minutes)
        result = run_production_tick(config, now=clock, now_provider=lambda clock=clock: clock)
        assert result.action == "waiting_data"
    events = [json.loads(record.message) for record in caplog.records]
    finished = [event for event in events if event["event"] == "production_attempt_finished"]
    assert [event["attempt"] for event in finished] == [1, 2]
    assert all("MGN" in event["error"] for event in finished)
    assert all(event["last_stage"] == "load_source" for event in finished)
    clock = datetime(2026, 9, 23, 9, 30, tzinfo=NY)
    expired = run_production_tick(config, now=clock, now_provider=lambda: clock)
    assert expired.action == "expired" and "MGN" in expired.error and "cutoff_at=" in expired.error
    with ProductionState(config.state_database) as state:
        before = state.get(observed.date())
    repeated = run_production_tick(config, now=clock, now_provider=lambda: clock)
    assert repeated.action == "not_due" and repeated.target_date == clock.date()
    with ProductionState(config.state_database) as state:
        assert state.get(observed.date()) == before
