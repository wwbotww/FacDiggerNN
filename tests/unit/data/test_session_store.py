from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from facdigger.data.contracts import DataContractError, table_audit, validate_bars
from facdigger.data.market_calendar import regular_session_frame
from facdigger.data.provenance import build_standardization_contract
from facdigger.data.providers.eodhd.config import EODHDConfig
from facdigger.data.providers.eodhd.daily import EODHDDailyRevision
from facdigger.data.providers.eodhd.mapper import build_universe
from facdigger.data.session_store import (
    AdjustmentBackfillRequired,
    TargetSessionIncomplete,
    bootstrap_production_store,
    initialize_production_store,
    load_current_revision,
    prune_source_revisions,
    publish_daily_source_revision,
)
from facdigger.data.snapshots import sha256_file


def _bars(days: list[date], *, changed_last_factor: float = 1.0) -> pl.DataFrame:
    records = []
    for index, day in enumerate(days):
        for symbol, isin, base in [
            ("AAA.US", "US0000000001", 10.0),
            ("BBB.US", "US0000000002", 20.0),
        ]:
            close = base + index * 0.1
            records.append(
                {
                    "security_id": f"eodhd:isin:{isin}",
                    "symbol": symbol.split(".")[0],
                    "trade_date": day,
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": 1_000_000.0,
                    "dollar_volume": close * 1_000_000.0,
                    "adj_factor": (
                        changed_last_factor
                        if symbol == "AAA.US" and day == days[-1]
                        else 1.0
                    ),
                    "source_revision": "source",
                    "ingested_at": datetime(2026, 8, 1, tzinfo=timezone.utc),
                    "provider": "eodhd",
                    "provider_symbol": symbol,
                    "adjusted_close": close,
                    "adjustment_basis": "test",
                    "identity_quality": "isin",
                    "exchange_source": "NASDAQ",
                    "security_type_source": "Common Stock",
                    "is_delisted_source": False,
                }
            )
    return validate_bars(pl.DataFrame(records))


def _provider_config(tmp_path: Path) -> EODHDConfig:
    return EODHDConfig.model_validate(
        {
            "allow_demo_token": False,
            "universe": {"mode": "historical_liquid", "max_symbols": 2},
            "cache_dir": tmp_path / "cache",
            "state_dir": tmp_path / "state",
            "min_listed_sessions": 1,
            "min_price": 1,
            "min_adv20_usd": 1,
            "delisting_imputation": {"enabled": True},
        }
    )


def _bootstrap_source(tmp_path: Path, days: list[date]) -> Path:
    source = tmp_path / "bronze"
    source.mkdir()
    bars = _bars(days)
    universe = build_universe(
        bars,
        min_listed_sessions=1,
        min_price=1,
        min_adv20_usd=1,
        max_daily_symbols=2,
        calendar=regular_session_frame(days[0], days[-1]),
    )
    bars_path = source / "bars_daily.parquet"
    universe_path = source / "universe_daily.parquet"
    bars.write_parquet(bars_path)
    universe.write_parquet(universe_path)
    evidence = {
        "bars": {
            **table_audit(bars, "trade_date"),
            "file": bars_path.name,
            "sha256": sha256_file(bars_path),
        },
        "universe": {
            **table_audit(universe, "trade_date"),
            "file": universe_path.name,
            "sha256": sha256_file(universe_path),
        },
    }
    (source / "eodhd_ingestion_manifest.json").write_text(
        json.dumps(
            {
                "provider": "eodhd",
                "source_revision": "historical",
                "ingested_at": "2026-08-01T00:00:00+00:00",
                "warnings": [],
                "standardization": build_standardization_contract(
                    evidence,
                    research_ready=False,
                ),
            }
        ),
        encoding="utf-8",
    )
    return source


def _revision(days: list[date], bars: pl.DataFrame) -> EODHDDailyRevision:
    metadata = (
        {
            "Code": "AAA",
            "Exchange": "NASDAQ",
            "Type": "Common Stock",
            "Isin": "US0000000001",
            "_is_delisted": False,
        },
        {
            "Code": "BBB",
            "Exchange": "NASDAQ",
            "Type": "Common Stock",
            "Isin": "US0000000002",
            "_is_delisted": False,
        },
    )
    return EODHDDailyRevision(
        target_date=days[-1],
        revision_start=days[0],
        bars=bars,
        metadata_rows=metadata,
        symbol_count=2,
        rejected_rows=0,
        request_log=(),
        source_revision="daily",
        ingested_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
    )


def test_store_is_bounded_and_preserves_listed_day_count(tmp_path) -> None:
    historical_days = regular_session_frame(date(2026, 5, 1), date(2026, 7, 29))[
        "trade_date"
    ].to_list()
    source = _bootstrap_source(tmp_path, historical_days)
    store = tmp_path / "store"
    current = bootstrap_production_store(source, store, history_sessions=40)
    assert pl.read_parquet(current.root / "bars_daily.parquet")["trade_date"].n_unique() == 40
    previous_universe = pl.read_parquet(current.root / "universe_daily.parquet")
    assert previous_universe["trade_date"].n_unique() == 40

    next_days = [date(2026, 7, 28), date(2026, 7, 29), date(2026, 7, 30)]
    updated = publish_daily_source_revision(
        current,
        _revision(next_days, _bars(next_days)),
        _provider_config(tmp_path),
        store,
        history_sessions=40,
    )

    bars = pl.read_parquet(updated.root / "bars_daily.parquet")
    universe = pl.read_parquet(updated.root / "universe_daily.parquet")
    assert bars["trade_date"].n_unique() == universe["trade_date"].n_unique() == 40
    assert universe.filter(pl.col("trade_date") == date(2026, 7, 30))["listed_days"].min() > 40
    preserved = previous_universe.filter(
        pl.col("trade_date").is_between(bars["trade_date"].min(), next_days[0], closed="left")
    )
    assert_frame_equal(universe.filter(pl.col("trade_date") < next_days[0]), preserved)
    assert set(updated.manifest["production_revision"]) == {
        "contract",
        "parent_revision_id",
        "target_date",
        "revision_start",
        "history_sessions",
    }
    assert len([path for path in (store / "revisions").iterdir() if path.is_dir()]) == 2
    prune_source_revisions(store, keep={updated.revision_id})
    assert [path.name for path in (store / "revisions").iterdir() if path.is_dir()] == [
        updated.revision_id
    ]


def test_live_initialization_preserves_full_dated_universe_and_warmup(tmp_path) -> None:
    days = regular_session_frame(date(2026, 4, 1), date(2026, 7, 31))[
        "trade_date"
    ].to_list()[-60:]
    bars = _bars(days).with_columns(
        pl.when((pl.col("symbol") == "AAA") & (pl.col("trade_date") >= days[-5]))
        .then(100_000_000.0).otherwise(pl.col("volume")).alias("volume"),
    ).with_columns((pl.col("close") * pl.col("volume")).alias("dollar_volume"))
    config = _provider_config(tmp_path)
    config.universe.max_symbols = 1
    store = tmp_path / "live"
    current = initialize_production_store(
        _revision(days, bars), config, store, history_sessions=40,
    )
    universe = pl.read_parquet(current.root / "universe_daily.parquet")
    assert universe["trade_date"].unique().sort().to_list() == days[-40:]
    assert universe["security_id"].n_unique() == 2
    assert universe.filter(pl.col("trade_date") == days[-40])["listed_days"].min() == 21
    assert universe.filter(
        (pl.col("trade_date") == days[-40]) & pl.col("eligible")
    )["symbol"].to_list() == ["BBB"]
    assert universe.filter(
        (pl.col("trade_date") == days[-1]) & pl.col("eligible")
    )["symbol"].to_list() == ["AAA"]
    assert current.manifest["bootstrap"]["fetched_sessions"] == 60
    assert current.manifest["quality"]["gate"]["status"] == "passed"
    original_hash = sha256_file(current.root / "eodhd_ingestion_manifest.json")
    with pytest.raises(DataContractError, match="empty production"):
        initialize_production_store(_revision(days, bars), config, store, history_sessions=40)
    assert sha256_file(current.root / "eodhd_ingestion_manifest.json") == original_hash


@pytest.mark.parametrize("invalid", ["missing_session", "short_warmup"])
def test_live_initialization_never_commits_incomplete_source(tmp_path, invalid) -> None:
    days = regular_session_frame(date(2026, 4, 1), date(2026, 7, 31))[
        "trade_date"
    ].to_list()[-60:]
    if invalid == "short_warmup":
        days = days[-40:]
    bars = _bars(days)
    if invalid == "missing_session":
        bars = bars.filter(pl.col("trade_date") != days[-8])
    store = tmp_path / "live"
    with pytest.raises((DataContractError, TargetSessionIncomplete)):
        initialize_production_store(
            _revision(days, bars), _provider_config(tmp_path), store, history_sessions=40,
        )
    assert not (store / "CURRENT").exists()


def test_live_initialization_persists_real_quarantine_audit_dates(tmp_path) -> None:
    days = regular_session_frame(date(2026, 4, 1), date(2026, 7, 31))[
        "trade_date"
    ].to_list()[-60:]
    bars = _bars(days, changed_last_factor=20.0)
    config = _provider_config(tmp_path)
    config.quality_gate.max_quarantined_security_fraction = 0.5
    current = initialize_production_store(
        _revision(days, bars), config, tmp_path / "live", history_sessions=40,
    )
    audit = current.manifest["quality"]["identity"]
    assert audit["quarantined_securities"] == 1
    assert audit["extreme_return_examples"][0]["trade_date"] == days[-1].isoformat()
    assert current.manifest["quality"]["gate"]["status"] == "passed"


def test_quarantined_history_cannot_reenter_from_a_clean_short_window(tmp_path):
    days = regular_session_frame(date(2026, 4, 1), date(2026, 7, 31))[
        "trade_date"
    ].to_list()[-60:]
    config = _provider_config(tmp_path)
    config.quality_gate.max_quarantined_security_fraction = 0.5
    store = tmp_path / "live"
    current = initialize_production_store(
        _revision(days, _bars(days, changed_last_factor=20)), config, store,
        history_sessions=40,
    )
    for _ in range(2):
        current = publish_daily_source_revision(
            current, _revision(days[-10:], _bars(days[-10:])), config, store,
            history_sessions=40,
        )
        bars = pl.read_parquet(current.root / "bars_daily.parquet")
        assert bars.filter(pl.col("symbol") == "AAA").is_empty()
        candidates = pl.read_parquet(current.root / "universe_daily.parquet").filter(
            pl.col("trade_date") == days[-1]
        )
        assert set(candidates["symbol"]) == {"AAA", "BBB"}
        row = candidates.filter(pl.col("symbol") == "AAA").row(0, named=True)
        assert row["eligible"] is False and row["close"] is None
        assert row["trade_status_quality"] == "source_quality_quarantined"


def test_adjustment_change_requires_full_hot_history_backfill(tmp_path) -> None:
    days = regular_session_frame(date(2026, 5, 1), date(2026, 7, 29))["trade_date"].to_list()
    source = _bootstrap_source(tmp_path, days)
    store = tmp_path / "store"
    current = bootstrap_production_store(source, store, history_sessions=40)
    window = days[-3:]
    revised = _bars(window, changed_last_factor=0.5)

    with pytest.raises(AdjustmentBackfillRequired) as caught:
        publish_daily_source_revision(
            current,
            _revision(window, revised),
            _provider_config(tmp_path),
            store,
            history_sessions=40,
        )

    assert caught.value.provider_symbols == ["AAA.US"]
    full_aaa = _bars(days, changed_last_factor=0.5).filter(
        pl.col("provider_symbol") == "AAA.US"
    )
    full_aaa = full_aaa.with_columns(
        pl.when(pl.col("trade_date") == window[0])
        .then(0.5)
        .otherwise(pl.col("adj_factor"))
        .alias("adj_factor")
    )
    complete = replace(
        _revision(window, revised),
        bars=pl.concat(
            [
                revised.filter(pl.col("provider_symbol") != "AAA.US"),
                full_aaa,
            ],
            how="vertical_relaxed",
        ),
        backfilled_provider_symbols=("AAA.US",),
    )
    updated = publish_daily_source_revision(
        current,
        complete,
        _provider_config(tmp_path),
        store,
        history_sessions=40,
    )
    assert updated.manifest["daily_update"]["adjustment_changed_securities"] == [
        "eodhd:isin:US0000000001"
    ]


def test_daily_membership_rebuild_covers_gap_and_matches_full_history(tmp_path) -> None:
    days = regular_session_frame(date(2026, 4, 1), date(2026, 8, 10))["trade_date"].to_list()
    source = _bootstrap_source(tmp_path, days[:-15])
    store = tmp_path / "store"
    current = bootstrap_production_store(source, store, history_sessions=40)
    full_bars = _bars(days)
    revised_days = days[-20:]
    revised_bars = full_bars.filter(pl.col("trade_date").is_in(revised_days))
    updated = publish_daily_source_revision(
        current, _revision(revised_days, revised_bars), _provider_config(tmp_path), store,
        history_sessions=40,
    )
    observed = pl.read_parquet(updated.root / "universe_daily.parquet")
    expected = build_universe(
        full_bars, min_listed_sessions=1, min_price=1, min_adv20_usd=1, max_daily_symbols=2,
        calendar=regular_session_frame(days[0], days[-1]),
    ).filter(pl.col("trade_date").is_in(days[-40:]))
    fields = ["security_id", "trade_date", "listed_days", "eligible", "adv20_usd"]
    assert_frame_equal(observed.select(fields), expected.select(fields))
    assert observed["trade_date"].unique().sort().to_list() == days[-40:]


def test_old_latest_only_store_requires_rebootstrap(tmp_path) -> None:
    days = regular_session_frame(date(2026, 4, 1), date(2026, 8, 10))["trade_date"].to_list()
    source = _bootstrap_source(tmp_path, days)
    store = tmp_path / "store"
    current = bootstrap_production_store(source, store, history_sessions=40)
    universe_path = current.root / "universe_daily.parquet"
    # Reconstruct the previous storage format, including a matching file hash.
    latest = pl.read_parquet(universe_path).filter(pl.col("trade_date") == days[-1])
    latest.write_parquet(universe_path)
    manifest_path = current.root / "eodhd_ingestion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["standardization"]["tables"]["universe"] = {
        **table_audit(latest, "trade_date"), "file": universe_path.name,
        "sha256": sha256_file(universe_path),
    }
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DataContractError, match="re-bootstrap"):
        load_current_revision(store)


def test_insufficient_membership_warmup_cannot_advance_current(tmp_path) -> None:
    days = regular_session_frame(date(2026, 4, 1), date(2026, 8, 10))["trade_date"].to_list()
    source = _bootstrap_source(tmp_path, days[:-1])
    store = tmp_path / "store"
    current = bootstrap_production_store(source, store, history_sessions=20)
    with pytest.raises(DataContractError, match="ADV20 warm-up"):
        publish_daily_source_revision(
            current, _revision(days[-3:], _bars(days[-3:])), _provider_config(tmp_path), store,
            history_sessions=20,
        )
    assert load_current_revision(store).revision_id == current.revision_id


def test_missing_stock_keeps_candidate_row_and_does_not_fill_price(tmp_path):
    days = regular_session_frame(date(2026, 4, 1), date(2026, 8, 10))["trade_date"].to_list()
    source = _bootstrap_source(tmp_path, days[:-1])
    store = tmp_path / "store"
    current = bootstrap_production_store(source, store, history_sessions=40)
    revised = _bars(days[-10:]).filter(
        ~((pl.col("symbol") == "BBB") & (pl.col("trade_date") == days[-1]))
    )
    updated = publish_daily_source_revision(
        current, _revision(days[-10:], revised), _provider_config(tmp_path), store,
        history_sessions=40,
    )
    target = pl.read_parquet(updated.root / "universe_daily.parquet").filter(
        pl.col("trade_date") == days[-1]
    )
    assert target.height == 2
    missing = target.filter(pl.col("symbol") == "BBB")
    assert missing["eligible"].to_list() == [False]
    assert missing["close"].to_list() == [None]
    bars = pl.read_parquet(updated.root / "bars_daily.parquet")
    assert bars.filter((pl.col("symbol") == "BBB") & (pl.col("trade_date") == days[-1])).is_empty()


def test_missing_whole_market_session_is_retryable_and_does_not_advance_store(tmp_path):
    from facdigger.data.session_store import TargetSessionIncomplete

    days = regular_session_frame(date(2026, 4, 1), date(2026, 8, 10))["trade_date"].to_list()
    source = _bootstrap_source(tmp_path, days[:-1])
    store = tmp_path / "store"
    current = bootstrap_production_store(source, store, history_sessions=40)
    revised = _bars(days[-10:]).filter(pl.col("trade_date") != days[-3])
    with pytest.raises(TargetSessionIncomplete, match="missing market sessions"):
        publish_daily_source_revision(
            current, _revision(days[-10:], revised), _provider_config(tmp_path), store,
            history_sessions=40,
        )
    assert load_current_revision(store).revision_id == current.revision_id


@pytest.mark.parametrize("missing_middle", [False, True])
def test_full_backfill_replaces_old_history_even_without_overlap_factor_change(
    tmp_path, missing_middle,
):
    days = regular_session_frame(date(2026, 4, 1), date(2026, 8, 10))["trade_date"].to_list()
    source = _bootstrap_source(tmp_path, days)
    store = tmp_path / "store"
    current = bootstrap_production_store(source, store, history_sessions=40)
    original_hash = sha256_file(current.root / "bars_daily.parquet")
    old = pl.read_parquet(current.root / "bars_daily.parquet")
    changed_day = days[-25]
    complete = old.with_columns(
        pl.when((pl.col("symbol") == "AAA") & (pl.col("trade_date") == changed_day))
        .then(0.5).otherwise(pl.col("adj_factor")).alias("adj_factor"),
    ).filter((pl.col("symbol") == "AAA") | (pl.col("trade_date") >= days[-10]))
    if missing_middle:
        complete = complete.filter(
            ~((pl.col("symbol") == "AAA") & (pl.col("trade_date") == days[-20]))
        )
    revision = replace(_revision(days[-10:], complete), backfilled_provider_symbols=("AAA.US",))
    config = _provider_config(tmp_path)
    config.quality_gate.max_quarantined_security_fraction = 0.5
    updated = publish_daily_source_revision(current, revision, config, store, history_sessions=40)
    assert sha256_file(current.root / "bars_daily.parquet") == original_hash
    observed = pl.read_parquet(updated.root / "bars_daily.parquet")
    target = pl.read_parquet(updated.root / "universe_daily.parquet").filter(
        pl.col("trade_date") == days[-1]
    )
    assert target.height == 2  # Exclusion never shrinks membership.
    if missing_middle:
        assert observed.filter(pl.col("symbol") == "AAA").is_empty()
        assert updated.manifest["quarantines"][0]["reasons"] == ["incomplete_adjustment_history"]
        assert updated.manifest["daily_update"]["backfilled_provider_symbols"] == []
        assert target.filter(pl.col("symbol") == "AAA")["close"].item() is None
    else:
        assert observed.filter(
            (pl.col("symbol") == "AAA") & (pl.col("trade_date") == changed_day)
        )["adj_factor"].item() == 0.5
        assert not updated.manifest["quarantines"]
        assert updated.manifest["daily_update"]["backfilled_provider_symbols"] == ["AAA.US"]


def test_full_clean_context_can_clear_persisted_quarantine(tmp_path):
    days = regular_session_frame(date(2026, 4, 1), date(2026, 8, 10))["trade_date"].to_list()
    config = _provider_config(tmp_path)
    config.quality_gate.max_quarantined_security_fraction = 0.5
    store = tmp_path / "store"
    current = initialize_production_store(
        _revision(days, _bars(days, changed_last_factor=20)), config, store, history_sessions=40,
    )
    full = _bars(days).filter((pl.col("symbol") == "AAA") | (pl.col("trade_date") >= days[-10]))
    revision = replace(_revision(days[-10:], full), backfilled_provider_symbols=("AAA.US",))
    updated = publish_daily_source_revision(current, revision, config, store, history_sessions=40)
    assert updated.manifest["quarantines"] == []
    assert pl.read_parquet(updated.root / "bars_daily.parquet")["security_id"].n_unique() == 2
    target = pl.read_parquet(updated.root / "universe_daily.parquet").filter(
        pl.col("trade_date") == days[-1]
    )
    assert target["eligible"].to_list() == [True, True]


def test_metadata_cannot_silently_relabel_quarantined_isin(tmp_path):
    days = regular_session_frame(date(2026, 4, 1), date(2026, 8, 10))["trade_date"].to_list()
    config = _provider_config(tmp_path)
    config.quality_gate.max_quarantined_security_fraction = 0.5
    store = tmp_path / "store"
    current = initialize_production_store(
        _revision(days, _bars(days, changed_last_factor=20)), config, store, history_sessions=40,
    )
    revision = _revision(days[-10:], _bars(days[-10:]))
    revision = replace(revision, metadata_rows=(
        {**revision.metadata_rows[0], "Isin": "US0000000099"}, revision.metadata_rows[1],
    ))
    original = sha256_file(current.root / "bars_daily.parquet")
    updated = publish_daily_source_revision(current, revision, config, store, history_sessions=40)
    assert sha256_file(current.root / "bars_daily.parquet") == original
    bars = pl.read_parquet(updated.root / "bars_daily.parquet")
    assert bars["symbol"].unique().to_list() == ["BBB"]
    universe = pl.read_parquet(updated.root / "universe_daily.parquet")
    isolated = universe.filter(pl.col("symbol") == "AAA")
    assert isolated["security_id"].unique().to_list() == ["eodhd:isin:US0000000001"]
    assert isolated["eligible"].sum() == 0
    assert isolated["close"].null_count() == isolated.height
    assert "identity_change_pending" in updated.manifest["quarantines"][0]["reasons"]


def test_legacy_quarantine_recovery_requires_valid_parent_evidence(tmp_path):
    from facdigger.data.session_store import _source_quarantines

    days = regular_session_frame(date(2026, 4, 1), date(2026, 8, 10))["trade_date"].to_list()
    config = _provider_config(tmp_path)
    config.quality_gate.max_quarantined_security_fraction = 0.5
    store = tmp_path / "store"
    parent = initialize_production_store(
        _revision(days, _bars(days, changed_last_factor=20)), config, store, history_sessions=40,
    )
    updated = publish_daily_source_revision(
        parent, _revision(days[-10:], _bars(days[-10:])), config, store, history_sessions=40,
    )
    # Reproduce the old format, which retained evidence only in the parent audit.
    for revision in (parent, updated):
        path = revision.root / "eodhd_ingestion_manifest.json"
        manifest = json.loads(path.read_text())
        manifest.pop("quarantines")
        path.write_text(json.dumps(manifest))
    current = load_current_revision(store)
    recovered = _source_quarantines(current)
    assert recovered["eodhd:isin:US0000000001"]["reasons"] == ["extreme_adjusted_return"]
    parent_bars = parent.root / "bars_daily.parquet"
    parent_bars.write_bytes(b"corrupt fixture")
    with pytest.raises(DataContractError, match="artifact integrity failure"):
        _source_quarantines(current)
    parent_bars.unlink()
    with pytest.raises(DataContractError, match="artifact integrity failure"):
        _source_quarantines(current)
    (parent.root / "eodhd_ingestion_manifest.json").unlink()
    with pytest.raises(DataContractError, match="incomplete revision"):
        _source_quarantines(current)


def test_source_commit_interruption_does_not_change_previous_current(tmp_path, monkeypatch):
    from facdigger.data import session_store

    days = regular_session_frame(date(2026, 4, 1), date(2026, 8, 10))["trade_date"].to_list()
    source = _bootstrap_source(tmp_path, days)
    store = tmp_path / "store"
    current = bootstrap_production_store(source, store, history_sessions=40)
    revision = _revision(days[-10:], _bars(days[-10:]))
    original = session_store._advance_current

    def interrupted(*args):
        raise SystemExit("stopped before CURRENT commit")

    monkeypatch.setattr(session_store, "_advance_current", interrupted)
    with pytest.raises(SystemExit):
        publish_daily_source_revision(
            current, revision, _provider_config(tmp_path), store, history_sessions=40,
        )
    assert load_current_revision(store).revision_id == current.revision_id
    monkeypatch.setattr(session_store, "_advance_current", original)
    updated = publish_daily_source_revision(
        load_current_revision(store), revision, _provider_config(tmp_path), store,
        history_sessions=40,
    )
    assert load_current_revision(store).revision_id == updated.revision_id


def test_missing_legacy_quality_audit_is_not_treated_as_empty_quarantine(tmp_path):
    from facdigger.data.session_store import _source_quarantines

    days = regular_session_frame(date(2026, 4, 1), date(2026, 8, 10))["trade_date"].to_list()
    config = _provider_config(tmp_path)
    current = initialize_production_store(
        _revision(days, _bars(days)), config, tmp_path / "store", history_sessions=40,
    )
    manifest = {key: value for key, value in current.manifest.items()
                if key not in {"quarantines", "quality"}}
    with pytest.raises(DataContractError, match="missing original production quarantine evidence"):
        _source_quarantines(replace(current, manifest=manifest))


def test_real_mgn_raw_revision_requires_backfill_even_when_adjusted_close_is_unchanged():
    from facdigger.data.session_store import _adjustment_changes

    sample = json.loads((Path(__file__).parents[2] / "fixtures/eodhd_adjustment_quality.json")
                        .read_text())
    example = sample["revision_example"]
    day = date.fromisoformat(example["date"])
    identity = f"eodhd:isin:{sample['isin']}"
    old = pl.DataFrame({"security_id": [identity], "trade_date": [day],
                        "adj_factor": [example["old_adjusted_close"] / example["old_close"]]})
    revised = old.with_columns(pl.lit(
        example["revised_adjusted_close"] / example["revised_close"]
    ).alias("adj_factor"))
    assert example["old_adjusted_close"] == example["revised_adjusted_close"]
    assert _adjustment_changes(old, revised, day) == [identity]
