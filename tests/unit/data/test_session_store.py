from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from facdigger.data.contracts import DataContractError, table_audit, validate_bars
from facdigger.data.provenance import build_standardization_contract
from facdigger.data.providers.eodhd.config import EODHDConfig
from facdigger.data.providers.eodhd.daily import EODHDDailyRevision
from facdigger.data.providers.eodhd.mapper import build_universe
from facdigger.data.providers.eodhd.market_calendar import regular_session_frame
from facdigger.data.session_store import (
    AdjustmentBackfillRequired,
    bootstrap_production_store,
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
        minimum_candidate_rows=2,
        minimum_eligible_rows=2,
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
            minimum_candidate_rows=2,
            minimum_eligible_rows=2,
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
        minimum_candidate_rows=2,
        minimum_eligible_rows=2,
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
        history_sessions=40, minimum_candidate_rows=2, minimum_eligible_rows=2,
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
            history_sessions=20, minimum_candidate_rows=2, minimum_eligible_rows=2,
        )
    assert load_current_revision(store).revision_id == current.revision_id
