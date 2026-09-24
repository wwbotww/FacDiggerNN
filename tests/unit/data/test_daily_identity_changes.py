"""Identity changes never relabel history or silently shrink daily membership."""

from dataclasses import replace
from datetime import date

import polars as pl
import pytest
from test_session_store import _bars, _provider_config, _revision

from facdigger.data.contracts import DataContractError
from facdigger.data.inference_snapshots import describe_unscorable
from facdigger.data.market_calendar import regular_sessions
from facdigger.data.session_store import (
    initialize_production_store,
    load_current_revision,
    publish_daily_source_revision,
)
from facdigger.data.snapshots import sha256_file

SAMPLES = [
    ("MLGO", "KYG6077Y3015", "KYG6077Y4005"),
    ("RNAZ", "US89357L1052", "US89357L5012"),
    ("TNON", "US88066N3035", "US88066N4025"),
]


def _sample(days, *, changed=False):
    parts = [_bars(days).filter(pl.col("symbol") == "BBB")]
    metadata = [_revision(days, _bars(days)).metadata_rows[1]]
    for symbol, old, new in SAMPLES:
        parts.append(_bars(days).filter(pl.col("symbol") == "AAA").with_columns(
            pl.lit(symbol).alias("symbol"), pl.lit(f"{symbol}.US").alias("provider_symbol"),
            pl.lit(f"eodhd:isin:{new if changed else old}").alias("security_id"),
        ))
        metadata.append({"Code": symbol, "Isin": new if changed else old,
                         "Exchange": "NASDAQ", "Type": "Common Stock"})
    return replace(_revision(days, pl.concat(parts)), metadata_rows=tuple(metadata))


def test_real_identity_pairs_are_isolated_across_context_and_repeated_runs(tmp_path):
    days = regular_sessions(date(2026, 5, 1), date(2026, 9, 23))
    config = _provider_config(tmp_path)
    config.quality_gate.max_quarantined_security_fraction = 0.75
    store = tmp_path / "store"
    original = initialize_production_store(_sample(days[:-1]), config, store, history_sessions=60)
    before = {p.name: sha256_file(p) for p in original.root.iterdir()}
    current = original
    for changed in (True, True, False):
        # Even a full, clean history or a provider rollback cannot clear identity isolation.
        revision = replace(_sample(days, changed=changed), revision_start=days[-10],
                           backfilled_provider_symbols=tuple(f"{s}.US" for s, _, _ in SAMPLES))
        current = publish_daily_source_revision(
            current, revision, config, store, history_sessions=60,
        )
        current = load_current_revision(store)  # Restart using persisted evidence only.
        bars = pl.read_parquet(current.root / "bars_daily.parquet")
        universe = pl.read_parquet(current.root / "universe_daily.parquet")
        assert bars["symbol"].unique().to_list() == ["BBB"]
        assert universe.height == 60 * 4
        assert set(universe["security_id"]) == {
            "eodhd:isin:US0000000002", *(f"eodhd:isin:{old}" for _, old, _ in SAMPLES),
        }
        isolated = universe.filter(pl.col("symbol") != "BBB")
        assert isolated["eligible"].sum() == 0
        assert isolated["close"].null_count() == isolated.height
        assert isolated["trade_status_quality"].unique().to_list() == [
            "identity_change_quarantined",
        ]
        assert len(current.manifest["quarantines"]) == 3
        for record in current.manifest["quarantines"]:
            assert "identity_change_pending" in record["reasons"]
            assert len(record["identity_changes"]) == 1
        target = universe.filter(pl.col("trade_date") == days[-1])
        candidates = target.select("security_id", "symbol", "eligible",
                                   pl.col("trade_date").alias("asof_date"))
        assert {r["reason"] for r in describe_unscorable(target, bars, candidates)} == {
            "unresolved_security_identity",
        }
    assert before == {p.name: sha256_file(p) for p in original.root.iterdir()}


def test_new_identity_cannot_dilute_source_quarantine_denominator(tmp_path):
    days = regular_sessions(date(2026, 5, 1), date(2026, 9, 23))
    config = _provider_config(tmp_path)
    config.quality_gate.max_quarantined_security_fraction = 0.5
    store = tmp_path / "store"
    original = initialize_production_store(_sample(days[:-1]), config, store, history_sessions=60)
    with pytest.raises(DataContractError, match="quarantine exceeds configured limit: 3/4"):
        publish_daily_source_revision(original, _sample(days[-10:], changed=True), config,
                                      store, history_sessions=60)
    assert load_current_revision(store).revision_id == original.revision_id
