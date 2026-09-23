from __future__ import annotations

import json
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
    backfill_adjusted_histories,
    daily_revision_audit,
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


def test_one_invalid_stock_is_audited_without_fabricating_a_bar(tmp_path):
    class PartialClient(DailyClient):
        def get_json(self, path, params=None, *, call_cost=1):
            rows = super().get_json(path, params, call_cost=call_cost)
            if not rows:
                return rows
            if path.startswith("exchange-symbol-list/"):
                return [*rows, {**rows[0], "Code": "BBB", "Isin": "US0000000002"}]
            extra = {**rows[0], "code": "BBB"}
            if extra["date"] == "2026-08-12":
                extra["high"] = 1  # Illegal OHLC is rejected, never repaired.
            return [*rows, extra]

    revision = fetch_daily_revision(
        PartialClient(tmp_path), _config(tmp_path), revision_start=date(2026, 8, 10),
        target_date=date(2026, 8, 12),
    )
    assert revision.rejected_rows == 1
    assert dict(revision.rejected_rows_by_symbol) == {"BBB.US": 1}
    target = revision.bars.filter(revision.bars["trade_date"] == date(2026, 8, 12))
    assert target["symbol"].to_list() == ["AAA"]
    assert revision.bars.height == 5


class AliasClient(DailyClient):
    def __init__(self, tmp_path, *, alias_price):
        super().__init__(tmp_path)
        self.alias_price = alias_price

    def get_json(self, path, params=None, *, call_cost=1):
        if path.startswith("eod/"):
            return [{
                "date": str((params or {})["to"]), "open": 10.0, "high": 11.0, "low": 9.0,
                "close": 10.5, "adjusted_close": (
                    self.alias_price if path == "eod/AAOLD.US" else 10.5
                ), "volume": 1000.0,
            }]
        rows = super().get_json(path, params, call_cost=call_cost)
        if not rows:
            return rows
        if path.startswith("exchange-symbol-list/"):
            return [*rows, {**rows[0], "Code": "AAOLD"},
                    {**rows[0], "Code": "BBB", "Isin": "US0000000002"}]
        return [*rows, {**rows[0], "code": "AAOLD", "adjusted_close": self.alias_price},
                {**rows[0], "code": "BBB"}]


@pytest.mark.parametrize("alias_price,quarantined", [(10.6, 0), (21.0, 1)])
def test_raw_aliases_are_checked_before_daily_consolidation(tmp_path, alias_price, quarantined):
    config = _config(tmp_path)
    # The fixture has only two identities; production retains its unchanged 10% cap.
    config.quality_gate.max_quarantined_security_fraction = 0.5
    revision = fetch_daily_revision(
        AliasClient(tmp_path, alias_price=alias_price), config,
        revision_start=date(2026, 8, 10), target_date=date(2026, 8, 12),
    )
    audit = revision.raw_quality_audits[0]["identity"]
    # Production stores use strict JSON; audit dates must already be ISO strings.
    json.dumps(daily_revision_audit(revision))
    assert audit["quarantined_securities"] == quarantined
    assert audit["alias_overlap_conflict_groups"] == 3 * quarantined
    assert revision.bars.height == 3 * (2 - quarantined)
    if quarantined:
        assert revision.bars["symbol"].unique().to_list() == ["BBB"]
        assert audit["quarantined_security_ids"] == ["eodhd:isin:US0000000001"]


@pytest.mark.parametrize("alias_price,conflict_rows", [(10.6, 0), (21.0, 6)])
def test_alias_evidence_lists_only_materialize_conflicts(tmp_path, monkeypatch,
                                                       alias_price, conflict_rows):
    import polars as pl
    from polars.dataframe.group_by import GroupBy

    # Millions of clean security/date groups must not each allocate a list of
    # aliases just to discard it. Scalar gate checks still see every raw row.
    aggregate = GroupBy.agg
    list_rows = []

    def observe(self, *expressions, **named_expressions):
        if any(isinstance(expr, pl.Expr) and expr.meta.output_name() == "provider_symbols"
               for expr in expressions):
            list_rows.append(self.df.height)
        return aggregate(self, *expressions, **named_expressions)

    monkeypatch.setattr(GroupBy, "agg", observe)
    config = _config(tmp_path)
    config.quality_gate.max_quarantined_security_fraction = 0.5
    revision = fetch_daily_revision(
        AliasClient(tmp_path, alias_price=alias_price), config,
        revision_start=date(2026, 8, 10), target_date=date(2026, 8, 12),
    )
    assert sum(list_rows) == conflict_rows
    audit = revision.raw_quality_audits[0]["identity"]
    assert audit["alias_overlap_conflict_groups"] == (3 if conflict_rows else 0)
    for example in audit["alias_overlap_examples"]:
        assert example["provider_symbols"] == ["AAA.US", "AAOLD.US"]


def test_systemic_raw_alias_conflicts_still_fail_the_quality_gate(tmp_path):
    with pytest.raises(DataContractError, match="would quarantine"):
        fetch_daily_revision(
            AliasClient(tmp_path, alias_price=21.0), _config(tmp_path),
            revision_start=date(2026, 8, 10), target_date=date(2026, 8, 12),
        )


def test_adjustment_backfill_cannot_hide_conflicting_aliases(tmp_path):
    config = _config(tmp_path)
    config.quality_gate.max_quarantined_security_fraction = 0.5
    client = AliasClient(tmp_path, alias_price=10.6)
    revision = fetch_daily_revision(
        client, config, revision_start=date(2026, 8, 10), target_date=date(2026, 8, 12),
    )
    client.alias_price = 21.0
    result = backfill_adjusted_histories(
        client, config, revision, provider_symbols=["AAA.US", "BBB.US"],
        history_start=date(2026, 8, 10),
    )
    assert result.bars["symbol"].unique().to_list() == ["BBB"]
    assert result.backfilled_provider_symbols == ("BBB.US",)
    assert result.raw_quality_audits[-1]["identity"]["quarantined_security_ids"] == [
        "eodhd:isin:US0000000001",
    ]


def test_consistent_same_isin_aliases_remain_one_identity_after_backfill(tmp_path):
    client = AliasClient(tmp_path, alias_price=10.5)
    config = _config(tmp_path)
    revision = fetch_daily_revision(
        client, config, revision_start=date(2026, 8, 12), target_date=date(2026, 8, 12),
    )
    result = backfill_adjusted_histories(
        client, config, revision, provider_symbols=["AAA.US", "BBB.US"],
        history_start=date(2026, 8, 12),
    )
    assert result.backfilled_provider_symbols == ("AAA.US", "AAOLD.US", "BBB.US")
    assert result.bars.height == result.bars["security_id"].n_unique() == 2
    assert not result.raw_quality_audits[-1]["identity"]["quarantines"]


def test_bulk_targeted_disagreement_is_not_resolved_by_deduplication(tmp_path):
    import polars as pl

    config = _config(tmp_path)
    config.quality_gate.max_quarantined_security_fraction = 0.5
    client = AliasClient(tmp_path, alias_price=10.5)
    revision = fetch_daily_revision(
        client, config, revision_start=date(2026, 8, 12), target_date=date(2026, 8, 12),
    )
    revision = replace(revision, bars=revision.bars.with_columns(
        pl.when(pl.col("symbol") == "AAA").then(2000.0)
        .otherwise(pl.col("volume")).alias("volume"),
    ))
    result = backfill_adjusted_histories(
        client, config, revision, provider_symbols=["AAA.US", "BBB.US"],
        history_start=date(2026, 8, 12),
    )
    assert result.bars["symbol"].to_list() == ["BBB"]
    assert result.backfilled_provider_symbols == ("BBB.US",)
    assert result.raw_quality_audits[-1]["unavailable_histories"][0]["reasons"] == [
        "endpoint_history_conflict",
    ]


@pytest.mark.parametrize("empty_required", [False, True])
def test_empty_retired_alias_is_distinguished_from_missing_expected_history(
    tmp_path, empty_required,
):
    class EmptyAliasClient(AliasClient):
        def get_json(self, path, params=None, *, call_cost=1):
            if path == ("eod/AAA.US" if empty_required else "eod/AAOLD.US"):
                return []
            return super().get_json(path, params, call_cost=call_cost)

    config = _config(tmp_path)
    config.quality_gate.max_quarantined_security_fraction = 0.5
    client = EmptyAliasClient(tmp_path, alias_price=10.5)
    revision = fetch_daily_revision(
        client, config, revision_start=date(2026, 8, 12), target_date=date(2026, 8, 12),
    )
    revision = replace(revision, metadata_rows=tuple(
        {**row, "_is_delisted": True} if row["Code"] == "AAOLD" else row
        for row in revision.metadata_rows
    ))
    result = backfill_adjusted_histories(
        client, config, revision, provider_symbols=["AAA.US", "BBB.US"],
        history_start=date(2026, 8, 12),
    )
    audit = result.raw_quality_audits[-1]
    if empty_required:
        assert result.backfilled_provider_symbols == ("BBB.US",)
        assert audit["unavailable_histories"][0]["reasons"] == ["incomplete_adjustment_history"]
    else:
        assert result.backfilled_provider_symbols == ("AAA.US", "AAOLD.US", "BBB.US")
        assert audit["unavailable_histories"] == []
        assert audit["empty_related_aliases"] == ["AAOLD.US"]


def test_real_mgn_extreme_return_is_isolated_not_misreported_as_alias_conflict(tmp_path):
    from datetime import datetime, timezone

    import polars as pl

    from facdigger.data.providers.eodhd.daily import EODHDDailyRevision, _mapped_rows
    from facdigger.data.providers.eodhd.mapper import build_metadata_index

    sample = json.loads((Path(__file__).parents[3] / "fixtures/eodhd_adjustment_quality.json")
                        .read_text())
    metadata_rows = [{"Code": "MGN", "Isin": sample["isin"], "Exchange": "NASDAQ"}]
    metadata_rows += [{"Code": f"OK{i}", "Isin": f"US{i:010d}", "Exchange": "NASDAQ"}
                      for i in range(10)]
    metadata = build_metadata_index(metadata_rows, "US")
    now = datetime(2026, 9, 23, tzinfo=timezone.utc)
    payloads = {"MGN.US": sample["rows"]}
    for i in range(10):
        payloads[f"OK{i}.US"] = [
            {**row, "open": 10, "high": 11, "low": 9, "close": 10, "adjusted_close": 10}
            for row in sample["rows"]
        ]
    raw, _, _ = _mapped_rows(payloads, metadata, source_revision="test", ingested_at=now)
    revision = EODHDDailyRevision(
        date(2026, 3, 26), date(2026, 3, 26),
            raw.filter(pl.col("trade_date") == "2026-03-26").with_columns(
                pl.col("trade_date").str.to_date()
            ), tuple(metadata_rows),
        11, 0, (), "test", now,
    )

    class Client:
        request_log = []

        def get_json(self, path, params):
            return payloads[path.removeprefix("eod/")]

    result = backfill_adjusted_histories(
        Client(), _config(tmp_path), revision, provider_symbols=list(payloads),
        history_start=date(2026, 3, 25),
    )
    audit = result.raw_quality_audits[-1]["identity"]
    assert audit["alias_overlap_conflict_groups"] == 0
    assert audit["extreme_consecutive_return_rows"] == 1
    assert audit["quarantines"][0]["reasons"] == ["extreme_adjusted_return"]
    assert result.bars["security_id"].n_unique() == 10
    assert "MGN.US" not in result.backfilled_provider_symbols
