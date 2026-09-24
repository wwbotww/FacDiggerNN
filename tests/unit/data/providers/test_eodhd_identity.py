from datetime import date, datetime, timezone

import polars as pl
import pytest

from facdigger.data.contracts import DataContractError
from facdigger.data.providers.eodhd.identity import isolate_identity_changes
from facdigger.data.providers.eodhd.mapper import build_metadata_index
from facdigger.data.providers.eodhd.quarantine import manifest_quarantines

DAY = date(2026, 9, 23)


def _check(prior, fresh, persisted=None):
    previous = pl.DataFrame([
        {"security_id": f"eodhd:isin:{identity}", "provider_symbol": symbol,
         "trade_date": DAY} for symbol, identity in prior
    ])
    metadata = build_metadata_index([
        {"Code": symbol, "Isin": identity} for symbol, identity in fresh
    ], "US")
    return isolate_identity_changes(previous, metadata, persisted or {}, target_date=DAY,
                                    observed_at=datetime(2026, 9, 23, 23, tzinfo=timezone.utc))


def test_same_identity_ticker_rename_is_not_a_transition():
    result = _check([("OLD.US", "A")], [("NEW.US", "A")])
    assert not result.quarantines and not result.excluded_ids


@pytest.mark.parametrize("new", ["B", None])
def test_new_or_missing_isin_is_not_a_new_candidate(new):
    result = _check([("AAA.US", "A")], [("AAA.US", new)])
    assert list(result.quarantines) == ["eodhd:isin:A"]
    observed = "eodhd:isin:B" if new else "eodhd:symbol:AAA.US"
    assert result.unaccepted_ids == {observed}
    assert result.excluded_ids == {"eodhd:isin:A", observed}


def test_shared_alias_and_collision_expand_exclusions_not_identity_merges():
    result = _check([("AAA.US", "A"), ("OLD.US", "A"), ("BBB.US", "B"), ("CCC.US", "C")],
                    [("AAA.US", "B"), ("BBB.US", "B"), ("CCC.US", "C")])
    assert result.excluded_ids == {"eodhd:isin:A", "eodhd:isin:B"}
    assert not result.unaccepted_ids
    assert len(result.quarantines) == 2
    assert result.quarantines["eodhd:isin:B"]["provider_symbols"] == ["AAA.US", "BBB.US", "OLD.US"]


def test_multiple_changes_and_rename_cannot_escape_persisted_isolation():
    first = _check([("AAA.US", "A")], [("AAA.US", "B")])
    second = _check([("AAA.US", "A")], [("AAA.US", "C")], first.quarantines)
    third = _check([("AAA.US", "A")], [("RENAMED.US", "C")], second.quarantines)
    assert third.excluded_ids == {"eodhd:isin:A", "eodhd:isin:B", "eodhd:isin:C"}
    assert len(third.quarantines["eodhd:isin:A"]["identity_changes"]) == 2
    assert "RENAMED.US" in third.quarantines["eodhd:isin:A"]["provider_symbols"]


@pytest.mark.parametrize("damage", ["missing", "timestamp", "symbol", "reason"])
def test_damaged_identity_evidence_is_not_silently_forgotten(damage):
    result = _check([("AAA.US", "A")], [("AAA.US", "B")])
    record = result.quarantines["eodhd:isin:A"]
    if damage == "missing":
        del record["identity_changes"]
    elif damage == "timestamp":
        record["identity_changes"][0]["observed_at"] = "not a date"
    elif damage == "symbol":
        record["provider_symbols"] = []
    else:
        record["reasons"] = ["historical_source_quality"]
    with pytest.raises(DataContractError, match="identity"):
        manifest_quarantines({"quarantines": [record]})
