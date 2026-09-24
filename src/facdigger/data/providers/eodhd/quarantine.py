"""Small, durable EODHD quality exclusions, carried by existing source manifests."""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl

from facdigger.data.contracts import DataContractError
from facdigger.data.market_calendar import regular_sessions
from facdigger.data.providers.eodhd.identity import IDENTITY_CHANGE_PENDING, merge_identity_changes
from facdigger.data.providers.eodhd.mapper import (
    EXCHANGE_MAP,
    build_metadata_index,
    security_identity,
)


def merge_quarantines(
    previous: dict[str, dict[str, Any]], incoming: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Keep the original evidence while adding bounded, distinct later findings."""
    merged = dict(previous)
    for security_id, row in incoming.items():
        old = merged.get(security_id)
        if old is None:
            merged[security_id] = dict(row)
            continue
        examples = list(old.get("examples", []))
        examples += [item for item in row.get("examples", []) if item not in examples]
        starts = [value for value in (old.get("first_trade_date"), row.get("first_trade_date"))
                  if value]
        ends = [value for value in (old.get("last_trade_date"), row.get("last_trade_date"))
                if value]
        merged[security_id] = {
            "security_id": security_id,
            "reasons": sorted(set(old["reasons"]) | set(row["reasons"])),
            "provider_symbols": sorted(set(old.get("provider_symbols", []))
                                       | set(row.get("provider_symbols", []))),
            "first_trade_date": min(starts) if starts else None,
            "last_trade_date": max(ends) if ends else None,
            "examples": examples[:6],
        }
        if "identity_changes" in old or "identity_changes" in row:
            merged[security_id]["identity_changes"] = merge_identity_changes(
                old.get("identity_changes", []), row.get("identity_changes", []),
            )
    return merged


def audit_quarantines(audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Read current diagnostics or the retained ID-only evidence of an old audit."""
    records = {row["security_id"]: dict(row) for row in audit.get("quarantines", [])}
    for security_id in audit.get("quarantined_security_ids", []):
        records.setdefault(security_id, {
            "security_id": security_id, "reasons": ["historical_source_quality"],
            "provider_symbols": [], "first_trade_date": None, "last_trade_date": None,
            "examples": [],
        })
    return records


def revision_quarantines(audits: tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
    records = {}
    for audit in audits:
        records = merge_quarantines(records, audit_quarantines(audit.get("identity", {})))
        for record in audit.get("unavailable_histories", []):
            records = merge_quarantines(records, {record["security_id"]: record})
    return records


def manifest_quarantines(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "quarantines" in manifest:
        rows = manifest["quarantines"]
        if not isinstance(rows, list) or any(
            not isinstance(row, dict)
            or not isinstance(row.get("security_id"), str) or not row["security_id"]
            or not isinstance(row.get("reasons"), list) or not row["reasons"]
            or any(not isinstance(reason, str) or not reason for reason in row["reasons"])
            or not isinstance(row.get("provider_symbols", []), list)
            or any(not isinstance(symbol, str) for symbol in row.get("provider_symbols", []))
            or not isinstance(row.get("examples", []), list)
            for row in rows
        ):
            raise DataContractError("invalid persisted source quality quarantines")
        records = {row["security_id"]: dict(row) for row in rows}
        if len(records) != len(rows):
            raise DataContractError("duplicate persisted source quality quarantines")
        for record in records.values():
            if IDENTITY_CHANGE_PENDING in record["reasons"]:
                changes = merge_identity_changes(record.get("identity_changes"))
                if not changes or any(
                    row["provider_symbol"] not in record.get("provider_symbols", [])
                    for row in changes
                ):
                    raise DataContractError("identity quarantine lacks transition evidence")
            elif "identity_changes" in record:
                raise DataContractError("identity evidence is missing its quarantine reason")
        return records
    quality = manifest.get("quality") or {}
    records = audit_quarantines(quality)
    records = merge_quarantines(records, audit_quarantines(quality.get("identity", {})))
    for key in ("bootstrap", "daily_update"):
        audits = tuple((manifest.get(key) or {}).get("raw_quality", []))
        records = merge_quarantines(records, revision_quarantines(audits))
    return records


def retain_quarantined_candidates(
    universe: pl.DataFrame, previous: pl.DataFrame, known_bars: pl.DataFrame,
    records: dict[str, dict[str, Any]], *, days: list[date],
    metadata_rows: tuple[dict[str, Any], ...], exchange_code: str,
) -> pl.DataFrame:
    """Retain observed membership, never fabricate prices, scores or new listings."""
    if not records:
        return universe
    metadata = build_metadata_index(list(metadata_rows), exchange_code)
    by_id = {}
    for symbol, row in sorted(metadata.items()):
        security_id, _ = security_identity(symbol, row)
        if security_id not in by_id or not row["is_delisted"]:
            by_id[security_id] = row
    prior = previous.filter(pl.col("security_id").is_in(records.keys()))
    prior_groups = prior.partition_by("security_id", as_dict=True)
    observations = known_bars.filter(pl.col("security_id").is_in(records.keys()))
    bounds = {row["security_id"]: row for row in observations.group_by("security_id").agg(
        pl.col("trade_date").min().alias("first"),
        pl.col("trade_date").max().alias("last"),
        pl.col("symbol").first(), pl.col("provider_symbol").first(),
    ).to_dicts()}
    rows = []
    positions = {day: index for index, day in enumerate(days)}
    for security_id, evidence in sorted(records.items()):
        group = prior_groups.get((security_id,))
        bound = bounds.get(security_id)
        meta = by_id.get(security_id)
        first = evidence.get("first_trade_date")
        first = date.fromisoformat(first) if first else None
        if group is not None:
            first = group["trade_date"].min()
            anchor = group.sort("trade_date").row(-1, named=True)
            actual = {row["trade_date"]: row for row in group.to_dicts()}
        else:
            first = bound["first"] if bound else first
            last = bound["last"] if bound else evidence.get("last_trade_date")
            last = date.fromisoformat(last) if isinstance(last, str) else last
            if first is None or last is None or last < days[0] or meta is None:
                continue  # No observation in this context; metadata alone is not membership.
            symbol = bound["symbol"] if bound else meta["provider_symbol"].rsplit(".", 1)[0]
            anchor = {column: None for column in universe.columns}
            anchor.update({
                "security_id": security_id, "symbol": symbol, "trade_date": first,
                "listed_days": 1, "exchange": EXCHANGE_MAP.get(meta["exchange"], meta["exchange"]),
                "security_type": "common_stock", "is_primary_listing": True,
                "is_listed": not meta["is_delisted"], "is_delisted": meta["is_delisted"],
                "is_halted": False, "provider_symbol": meta["provider_symbol"],
                "identity_quality": "isin" if security_id.startswith("eodhd:isin:")
                else "provider_symbol_fallback",
            })
            actual = {}
        # Preserve prior dated metadata, extending only its known membership.
        anchor_position = positions.get(anchor["trade_date"])
        if anchor_position is None:
            anchor_position = 1 - len(regular_sessions(anchor["trade_date"], days[0]))
        for day in days:
            if day < first:
                continue
            row = dict(actual.get(day, anchor))
            if day not in actual:
                row["listed_days"] = (
                    int(anchor["listed_days"] or 0) + positions[day] - anchor_position
                )
            row.update({
                "trade_date": day, "eligible": False, "close": None, "adv20_usd": None,
                "trade_status_quality": (
                    "identity_change_quarantined"
                    if IDENTITY_CHANGE_PENDING in evidence["reasons"]
                    else "source_quality_quarantined"
                ), "liquidity_rank": None,
            })
            rows.append({column: row.get(column) for column in universe.columns})
    cleaned = universe.filter(~pl.col("security_id").is_in(records.keys()))
    if not rows:
        return cleaned
    return pl.concat([cleaned, pl.DataFrame(rows, schema=universe.schema)], how="vertical_relaxed")
