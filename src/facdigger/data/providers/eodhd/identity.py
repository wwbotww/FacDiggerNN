"""Conservative daily identity isolation, not ticker-based history succession."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import polars as pl

from facdigger.data.contracts import DataContractError
from facdigger.data.providers.eodhd.mapper import security_identity

IDENTITY_CHANGE_PENDING = "identity_change_pending"


def merge_identity_changes(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain the first observation of each transition, including subsequent reversals."""
    result = {}
    for group in groups:
        if not isinstance(group, list):
            raise DataContractError("invalid persisted identity change evidence")
        for row in group:
            try:
                keys = ("provider_symbol", "previous_security_id", "observed_security_id")
                if any(not isinstance(row[key], str) or not row[key] for key in keys):
                    raise ValueError("empty identity")
                if row["previous_security_id"] == row["observed_security_id"]:
                    raise ValueError("unchanged identity is not a transition")
                if datetime.fromisoformat(row["observed_at"]).tzinfo is None:
                    raise ValueError("observation time is naive")
                date.fromisoformat(row["target_date"])
                result.setdefault(tuple(row[key] for key in keys), dict(row))
            except (KeyError, TypeError, ValueError) as exc:
                raise DataContractError("invalid persisted identity change evidence") from exc
    return [result[key] for key in sorted(result)]


@dataclass(frozen=True)
class IdentityIsolation:
    quarantines: dict[str, dict[str, Any]]
    excluded_ids: frozenset[str]
    unaccepted_ids: frozenset[str]


def isolate_identity_changes(
    previous: pl.DataFrame,
    metadata: dict[str, dict[str, Any]],
    persisted: dict[str, dict[str, Any]],
    *,
    target_date: date,
    observed_at: datetime,
) -> IdentityIsolation:
    """Keep original members; isolate connected aliases without accepting new IDs.

    Linking identities here only expands exclusions. It never declares economic
    continuity, merges prices, or authorizes cross-system identity mappings.
    """
    graph: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)

    def connect(symbol: str, identity: str) -> None:
        a, b = ("symbol", symbol), ("id", identity)
        graph[a].add(b)
        graph[b].add(a)

    prior = previous.select("security_id", "provider_symbol").unique()
    prior_ids = set(prior["security_id"])
    observations = []
    seeds = set()
    for identity, symbol in prior.iter_rows():
        connect(symbol, identity)
        if symbol in metadata:
            incoming, _ = security_identity(symbol, metadata[symbol])
            if incoming != identity:
                seeds.add(("id", identity))
                observations.append({
                    "provider_symbol": symbol, "previous_security_id": identity,
                    "observed_security_id": incoming, "target_date": str(target_date),
                    "observed_at": observed_at.isoformat(),
                })
    for identity, record in persisted.items():
        if IDENTITY_CHANGE_PENDING not in record["reasons"]:
            continue
        changes = merge_identity_changes(record.get("identity_changes"))
        if not changes or identity not in prior_ids:
            raise DataContractError("identity quarantine lacks evidence or original membership")
        seeds.add(("id", identity))
        for symbol in record["provider_symbols"]:
            connect(symbol, identity)
        for row in changes:
            connect(row["provider_symbol"], row["previous_security_id"])
            connect(row["provider_symbol"], row["observed_security_id"])
        observations = merge_identity_changes(changes, observations)
    for symbol, row in metadata.items():
        connect(symbol, security_identity(symbol, row)[0])
    observations = merge_identity_changes(observations)
    bounds = {row["security_id"]: row for row in previous.group_by("security_id").agg(
        pl.col("trade_date").min().alias("first"),
        pl.col("trade_date").max().alias("last"),
    ).to_dicts()}
    excluded: set[str] = set()
    records = {}
    visited = set()
    for seed in sorted(seeds):
        if seed in visited:
            continue
        component, pending = set(), [seed]
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            pending.extend(graph[node] - component)
        visited.update(component)
        ids = {value for kind, value in component if kind == "id"}
        symbols = sorted(value for kind, value in component if kind == "symbol")
        changes = [row for row in observations if row["provider_symbol"] in symbols]
        excluded.update(ids)
        for identity in sorted(ids & prior_ids):
            records[identity] = {
                "security_id": identity, "provider_symbols": symbols,
                "reasons": [IDENTITY_CHANGE_PENDING], "examples": [],
                "first_trade_date": str(bounds[identity]["first"]),
                "last_trade_date": str(bounds[identity]["last"]),
                "identity_changes": changes,
            }
    return IdentityIsolation(records, frozenset(excluded), frozenset(excluded - prior_ids))
