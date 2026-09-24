"""Daily operational readiness, separate from immutable delivery integrity.

The pre-fetch reference is persisted by the runner. Missing observations cannot
shrink its denominator on retries; no function here creates bars or scores.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from facdigger.data.contracts import DataContractError
from facdigger.data.market_calendar import regular_sessions
from facdigger.data.paths import artifact_path
from facdigger.inference.delivery import (
    DeliveryConfig,
    require_resolved_delivery_identities,
    resolve_delivery,
)
from facdigger.production.config import ProductionInferenceConfig, ProductionQualityConfig


def build_quality_reference(
    universe: pl.DataFrame, *, target_date: date, context_length: int,
) -> dict[str, Any]:
    """Capture known membership before D, never today's already-reduced pool."""
    history = universe.filter(pl.col("trade_date") < target_date)
    reference_date = history["trade_date"].max()
    if reference_date is None:
        raise DataContractError("production quality reference requires membership before D")
    lengths = history.group_by("security_id").agg(pl.len().alias("sessions"))
    reference = history.filter(
        (pl.col("trade_date") == reference_date) & pl.col("eligible")
    ).join(lengths, on="security_id", validate="1:1")
    ids = sorted(reference.filter(pl.col("sessions") >= context_length)["security_id"])
    if not ids:
        raise DataContractError("production quality reference has no mature eligible securities")
    market_counts = history.group_by("trade_date").agg(
        pl.col("eligible").sum().alias("eligible_rows")
    ).sort("trade_date")
    return {
        "target_date": target_date.isoformat(),
        "reference_date": reference_date.isoformat(),
        "security_ids": ids,
        "market_eligible_rows": {
            day.isoformat(): int(count)
            for day, count in market_counts.iter_rows()
        },
    }


def assess_daily_quality(
    candidates: pl.DataFrame,
    *,
    reference: dict[str, Any],
    delivery: DeliveryConfig,
    inference: ProductionInferenceConfig,
    policy: ProductionQualityConfig,
    stage: str,
    unscorable: list[dict[str, Any]],
    market: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess the full computational pool, then the independent delivery scope."""
    target = date.fromisoformat(reference["target_date"])
    if candidates.is_empty() or candidates["asof_date"].unique().to_list() != [target]:
        raise DataContractError("production quality candidates must contain exactly target D")
    # Identity/row omissions are contract errors, not tolerable missing scores.
    selected = resolve_delivery(candidates, delivery)
    require_resolved_delivery_identities(selected, unscorable)
    unresolved = {row["security_id"] for row in unscorable
                  if row["reason"] == "unresolved_security_identity"}
    delivered_source_ids = set(selected.source_candidates["security_id"])
    expected = set(reference["security_ids"])
    current = set(candidates["security_id"])
    eligible = set(candidates.filter(pl.col("eligible"))["security_id"])
    # A valid stock leaving the liquidity pool is not itself missing data. But a
    # source/window failure among the reference stocks must not be hidden by replacements.
    data_reasons = {
        "missing_target_bar", "insufficient_liquidity_history", "insufficient_model_history",
        "source_quality_quarantined",
        "unresolved_security_identity",
    }
    unavailable = (expected - current) | {
        row["security_id"] for row in unscorable
        if row["reason"] in data_reasons and row["security_id"] in expected
    }
    count_shortfall = max(len(expected) - len(eligible), 0)
    missing_fraction = max(len(unavailable), count_shortfall) / len(expected)
    delivery_rows = selected.candidates.height
    delivery_eligible = int(selected.candidates["eligible"].sum())
    delivery_missing_fraction = (
        (delivery_rows - delivery_eligible) / delivery_rows if delivery_rows else 1.0
    )
    violations: list[str] = []
    if candidates.height < inference.minimum_candidate_rows:
        violations.append("computational_candidates_below_minimum")
    if len(eligible) < inference.minimum_eligible_rows:
        violations.append("computational_eligible_below_minimum")
    if missing_fraction > policy.max_computational_missing_fraction + 1e-12:
        violations.append("computational_missing_fraction_exceeded")
    if delivery_missing_fraction > policy.max_delivery_unscorable_fraction + 1e-12:
        violations.append("delivery_unscorable_fraction_exceeded")
    if delivery_eligible < policy.minimum_delivery_eligible_rows:
        violations.append("delivery_eligible_below_minimum")
    if market is not None:
        violations.extend(market["violations"])
    relevant = expected | delivered_source_ids | unresolved
    unavailable_details = [row for row in unscorable if row["security_id"] in relevant]
    degraded = bool(unavailable or missing_fraction or delivery_missing_fraction or unresolved)
    return {
        "target_date": target.isoformat(),
        "stage": stage,
        "status": "insufficient" if violations else ("degraded" if degraded else "ready"),
        "computation": {
            "reference_date": reference["reference_date"],
            "reference_eligible_rows": len(expected),
            "candidate_rows": candidates.height,
            "eligible_rows": len(eligible),
            "missing_reference_ids": sorted(unavailable),
            "missing_fraction": missing_fraction,
        },
        "delivery": {
            "candidate_rows": delivery_rows,
            "eligible_rows": delivery_eligible,
            "unscorable_fraction": delivery_missing_fraction,
            "unscorable_source_ids": sorted(
                selected.source_candidates.filter(~pl.col("eligible"))["security_id"]
            ),
        },
        "unscorable": unavailable_details,
        "market": market,
        "violations": violations,
    }


def assess_market_quality(
    snapshot_dir: Path,
    manifest: dict[str, Any],
    universe: pl.DataFrame,
    reference: dict[str, Any],
    policy: ProductionQualityConfig,
) -> dict[str, Any] | None:
    """Check the shared market input, not only individual scoring row counts."""
    contract = manifest["feature_contract"]
    if contract["feature_set"] != "finance_transformer":
        return None
    target = date.fromisoformat(reference["target_date"])
    context = int(contract["context_length"])
    days = regular_sessions(universe["trade_date"].min(), target)[-context:]
    paths = manifest["artifacts"]
    market = pl.read_parquet(artifact_path(snapshot_dir, paths["market_features"], "market"))
    market = market.filter(pl.col("trade_date").is_in(days))
    channels = [f"observed_{name}" for name in contract["market_channels"]]
    missing_dates = sorted(set(days) - set(market["trade_date"]))
    cells = market.height * len(channels)
    observed = int(market.select(pl.sum_horizontal(channels).sum()).item()) if cells else 0
    missing_fraction = 1 - observed / cells if cells else 1.0
    target_market = market.filter(pl.col("trade_date") == target)
    target_complete = target_market.height == 1 and all(
        target_market[column].item() for column in channels
    )
    features = pl.scan_parquet(artifact_path(snapshot_dir, paths["features"], "features"))
    contributors = (
        features.select("security_id", "trade_date", "observed_r_close")
        .filter(pl.col("trade_date").is_in(days))
        .join(universe.select("security_id", "trade_date", "eligible").lazy(),
              on=["security_id", "trade_date"], validate="1:1")
        .group_by("trade_date")
        .agg((pl.col("eligible") & pl.col("observed_r_close")).sum().alias("rows"))
        .collect()
    )
    counts = dict(contributors.iter_rows())
    expected_counts = reference["market_eligible_rows"]
    reference_count = expected_counts[reference["reference_date"]]
    deficient = []
    for day in days:
        expected = expected_counts.get(day.isoformat(), reference_count)
        actual = int(counts.get(day, 0))
        if expected and actual / expected < 1 - policy.max_computational_missing_fraction - 1e-12:
            deficient.append({"date": day.isoformat(), "expected": expected, "observed": actual})
    violations = []
    if len(days) < context or missing_dates:
        violations.append("market_context_dates_incomplete")
    if not target_complete:
        violations.append("target_market_channels_unavailable")
    if missing_fraction > policy.max_computational_missing_fraction + 1e-12:
        violations.append("market_context_observations_insufficient")
    if deficient:
        violations.append("market_contributor_coverage_insufficient")
    return {
        "missing_dates": [day.isoformat() for day in missing_dates],
        "missing_observation_fraction": missing_fraction,
        "target_channels_complete": target_complete,
        "deficient_contributor_dates": deficient,
        "violations": violations,
    }
