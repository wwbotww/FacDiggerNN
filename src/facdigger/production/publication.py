"""Reconcile atomic deliveries with the daily ledger before touching mutable data."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

from facdigger.data.contracts import DataContractError
from facdigger.data.inference_snapshots import load_inference_snapshot
from facdigger.data.market_calendar import CALENDAR_VERSION
from facdigger.data.snapshots import sha256_file
from facdigger.inference.delivery import delivery_identity_policy, resolve_delivery
from facdigger.inference.factor_batch import (
    FactorBatchInput,
    FactorBatchManifest,
    FactorBatchTime,
    factor_batch_metadata,
    factor_universe_sha256,
    load_factor_batch,
)
from facdigger.inference.releases import load_model_release
from facdigger.production.calendar import ProductionWindow
from facdigger.production.config import ProductionServiceConfig
from facdigger.production.quality import assess_daily_quality
from facdigger.production.state import ProductionRecord, ProductionState

logger = logging.getLogger(__name__)


def _unique_published_batch(root: Path, target: date) -> FactorBatchManifest | None:
    if not root.exists():
        return None
    matches = []
    for path in sorted(root.iterdir()):
        # Only this publisher-owned staging namespace is not a completed delivery.
        # Never promote staging output on recovery, especially after the cutoff.
        if path.name.startswith(".tmp-factor-batch-") and path.is_dir() and not path.is_symlink():
            continue
        try:
            if path.is_symlink():
                raise DataContractError("production delivery must not be a symlink")
            manifest = load_factor_batch(path)
        except Exception as exc:
            raise DataContractError(
                f"publication recovery cannot validate {path.name}: {exc}"
            ) from exc
        if manifest.time.minimum_asof_date <= target <= manifest.time.maximum_asof_date:
            matches.append(manifest)
    if len(matches) > 1:
        raise DataContractError(
            f"ambiguous published deliveries for {target}: "
            + ", ".join(manifest.delivery_id for manifest in matches)
        )
    return matches[0] if matches else None


def recover_publication(
    config: ProductionServiceConfig,
    state: ProductionState,
    window: ProductionWindow,
    pending: ProductionRecord | None,
    *,
    observed: datetime,
) -> ProductionRecord | None:
    """Restore only the original, verified, timely publication; never publish anew.

    The existing inference-quality record is the durable pre-publication binding
    to the immutable snapshot. A directory alone is not sufficient evidence.
    No CURRENT/source data, labels or model scoring are read on this path.
    """
    manifest = _unique_published_batch(config.factor_batch.output_root, window.target_date)
    if manifest is None:
        return None
    quality = pending.quality_report if pending else None
    if (
        pending is None or pending.release_id != config.model.release_id
        or pending.target_date != window.target_date
        or not quality or quality.get("stage") != "inference"
        or quality.get("target_date") != window.target_date.isoformat()
        or quality.get("status") not in {"ready", "degraded"}
        or quality.get("violations") != [] or pending.quality_reference is None
        or quality.get("snapshot_id") != manifest.input.snapshot_id
        or pending.snapshot_id not in {None, manifest.input.snapshot_id}
        or pending.delivery_id not in {None, manifest.delivery_id}
    ):
        raise DataContractError(
            "published delivery has no matching pending inference-quality record"
        )
    if (
        not window.first_attempt_at <= manifest.created_at < window.cutoff_at
        or manifest.created_at > observed
    ):
        raise DataContractError(
            "published delivery timestamp is outside the original publication window"
        )
    if config.factor_batch.delivery is None:
        raise DataContractError("publication recovery requires an explicit delivery profile")

    release = load_model_release(config.model.release_root / config.model.release_id)
    source_metadata, model_metadata = factor_batch_metadata(release, source_kind="signal_inference")
    if manifest.source != source_metadata or manifest.model != model_metadata:
        raise DataContractError("published delivery lineage differs from the fixed ModelRelease")
    if manifest.time != FactorBatchTime(
        calendar_version=CALENDAR_VERSION,
        minimum_asof_date=window.target_date,
        maximum_asof_date=window.target_date,
    ):
        raise DataContractError(
            "published delivery date/calendar differs from the production target"
        )

    snapshot_id = manifest.input.snapshot_id
    # Snapshot IDs are content addresses, never paths supplied by a manifest.
    if len(snapshot_id) != 64 or any(char not in "0123456789abcdef" for char in snapshot_id):
        raise DataContractError("published delivery has an invalid inference snapshot ID")
    snapshot_dir = config.inference.output_root / window.target_date.isoformat() / snapshot_id
    snapshot, frames = load_inference_snapshot(snapshot_dir, release)
    candidates = frames["delivery_universe"]
    selection = resolve_delivery(candidates, config.factor_batch.delivery)
    expected_input = FactorBatchInput(
        snapshot_id=snapshot["snapshot_id"],
        snapshot_manifest_sha256=sha256_file(snapshot_dir / "manifest.json"),
        universe_semantics="complete_candidate_cross_section",
        universe_sha256=factor_universe_sha256(selection.candidates),
        identity_policy=delivery_identity_policy(selection.candidates),
    )
    if manifest.input != expected_input:
        raise DataContractError(
            "published delivery differs from the original snapshot or delivery targets"
        )
    # Reuse the full-pool and independent delivery gates. In particular, recovery
    # cannot substitute a smaller denominator or silently omit unscorable rows.
    checked_quality = assess_daily_quality(
        candidates, reference=pending.quality_reference, delivery=config.factor_batch.delivery,
        inference=config.inference, policy=config.quality, stage="inference",
        unscorable=quality["unscorable"], market=quality["market"],
    )
    checked_quality["snapshot_id"] = snapshot_id
    if checked_quality != quality:
        raise DataContractError("published delivery no longer matches the pending quality gates")
    record = state.put(
        window.target_date, pending.release_id, "published", attempts=pending.attempts,
        snapshot_id=snapshot_id, delivery_id=manifest.delivery_id,
    )
    logger.info(json.dumps({
        "event": "production_publication_recovered", "target_date": window.target_date.isoformat(),
        "delivery_id": manifest.delivery_id, "created_at": manifest.created_at.isoformat(),
        "recovered_at": observed.isoformat(), "cutoff_at": window.cutoff_at.isoformat(),
    }, sort_keys=True))
    return record
