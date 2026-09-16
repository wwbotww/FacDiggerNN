"""One idempotent, fail-closed daily production transaction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import polars as pl

from facdigger.data.config import InferenceSnapshotConfig, ParquetSourceConfig
from facdigger.data.contracts import DataContractError
from facdigger.data.inference_snapshots import (
    build_inference_snapshot,
    describe_unscorable,
    load_inference_snapshot,
)
from facdigger.data.market_calendar import (
    next_regular_session,
    shift_regular_session,
)
from facdigger.data.providers.eodhd.client import EODHDError
from facdigger.data.providers.eodhd.config import load_eodhd_config
from facdigger.data.providers.eodhd.daily import (
    DailyDataNotReady,
    backfill_adjusted_histories,
    fetch_daily_revision,
    require_fresh_daily_requests,
)
from facdigger.data.providers.eodhd.provider import EODHDProvider
from facdigger.data.session_store import (
    AdjustmentBackfillRequired,
    ProductionSourceRevision,
    TargetSessionIncomplete,
    bootstrap_production_store,
    load_current_revision,
    publish_daily_source_revision,
)
from facdigger.inference.releases import ModelReleaseManifest, load_model_release
from facdigger.inference.runner import run_signal_inference
from facdigger.production.calendar import NEW_YORK, ProductionWindow, production_window
from facdigger.production.config import ProductionServiceConfig
from facdigger.production.quality import (
    assess_daily_quality,
    assess_market_quality,
    build_quality_reference,
)
from facdigger.production.state import ProductionRecord, ProductionState

TickAction = Literal[
    "not_due",
    "retry_wait",
    "already_published",
    "waiting_data",
    "published",
    "expired",
    "blocked",
]


@dataclass(frozen=True)
class ProductionTickResult:
    action: TickAction
    target_date: date
    phase: str
    attempts: int
    next_retry_at: datetime | None = None
    snapshot_id: str | None = None
    delivery_id: str | None = None
    error: str | None = None
    quality: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target_date": self.target_date.isoformat(),
            "phase": self.phase,
            "attempts": self.attempts,
            "next_retry_at": (
                self.next_retry_at.isoformat() if self.next_retry_at else None
            ),
            "snapshot_id": self.snapshot_id,
            "delivery_id": self.delivery_id,
            "error": self.error,
            "quality": self.quality,
        }


def _release_dir(config: ProductionServiceConfig) -> Path:
    return (config.model.release_root / config.model.release_id).resolve()


def _load_fixed_release(config: ProductionServiceConfig) -> tuple[Path, ModelReleaseManifest]:
    release_dir = _release_dir(config)
    release = load_model_release(release_dir)
    if release.release_id != config.model.release_id:
        raise DataContractError("loaded ModelRelease differs from fixed production release_id")
    return release_dir, release


def _history_sessions(
    config: ProductionServiceConfig,
    release: ModelReleaseManifest,
) -> int:
    return release.feature_contract.context_length + config.data.feature_buffer_sessions


def bootstrap_store(
    config: ProductionServiceConfig,
) -> ProductionSourceRevision:
    _, release = _load_fixed_release(config)
    try:
        current = load_current_revision(config.data.store_root)
    except FileNotFoundError:
        current = bootstrap_production_store(
            config.data.bootstrap_source,
            config.data.store_root,
            history_sessions=_history_sessions(config, release),
        )
    return current


def _revision_start(
    current_end: date,
    target: date,
    revision_sessions: int,
) -> date:
    normal_start = shift_regular_session(target, -(revision_sessions - 1))
    first_missing = next_regular_session(current_end)
    if first_missing < normal_start:
        return shift_regular_session(current_end, -(revision_sessions - 1))
    return normal_start


def _source_config(
    config: ProductionServiceConfig,
    current: ProductionSourceRevision,
) -> InferenceSnapshotConfig:
    return InferenceSnapshotConfig(
        dataset_name="eodhd_daily_production_inference",
        sources=ParquetSourceConfig(
            bars=current.root / "bars_daily.parquet",
            universe=current.root / "universe_daily.parquet",
            source_manifest=current.root / "eodhd_ingestion_manifest.json",
        ),
        output_root=config.inference.output_root,
    )


def _wait_record(
    state: ProductionState,
    config: ProductionServiceConfig,
    window: ProductionWindow,
    attempts: int,
    error: Exception,
    now: datetime,
) -> ProductionTickResult:
    if now.tzinfo is None:
        raise ValueError("production retry clock must be timezone-aware")
    now = now.astimezone(NEW_YORK)
    next_retry = min(
        now + timedelta(minutes=config.schedule.retry_minutes),
        window.cutoff_at,
    )
    record = state.put(
        window.target_date,
        config.model.release_id,
        "waiting_data",
        attempts=attempts,
        next_retry_at=next_retry,
        error=f"{type(error).__name__}: {error}",
    )
    return ProductionTickResult(
        "waiting_data",
        window.target_date,
        window.phase,
        attempts,
        next_retry_at=next_retry,
        error=str(error),
        quality=record.quality_report,
    )


def _expired_record(
    state: ProductionState,
    config: ProductionServiceConfig,
    window: ProductionWindow,
    existing: ProductionRecord | None,
) -> ProductionTickResult:
    attempts = existing.attempts if existing else 0
    error = "publication cutoff reached without a complete target FactorBatch"
    state.put(
        window.target_date,
        config.model.release_id,
        "expired",
        attempts=attempts,
        error=error,
    )
    return ProductionTickResult(
        "expired",
        window.target_date,
        window.phase,
        attempts,
        error=error,
        quality=existing.quality_report if existing else None,
    )


def _publish_guard(cutoff: datetime, now_provider: Any) -> None:
    observed = now_provider()
    if observed.tzinfo is None:
        raise ValueError("production clock must be timezone-aware")
    if observed.astimezone(NEW_YORK) >= cutoff:
        raise DailyDataNotReady("publication cutoff reached before atomic FactorBatch commit")


def _clock_value(clock: Any) -> datetime:
    observed = clock()
    if observed.tzinfo is None:
        raise ValueError("production clock must be timezone-aware")
    return observed


def run_production_tick(
    config: ProductionServiceConfig,
    *,
    now: datetime | None = None,
    now_provider: Any | None = None,
) -> ProductionTickResult:
    """Run at most one target transaction; never select a stale factor date."""

    clock = now_provider or (lambda: datetime.now(timezone.utc))
    observed = now or _clock_value(clock)
    if observed.tzinfo is None:
        raise ValueError("production clock must be timezone-aware")
    window = production_window(observed, config.schedule)
    with ProductionState(config.state_database) as state:
        existing = state.get(window.target_date)
        latest = state.latest()
        if (
            window.phase == "not_due"
            and latest is not None
            and latest.target_date < window.target_date
            and latest.status not in {"published", "expired", "blocked"}
        ):
            state.put(
                latest.target_date,
                latest.release_id,
                "expired",
                attempts=latest.attempts,
                error="publication cutoff reached without a complete target FactorBatch",
            )
            return ProductionTickResult(
                "expired",
                latest.target_date,
                "expired",
                latest.attempts,
                error="publication cutoff reached without a complete target FactorBatch",
                quality=latest.quality_report,
            )
        if existing is not None and existing.release_id != config.model.release_id:
            raise DataContractError(
                "fixed production release_id differs from the target's persisted release"
            )
        if existing is not None and existing.status == "published":
            return ProductionTickResult(
                "already_published",
                window.target_date,
                window.phase,
                existing.attempts,
                snapshot_id=existing.snapshot_id,
                delivery_id=existing.delivery_id,
                quality=existing.quality_report,
            )
        if window.phase == "not_due":
            return ProductionTickResult("not_due", window.target_date, window.phase, 0)
        if window.phase == "expired":
            return _expired_record(state, config, window, existing)
        if (
            existing is not None
            and existing.status == "waiting_data"
            and existing.next_retry_at is not None
            and observed < existing.next_retry_at
        ):
            return ProductionTickResult(
                "retry_wait",
                window.target_date,
                window.phase,
                existing.attempts,
                next_retry_at=existing.next_retry_at,
                error=existing.error,
                quality=existing.quality_report,
            )
        if existing is not None and existing.status == "blocked":
            return ProductionTickResult(
                "blocked",
                window.target_date,
                window.phase,
                existing.attempts,
                error=existing.error,
                quality=existing.quality_report,
            )

        attempts = (existing.attempts if existing else 0) + 1
        state.put(
            window.target_date,
            config.model.release_id,
            "running",
            attempts=attempts,
        )
        try:
            release_dir, release = _load_fixed_release(config)
            if config.factor_batch.delivery is None:
                raise DataContractError("daily production requires an explicit delivery profile")
            history_sessions = _history_sessions(config, release)
            try:
                current = load_current_revision(config.data.store_root)
            except FileNotFoundError:
                current = bootstrap_production_store(
                    config.data.bootstrap_source,
                    config.data.store_root,
                    history_sessions=history_sessions,
                )
            current_end = date.fromisoformat(str(current.manifest["resolved_end"]))
            if current_end > window.target_date:
                raise DataContractError("production source is newer than requested target")
            reference = existing.quality_reference if existing else None
            if reference is None:
                previous_quality = (latest.quality_report or {}) if latest else {}
                untrusted_reference = (
                    previous_quality.get("stage") != "inference"
                    or previous_quality.get("status") == "insufficient"
                    or (previous_quality.get("computation") or {}).get("missing_fraction", 0) > 0
                )
                if (
                    latest is not None and latest.target_date < window.target_date
                    and latest.quality_reference is not None and untrusted_reference
                ):
                    # A failed/reduced day or interrupted source commit cannot
                    # become tomorrow's smaller baseline without an inference check.
                    reference = {
                        **latest.quality_reference, "target_date": window.target_date.isoformat(),
                    }
                else:
                    source = _source_config(config, current)
                    reference = build_quality_reference(
                        pl.read_parquet(source.sources.universe), target_date=window.target_date,
                        context_length=release.feature_contract.context_length,
                    )
                state.put(
                    window.target_date, config.model.release_id, "running", attempts=attempts,
                    quality_reference=reference,
                )
            # Source commit is NOT publication. Every due, unpublished attempt
            # fetches fresh revisions, including a retry whose CURRENT already ends at D.
            provider_config = load_eodhd_config(config.data.provider_config)
            if not provider_config.refresh or provider_config.cache_ttl_hours != 0:
                raise DataContractError(
                    "daily EODHD config must use refresh=true and cache_ttl_hours=0"
                )
            provider = EODHDProvider(provider_config)
            client = provider.client()
            revision = fetch_daily_revision(
                client,
                provider_config,
                revision_start=_revision_start(
                    current_end,
                    window.target_date,
                    config.data.revision_sessions,
                ),
                target_date=window.target_date,
            )
            require_fresh_daily_requests(revision)
            try:
                current = publish_daily_source_revision(
                    current,
                    revision,
                    provider_config,
                    config.data.store_root,
                    history_sessions=history_sessions,
                )
            except AdjustmentBackfillRequired as required:
                revision = backfill_adjusted_histories(
                    client,
                    provider_config,
                    revision,
                    provider_symbols=required.provider_symbols,
                    history_start=required.history_start,
                )
                require_fresh_daily_requests(revision)
                current = publish_daily_source_revision(
                    current,
                    revision,
                    provider_config,
                    config.data.store_root,
                    history_sessions=history_sessions,
                )
            source = _source_config(config, current)
            universe = pl.read_parquet(source.sources.universe)
            target_universe = universe.filter(pl.col("trade_date") == window.target_date)
            target_bars = (
                pl.scan_parquet(source.sources.bars)
                .filter(pl.col("trade_date") == window.target_date)
                .select("security_id", "trade_date").collect()
            )
            candidates = target_universe.select(
                "security_id", "symbol", pl.col("trade_date").alias("asof_date"), "eligible",
            )
            quality = assess_daily_quality(
                candidates, reference=reference, delivery=config.factor_batch.delivery,
                inference=config.inference, policy=config.quality, stage="source",
                unscorable=describe_unscorable(target_universe, target_bars, candidates),
            )
            state.put(
                window.target_date, config.model.release_id, "running", attempts=attempts,
                quality_report=quality,
            )
            if quality["violations"]:
                raise TargetSessionIncomplete(
                    "source readiness: " + ", ".join(quality["violations"])
                )
            snapshot_dir, snapshot_manifest = build_inference_snapshot(
                source,
                release_dir,
                asof_date=window.target_date,
            )
            _, snapshot_frames = load_inference_snapshot(snapshot_dir, release)
            computational_universe = snapshot_frames["delivery_universe"]
            quality = assess_daily_quality(
                computational_universe, reference=reference, delivery=config.factor_batch.delivery,
                inference=config.inference, policy=config.quality, stage="inference",
                unscorable=describe_unscorable(
                    target_universe, target_bars, computational_universe,
                ),
                market=assess_market_quality(
                    snapshot_dir, snapshot_manifest, universe, reference, config.quality,
                ),
            )
            quality["snapshot_id"] = snapshot_manifest["snapshot_id"]
            state.put(
                window.target_date, config.model.release_id, "running", attempts=attempts,
                quality_report=quality,
            )
            if quality["violations"]:
                raise TargetSessionIncomplete(
                    "inference readiness: " + ", ".join(quality["violations"])
                )
            destination, factor_manifest = run_signal_inference(
                release_dir,
                output_root=config.factor_batch.output_root,
                dataset_dir=snapshot_dir,
                asof=window.target_date.isoformat(),
                device=config.model.device,
                before_publish=lambda: _publish_guard(window.cutoff_at, clock),
                delivery=config.factor_batch.delivery,
            )
            del destination
            if factor_manifest["time"]["maximum_asof_date"] != (
                window.target_date.isoformat()
            ):
                raise DataContractError("published FactorBatch date differs from target")
            record = state.put(
                window.target_date,
                config.model.release_id,
                "published",
                attempts=attempts,
                snapshot_id=str(snapshot_manifest["snapshot_id"]),
                delivery_id=str(factor_manifest["delivery_id"]),
            )
            return ProductionTickResult(
                "published",
                window.target_date,
                window.phase,
                attempts,
                snapshot_id=record.snapshot_id,
                delivery_id=record.delivery_id,
                quality=record.quality_report,
            )
        except (
            DailyDataNotReady,
            EODHDError,
            TargetSessionIncomplete,
        ) as exc:
            retry_observed = _clock_value(clock)
            if retry_observed.astimezone(NEW_YORK) >= window.cutoff_at:
                return _expired_record(state, config, window, state.get(window.target_date))
            return _wait_record(
                state,
                config,
                window,
                attempts,
                exc,
                retry_observed,
            )
        except Exception as exc:
            record = state.put(
                window.target_date,
                config.model.release_id,
                "blocked",
                attempts=attempts,
                error=f"{type(exc).__name__}: {exc}",
            )
            return ProductionTickResult(
                "blocked",
                window.target_date,
                window.phase,
                attempts,
                error=f"{type(exc).__name__}: {exc}",
                quality=record.quality_report,
            )


def production_status(config: ProductionServiceConfig) -> dict[str, Any]:
    with ProductionState(config.state_database) as state:
        latest = state.latest()
        health = state.service_health()
    return {
        "fixed_release_id": config.model.release_id,
        "latest": (
            None
            if latest is None
            else {
                "target_date": latest.target_date.isoformat(),
                "release_id": latest.release_id,
                "status": latest.status,
                "attempts": latest.attempts,
                "next_retry_at": (
                    latest.next_retry_at.isoformat() if latest.next_retry_at else None
                ),
                "snapshot_id": latest.snapshot_id,
                "delivery_id": latest.delivery_id,
                "error": latest.error,
                "quality": latest.quality_report,
            }
        ),
        "service": health,
    }


def status_json(config: ProductionServiceConfig) -> str:
    return json.dumps(production_status(config), ensure_ascii=False, indent=2, sort_keys=True)
