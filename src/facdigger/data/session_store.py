"""Atomic provider-neutral storage for revised production sessions."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from facdigger.data.contracts import (
    DataContractError,
    table_audit,
    validate_bars,
    validate_universe,
)
from facdigger.data.market_calendar import (
    regular_session_frame,
    regular_sessions,
)
from facdigger.data.provenance import (
    build_standardization_contract,
    read_source_provenance_manifest,
    require_accepted_source,
)
from facdigger.data.providers.eodhd.config import EODHDConfig
from facdigger.data.providers.eodhd.daily import EODHDDailyRevision, daily_revision_audit
from facdigger.data.providers.eodhd.mapper import (
    build_metadata_index,
    build_universe,
    consolidate_bars,
    display_symbol,
    security_identity,
)
from facdigger.data.providers.eodhd.quality import (
    assert_historical_ingestion_quality,
    quarantine_suspicious_identities,
)
from facdigger.data.providers.eodhd.quarantine import (
    audit_quarantines,
    manifest_quarantines,
    merge_quarantines,
    retain_quarantined_candidates,
    revision_quarantines,
)
from facdigger.data.snapshots import sha256_file

PRODUCTION_SOURCE_FILES = {
    "bars": "bars_daily.parquet",
    "universe": "universe_daily.parquet",
}
PRODUCTION_SOURCE_CONTRACT = "facdigger.production_source"


class AdjustmentBackfillRequired(DataContractError):
    def __init__(
        self,
        security_ids: list[str],
        provider_symbols: list[str],
        history_start: date,
    ) -> None:
        super().__init__(
            "EODHD adjustment factors changed across the normal revision boundary; "
            f"targeted backfill required for {len(security_ids)} securities"
        )
        self.security_ids = security_ids
        self.provider_symbols = provider_symbols
        self.history_start = history_start


class TargetSessionIncomplete(DataContractError):
    """The requested target cannot safely be published yet."""


@dataclass(frozen=True)
class ProductionSourceRevision:
    revision_id: str
    root: Path
    manifest: dict[str, Any]


def _current_path(store_root: Path) -> Path:
    return store_root / "CURRENT"


def _advance_current(store_root: Path, revision_id: str) -> None:
    store_root.mkdir(parents=True, exist_ok=True)
    pointer_tmp = store_root / f".CURRENT.{uuid.uuid4().hex}.tmp"
    pointer_tmp.write_text(revision_id + "\n", encoding="utf-8")
    pointer_tmp.replace(_current_path(store_root))


def load_current_revision(store_root: str | Path) -> ProductionSourceRevision:
    root = Path(store_root).resolve()
    pointer = _current_path(root)
    if not pointer.is_file():
        raise FileNotFoundError("production data store is not bootstrapped")
    revision_id = pointer.read_text(encoding="utf-8").strip()
    revision_root = root / "revisions" / revision_id
    return _load_revision(revision_root, revision_id)


def _load_revision(revision_root: Path, revision_id: str) -> ProductionSourceRevision:
    if len(revision_id) != 32 or any(char not in "0123456789abcdef" for char in revision_id):
        raise DataContractError("invalid production source revision ID")
    manifest_path = revision_root / "eodhd_ingestion_manifest.json"
    if not manifest_path.is_file():
        raise DataContractError("production CURRENT points to an incomplete revision")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("revision_id") != revision_id:
        raise DataContractError("production revision identity does not match CURRENT")
    standardization = manifest.get("standardization") or {}
    tables = standardization.get("tables") or {}
    if set(tables) != set(PRODUCTION_SOURCE_FILES):
        raise DataContractError("production revision must contain only bars and universe")
    for name, evidence in tables.items():
        path = revision_root / evidence["file"]
        if not path.is_file() or sha256_file(path) != evidence["sha256"]:
            raise DataContractError(f"production revision artifact integrity failure: {name}")
    universe_dates = (
        pl.scan_parquet(revision_root / PRODUCTION_SOURCE_FILES["universe"])
        .select("trade_date")
        .unique()
        .collect()
    )
    _require_universe_history(universe_dates, manifest)
    return ProductionSourceRevision(revision_id, revision_root, manifest)


def _source_quarantines(current: ProductionSourceRevision) -> dict[str, dict[str, Any]]:
    """Migrate retained old audit evidence once; future revisions carry it themselves."""
    records = manifest_quarantines(current.manifest)
    if "quarantines" in current.manifest:
        return records
    revision = current
    seen = {current.revision_id}
    while parent := (revision.manifest.get("production_revision") or {}).get("parent_revision_id"):
        if parent in seen:
            raise DataContractError("cycle in production quality evidence")
        seen.add(parent)
        revision = _load_revision(current.root.parent / parent, parent)
        records = merge_quarantines(manifest_quarantines(revision.manifest), records)
        if "quarantines" in revision.manifest:
            return records
    kind = (revision.manifest.get("production_revision") or {}).get("kind")
    quality = revision.manifest.get("quality") or {}
    audits = [quality, quality.get("identity", {})]
    for key in ("bootstrap", "daily_update"):
        audits.extend(
            item.get("identity", {})
            for item in (revision.manifest.get(key) or {}).get("raw_quality", [])
        )
    has_evidence = any("quarantined_security_ids" in audit or "quarantines" in audit
                       for audit in audits)
    if kind not in {"bootstrap", "live_initialization"} or not has_evidence:
        raise DataContractError("missing original production quarantine evidence; review source")
    return records


def _history_start(days: list[date], history_sessions: int) -> date:
    if history_sessions < 20:
        raise ValueError("history_sessions must preserve feature warm-up")
    if not days:
        raise DataContractError("production source contains no regular sessions")
    return days[max(0, len(days) - history_sessions)]


def _require_universe_history(universe: pl.DataFrame, manifest: dict[str, Any]) -> None:
    expected = regular_sessions(
        date.fromisoformat(manifest["resolved_start"]),
        date.fromisoformat(manifest["resolved_end"]),
    )
    if universe["trade_date"].unique().sort().to_list() != expected:
        raise DataContractError(
            "production source lacks complete historical universe membership; "
            "re-bootstrap from historical bronze into a new store_root"
        )


def _write_revision(
    store_root: Path,
    revision_id: str,
    frames: dict[str, pl.DataFrame],
    manifest: dict[str, Any],
) -> None:
    _require_universe_history(frames["universe"], manifest)
    destination = store_root / "revisions" / revision_id
    if destination.exists():
        return
    temporary = store_root / "revisions" / f".tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        evidence: dict[str, dict[str, Any]] = {}
        for name, frame in frames.items():
            filename = PRODUCTION_SOURCE_FILES[name]
            path = temporary / filename
            frame.write_parquet(path)
            evidence[name] = {
                **table_audit(frame, "trade_date"),
                "file": filename,
                "sha256": sha256_file(path),
            }
        payload = {
            **manifest,
            "revision_id": revision_id,
            "standardization": build_standardization_contract(
                evidence,
                research_ready=False,
            ),
        }
        (temporary / "eodhd_ingestion_manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def bootstrap_production_store(
    source_dir: str | Path,
    store_root: str | Path,
    *,
    history_sessions: int,
) -> ProductionSourceRevision:
    """Import a bounded hot window without modifying the historical bronze source."""

    source = Path(source_dir).resolve()
    manifest_path = source / "eodhd_ingestion_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("bootstrap source has no EODHD provenance manifest")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = read_source_provenance_manifest(manifest_path)
    require_accepted_source(provenance)
    tables = ((payload.get("standardization") or {}).get("tables") or {})
    if payload.get("provider") != "eodhd" or not {"bars", "universe"}.issubset(tables):
        raise DataContractError("bootstrap source is not accepted EODHD standard data")
    for name in ("bars", "universe"):
        evidence = tables[name]
        path = source / evidence["file"]
        if not path.is_file() or sha256_file(path) != evidence["sha256"]:
            raise DataContractError(f"bootstrap source artifact integrity failure: {name}")

    universe_path = source / tables["universe"]["file"]
    session_dates = (
        pl.scan_parquet(universe_path)
        .select("trade_date")
        .unique()
        .sort("trade_date")
        .collect()["trade_date"]
        .to_list()
    )
    hot_start = session_dates[max(0, len(session_dates) - history_sessions)]
    retained_dates = set(session_dates[-history_sessions:])
    bars = validate_bars(
        pl.scan_parquet(source / tables["bars"]["file"])
        .filter(pl.col("trade_date").is_in(retained_dates))
        .collect()
    )
    universe = validate_universe(
        pl.scan_parquet(universe_path)
        .filter(pl.col("trade_date").is_in(retained_dates))
        .collect()
    )
    revision_metadata = {
        "contract": PRODUCTION_SOURCE_CONTRACT,
        "kind": "bootstrap",
        "source_manifest_sha256": sha256_file(manifest_path),
        "history_sessions": history_sessions,
        "resolved_start": hot_start.isoformat(),
        "resolved_end": session_dates[-1].isoformat(),
    }
    revision_id = uuid.uuid4().hex
    root = Path(store_root).resolve()
    _write_revision(
        root,
        revision_id,
        {"bars": bars, "universe": universe},
        {
            "provider": "eodhd",
            "resolved_start": hot_start.isoformat(),
            "resolved_end": session_dates[-1].isoformat(),
            "source_revision": payload.get("source_revision"),
            "ingested_at": payload.get("ingested_at"),
            "production_revision": revision_metadata,
            "warnings": list(payload.get("warnings") or []),
            "selection": {"research_ready": False},
            "quarantines": list(manifest_quarantines(payload).values()),
            "bootstrap": {
                "source_manifest_sha256": sha256_file(manifest_path),
                "history_sessions": history_sessions,
            },
        },
    )
    _advance_current(root, revision_id)
    return load_current_revision(root)


def initialize_production_store(
    revision: EODHDDailyRevision,
    config: EODHDConfig,
    store_root: str | Path,
    *,
    history_sessions: int,
) -> ProductionSourceRevision:
    """Create a new current-identity store; never relabel an old immutable source.

    The fetched prefix establishes listing age and ADV20 before the retained
    model context. Every past universe is rebuilt from that day's actual prices,
    not from the target day's top stocks. This publishes no factor or labels.
    """
    root = Path(store_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise DataContractError("live initialization requires an empty production store_root")
    if config.universe.mode != "historical_liquid" or not config.quality_gate.enabled:
        raise DataContractError("live initialization requires historical_liquid quality gates")
    days = regular_sessions(revision.revision_start, revision.target_date)
    prefix = max(config.min_listed_sessions, 20)
    if len(days) < history_sessions + prefix:
        raise DataContractError("live initialization lacks listing/liquidity warm-up history")
    retained_start = _history_start(days, history_sessions)
    calendar = regular_session_frame(days[0], days[-1])
    bars = validate_bars(revision.bars)
    if bars.filter(~pl.col("trade_date").is_in(days)).height:
        raise DataContractError("live initialization contains dates outside its requested sessions")
    if bars["trade_date"].unique().sort().to_list() != days:
        raise TargetSessionIncomplete("live initialization has missing market sessions")
    known_bars = bars
    bars, identity_audit = quarantine_suspicious_identities(
        bars, calendar,
        max_adjusted_price_ratio=config.quality_gate.max_adjusted_price_ratio,
        max_alias_overlap_relative_diff=config.quality_gate.max_alias_overlap_relative_diff,
        max_quarantined_security_fraction=config.quality_gate.max_quarantined_security_fraction,
    )
    bars = validate_bars(bars)
    quality_gate = assert_historical_ingestion_quality(
        bars, calendar, max_adjusted_price_ratio=config.quality_gate.max_adjusted_price_ratio,
    )
    universe = build_universe(
        bars, min_listed_sessions=config.min_listed_sessions,
        min_price=config.min_price, min_adv20_usd=config.min_adv20_usd,
        max_daily_symbols=config.universe.max_symbols, calendar=calendar,
    )
    quarantines = merge_quarantines(
        revision_quarantines(revision.raw_quality_audits), audit_quarantines(identity_audit),
    )
    universe = retain_quarantined_candidates(
        universe, universe.head(0), known_bars, quarantines, days=days,
        metadata_rows=revision.metadata_rows, exchange_code=config.exchange_code,
    )
    frames = {
        "bars": bars.filter(pl.col("trade_date") >= retained_start),
        "universe": universe.filter(pl.col("trade_date") >= retained_start),
    }
    revision_id = uuid.uuid4().hex
    _write_revision(root, revision_id, frames, {
        "provider": "eodhd",
        "resolved_start": retained_start.isoformat(),
        "resolved_end": revision.target_date.isoformat(),
        "source_revision": revision.source_revision,
        "ingested_at": revision.ingested_at.isoformat(),
        "production_revision": {
            "contract": PRODUCTION_SOURCE_CONTRACT,
            "kind": "live_initialization",
            "history_sessions": history_sessions,
        },
        "bootstrap": {
            "kind": "current_provider_identity",
            "fetched_start": days[0].isoformat(),
            "fetched_end": days[-1].isoformat(),
            "fetched_sessions": len(days),
            "history_sessions": history_sessions,
            **daily_revision_audit(revision),
        },
        "quality": {"gate": quality_gate, "identity": identity_audit},
        "quarantines": list(quarantines.values()),
        "selection": {"research_ready": False},
        "warnings": [
            "current provider identities apply only to this new production source; "
            "this is not a point-in-time historical security master",
            "historical industry and float market cap are unavailable",
        ],
    })
    _advance_current(root, revision_id)
    return load_current_revision(root)


def _adjustment_changes(
    old_bars: pl.DataFrame,
    revised_bars: pl.DataFrame,
    revision_start: date,
) -> list[str]:
    comparison = (
        old_bars.filter(pl.col("trade_date") >= revision_start)
        .select("security_id", "trade_date", pl.col("adj_factor").alias("old_factor"))
        .join(
            revised_bars.select(
                "security_id", "trade_date", pl.col("adj_factor").alias("new_factor")
            ),
            on=["security_id", "trade_date"],
            how="inner",
        )
        .filter((pl.col("old_factor") - pl.col("new_factor")).abs() > 1e-12)
    )
    return sorted(comparison["security_id"].unique().to_list())


def _refresh_current_metadata(
    bars: pl.DataFrame,
    revision: EODHDDailyRevision,
    exchange_code: str,
) -> pl.DataFrame:
    metadata = build_metadata_index(list(revision.metadata_rows), exchange_code)
    records: list[dict[str, Any]] = []
    for provider_symbol, row in metadata.items():
        security_id, identity_quality = security_identity(provider_symbol, row)
        records.append(
            {
                "provider_symbol": provider_symbol,
                "_security_id": security_id,
                "_symbol": display_symbol(provider_symbol),
                "_identity_quality": identity_quality,
                "_exchange_source": row.get("exchange"),
                "_security_type_source": row.get("security_type"),
                "_is_delisted_source": bool(row.get("is_delisted", False)),
            }
        )
    if not records:
        return bars
    refreshed = (
        bars.join(pl.DataFrame(records), on="provider_symbol", how="left", validate="m:1")
        .with_columns(
            pl.coalesce("_security_id", "security_id").alias("security_id"),
            pl.coalesce("_symbol", "symbol").alias("symbol"),
            pl.coalesce("_identity_quality", "identity_quality").alias(
                "identity_quality"
            ),
            pl.coalesce("_exchange_source", "exchange_source").alias(
                "exchange_source"
            ),
            pl.coalesce("_security_type_source", "security_type_source").alias(
                "security_type_source"
            ),
            pl.coalesce("_is_delisted_source", "is_delisted_source").alias(
                "is_delisted_source"
            ),
        )
        .drop(
            "_security_id",
            "_symbol",
            "_identity_quality",
            "_exchange_source",
            "_security_type_source",
            "_is_delisted_source",
        )
    )
    return consolidate_bars(refreshed)


def _listed_day_offsets(
    old_universe: pl.DataFrame,
    hot_bars: pl.DataFrame,
    calendar_days: list[date],
    target_date: date,
    universe_start: date,
) -> pl.DataFrame:
    if target_date not in calendar_days:
        raise DataContractError("target date is absent from production source calendar")
    position = {day: index for index, day in enumerate(calendar_days)}
    short_start_index = position[universe_start]
    latest_membership = old_universe.sort("trade_date").unique("security_id", keep="last")
    old_listed = {
        security_id: (position.get(old_date), int(listed_days))
        for security_id, old_date, listed_days in latest_membership.select(
            "security_id", "trade_date", "listed_days"
        ).iter_rows()
    }
    records: list[dict[str, Any]] = []
    for security_id, first_date in hot_bars.group_by("security_id").agg(
        pl.col("trade_date").min().alias("first_date")
    ).iter_rows():
        first_position = position.get(first_date)
        if first_position is None:
            continue
        raw_count = len(calendar_days) - max(first_position, short_start_index)
        old_position, old_count = old_listed.get(security_id, (None, 0))
        if old_position is not None:
            desired = old_count + len(calendar_days) - 1 - old_position
        else:
            desired = len(calendar_days) - first_position
        records.append(
            {
                "security_id": security_id,
                "listed_day_offset": max(desired - raw_count, 0),
            }
        )
    return pl.DataFrame(
        records,
        schema={"security_id": pl.String, "listed_day_offset": pl.Int64},
    )


def prune_source_revisions(store_root: str | Path, *, keep: set[str]) -> list[Path]:
    store_root = Path(store_root).resolve()
    revisions = store_root / "revisions"
    if not revisions.is_dir():
        return []
    removed: list[Path] = []
    for child in revisions.iterdir():
        if child.is_dir() and not child.is_symlink() and child.name not in keep:
            shutil.rmtree(child)
            removed.append(child)
    return removed


def publish_daily_source_revision(
    current: ProductionSourceRevision,
    revision: EODHDDailyRevision,
    config: EODHDConfig,
    store_root: str | Path,
    *,
    history_sessions: int,
) -> ProductionSourceRevision:
    """Commit valid source data; production owns scoring and delivery readiness."""

    old_bars = validate_bars(pl.read_parquet(current.root / PRODUCTION_SOURCE_FILES["bars"]))
    old_universe = validate_universe(
        pl.read_parquet(current.root / PRODUCTION_SOURCE_FILES["universe"])
    )
    known_bars = pl.concat([old_bars, revision.bars], how="vertical_relaxed")
    quarantines = _source_quarantines(current)
    fresh_quarantines = revision_quarantines(revision.raw_quality_audits)
    metadata = build_metadata_index(list(revision.metadata_rows), config.exchange_code)
    aliases: dict[str, set[str]] = {}
    for symbol, row in metadata.items():
        security_id, _ = security_identity(symbol, row)
        aliases.setdefault(security_id, set()).add(symbol)
    for row in old_universe.select("security_id", "provider_symbol").unique().to_dicts():
        symbol = row["provider_symbol"]
        if (
            symbol in metadata
            and security_identity(symbol, metadata[symbol])[0] != row["security_id"]
        ):
            raise DataContractError(
                f"daily metadata remapped existing production identity: {symbol}; "
                "review and re-bootstrap before publishing"
            )
    # A new, checked full context may clear an exclusion; a clean short window cannot.
    required_days = regular_sessions(
        date.fromisoformat(current.manifest["resolved_start"]), revision.target_date,
    )[-history_sessions:]
    for security_id in list(quarantines):
        history = revision.bars.filter(pl.col("security_id") == security_id)
        required_aliases = aliases.get(security_id, set()) | set(
            quarantines[security_id].get("provider_symbols", [])
        )
        aliases_reviewed = (
            bool(required_aliases)
            and required_aliases.issubset(set(revision.backfilled_provider_symbols))
        )
        alias_sensitive = (
            len(required_aliases) > 1
            or "alias_overlap_conflict" in quarantines[security_id]["reasons"]
        )
        full_review = aliases_reviewed or (
            not alias_sensitive and revision.revision_start <= required_days[0]
        )
        if (
            security_id not in fresh_quarantines and history.height and full_review
            and set(required_days).issubset(set(history["trade_date"]))
        ):
            # The final merged-source gate below revalidates the full history.
            del quarantines[security_id]
    quarantines = merge_quarantines(quarantines, fresh_quarantines)
    previous_ids = set(old_bars["security_id"])
    old_bars = validate_bars(
        _refresh_current_metadata(old_bars, revision, config.exchange_code)
    )
    remapped_ids = sorted(
        previous_ids
        - set(old_bars["security_id"].unique().to_list())
    )
    if remapped_ids:
        raise DataContractError(
            "daily metadata remapped existing production identities; review and "
            f"re-bootstrap before publishing ({len(remapped_ids)} identities)"
        )
    old_bars = old_bars.filter(~pl.col("security_id").is_in(quarantines.keys()))
    revised_bars = revision.bars.filter(~pl.col("security_id").is_in(quarantines.keys()))
    changed_adjustments = _adjustment_changes(
        old_bars,
        revised_bars,
        revision.revision_start,
    )
    backfilled_ids = {
        security_identity(symbol, metadata.get(symbol))[0]
        for symbol in revision.backfilled_provider_symbols
    }
    if changed_adjustments:
        insufficient = [
            security_id
            for security_id in changed_adjustments
            if security_id not in backfilled_ids
        ]
        if insufficient:
            symbols = sorted(
                old_bars.filter(pl.col("security_id").is_in(insufficient))[
                    "provider_symbol"
                ]
                .unique()
                .to_list()
            )
            history_start = old_bars.filter(pl.col("security_id").is_in(insufficient))[
                "trade_date"
            ].min()
            assert history_start is not None
            raise AdjustmentBackfillRequired(insufficient, symbols, history_start)

    # A response starting early is not proof that its middle is complete.
    for security_id in sorted(backfilled_ids):
        previous = old_bars.filter(
            (pl.col("security_id") == security_id) & (pl.col("trade_date") >= required_days[0])
        )
        replacement = revised_bars.filter(pl.col("security_id") == security_id)
        missing = sorted(set(previous["trade_date"]) - set(replacement["trade_date"]))
        if missing:
            quarantines[security_id] = {
                "security_id": security_id, "reasons": ["incomplete_adjustment_history"],
                "provider_symbols": sorted(previous["provider_symbol"].unique()),
                "first_trade_date": str(missing[0]), "last_trade_date": str(missing[-1]),
                "examples": [{"missing_dates": [str(day) for day in missing[:3]]}],
            }

    unchanged_old = old_bars.filter(
        (~pl.col("security_id").is_in(backfilled_ids))
        & (pl.col("trade_date") < revision.revision_start)
    )
    bars = validate_bars(
        _refresh_current_metadata(
            pl.concat([unchanged_old, revised_bars], how="vertical_relaxed").filter(
                ~pl.col("security_id").is_in(quarantines.keys())
            ),
            revision,
            config.exchange_code,
        )
    )
    start = bars["trade_date"].min()
    if start is None:
        raise DataContractError("merged production bars are empty")
    source_calendar = regular_session_frame(start, revision.target_date)
    bars = validate_bars(
        bars.join(source_calendar, on="trade_date", how="inner", validate="m:1")
    )
    start = bars["trade_date"].min()
    if start is None or bars["trade_date"].max() != revision.target_date:
        raise DataContractError("merged production bars do not end on the target date")
    calendar = regular_session_frame(start, revision.target_date)
    missing_sessions = calendar.join(
        bars.select("trade_date").unique(), on="trade_date", how="anti",
    )
    if missing_sessions.height:
        raise TargetSessionIncomplete(
            "daily source has missing market sessions: "
            + ", ".join(day.isoformat() for day in missing_sessions["trade_date"])
        )
    clean_bars, identity_audit = quarantine_suspicious_identities(
        bars,
        calendar,
        max_adjusted_price_ratio=config.quality_gate.max_adjusted_price_ratio,
        max_alias_overlap_relative_diff=config.quality_gate.max_alias_overlap_relative_diff,
        max_quarantined_security_fraction=(
            config.quality_gate.max_quarantined_security_fraction
        ),
    )
    clean_bars = validate_bars(clean_bars)
    quarantines = merge_quarantines(quarantines, audit_quarantines(identity_audit))
    total_ids = set(known_bars["security_id"]) | set(quarantines)
    if len(quarantines) / len(total_ids) > config.quality_gate.max_quarantined_security_fraction:
        raise DataContractError(
            "production quality quarantine exceeds configured limit: "
            f"{len(quarantines)}/{len(total_ids)}"
        )
    quality_gate = assert_historical_ingestion_quality(
        clean_bars,
        calendar,
        max_adjusted_price_ratio=config.quality_gate.max_adjusted_price_ratio,
    )
    sessions = regular_sessions(start, revision.target_date)
    _history_start(sessions, history_sessions)
    retained_sessions = set(sessions[-history_sessions:])
    retained_start = sessions[max(0, len(sessions) - history_sessions)]
    # Rebuild all revised/gap sessions with ADV20 warm-up; preserve actual membership
    # outside the revision window, rather than projecting today's stock pool backwards.
    rebuild_dates = [day for day in sessions if day >= revision.revision_start]
    if not rebuild_dates:
        raise DataContractError("daily revision has no retained universe sessions")
    rebuild_position = sessions.index(rebuild_dates[0])
    if rebuild_position < 19:
        raise DataContractError("daily universe revision has insufficient ADV20 warm-up history")
    universe_calendar_days = sessions[rebuild_position - 19:]
    hot_calendar = pl.DataFrame(
        {"trade_date": universe_calendar_days},
        schema={"trade_date": pl.Date},
    )
    hot_bars = validate_bars(
        clean_bars.filter(pl.col("trade_date").is_in(retained_sessions))
    )
    universe = build_universe(
        clean_bars,
        min_listed_sessions=config.min_listed_sessions,
        min_price=config.min_price,
        min_adv20_usd=config.min_adv20_usd,
        max_daily_symbols=config.universe.max_symbols,
        calendar=hot_calendar,
        listed_day_offsets=_listed_day_offsets(
            old_universe,
            clean_bars,
            sessions,
            revision.target_date,
            universe_calendar_days[0],
        ),
    )
    universe = validate_universe(
        pl.concat(
            [
                old_universe.filter(pl.col("trade_date") < rebuild_dates[0]),
                universe.filter(pl.col("trade_date") >= rebuild_dates[0]),
            ],
            how="vertical_relaxed",
        ).filter(pl.col("trade_date").is_in(retained_sessions))
    )
    universe = validate_universe(retain_quarantined_candidates(
        universe, old_universe, known_bars, quarantines, days=sessions[-history_sessions:],
        metadata_rows=revision.metadata_rows, exchange_code=config.exchange_code,
    ))
    observed_target_rows = hot_bars.filter(
        pl.col("trade_date") == revision.target_date
    ).height
    target_universe = universe.filter(pl.col("trade_date") == revision.target_date)
    candidate_rows = target_universe.height
    eligible_rows = target_universe.filter(pl.col("eligible")).height
    revision_metadata = {
        "contract": PRODUCTION_SOURCE_CONTRACT,
        "parent_revision_id": current.revision_id,
        "target_date": revision.target_date.isoformat(),
        "revision_start": revision.revision_start.isoformat(),
        "history_sessions": history_sessions,
    }
    revision_id = uuid.uuid4().hex
    root = Path(store_root).resolve()
    _write_revision(
        root,
        revision_id,
        {"bars": hot_bars, "universe": universe},
        {
            "provider": "eodhd",
            "resolved_start": retained_start.isoformat(),
            "resolved_end": revision.target_date.isoformat(),
            "source_revision": revision.source_revision,
            "ingested_at": revision.ingested_at.isoformat(),
            "production_revision": revision_metadata,
            "warnings": [
                "historical industry and float market cap are unavailable",
                "delisting terminal returns are imputed policy assumptions",
            ],
            "selection": {"research_ready": False},
            "daily_update": {
                **daily_revision_audit(revision),
                "backfilled_provider_symbols": [
                    symbol for symbol in revision.backfilled_provider_symbols
                    if security_identity(symbol, metadata.get(symbol))[0] not in quarantines
                ],
                "adjustment_changed_securities": changed_adjustments,
                "target_observed_rows": observed_target_rows,
                "target_candidate_rows": candidate_rows,
                "target_eligible_rows": eligible_rows,
            },
            "quality": {"gate": quality_gate, "identity": identity_audit},
            "quarantines": list(quarantines.values()),
        },
    )
    _advance_current(root, revision_id)
    return load_current_revision(root)
