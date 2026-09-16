"""Fixed-release historical factor replay for isolated backtests."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import polars as pl
import yaml
from pydantic import Field, field_validator, model_validator

from facdigger.data.config import StrictModel
from facdigger.data.contracts import DataContractError
from facdigger.data.inference_snapshots import load_inference_snapshot
from facdigger.data.market_calendar import CALENDAR_VERSION
from facdigger.data.snapshots import sha256_file
from facdigger.inference.delivery import (
    DeliveryConfig,
    DeliverySelection,
    delivery_identity_policy,
    resolve_delivery,
)
from facdigger.inference.factor_batch import (
    FactorBatchInput,
    FactorBatchModel,
    FactorBatchSource,
    FactorBatchTime,
    build_factor_frame,
    factor_batch_metadata,
    factor_universe_sha256,
    load_factor_batch,
    publish_factor_batch,
)
from facdigger.inference.scoring import (
    FactorInferenceRuntime,
    load_factor_inference_runtime,
    score_inference_rows,
)

HISTORICAL_REPLAY_CONTRACT = "facdigger.historical_factor_replay"
HISTORICAL_PLAN_CONTRACT = "facdigger.historical_factor_replay_plan"
EXPORT_FILES = {
    "resolved_config.yaml",
    "plan.json",
    "state.json",
    "manifest.json",
    "factor_batches",
}


class HistoricalReplayConfig(StrictModel):
    """One explicit fixed-model, non-OOS historical replay request."""

    history_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,79}$")
    release_dir: Path
    inference_snapshot_dir: Path
    output_root: Path = Path("artifacts/factor_history")
    start_date: date | None = None
    end_date: date | None = None
    security_ids: list[str] = Field(default_factory=list)
    delivery: DeliveryConfig | None = None
    device: Literal["auto", "cpu", "cuda"] = "cpu"
    batch_size: int | None = Field(default=None, ge=1)
    num_workers: int | None = Field(default=None, ge=0)
    acknowledge_non_oos: Literal[True]

    @field_validator("security_ids")
    @classmethod
    def canonicalize_security_ids(cls, values: list[str]) -> list[str]:
        return sorted(values)

    @model_validator(mode="after")
    def validate_request(self) -> HistoricalReplayConfig:
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.start_date > self.end_date
        ):
            raise ValueError("start_date must not be after end_date")
        if any(not value.strip() for value in self.security_ids):
            raise ValueError("security_ids must not contain empty values")
        if len(self.security_ids) != len(set(self.security_ids)):
            raise ValueError("security_ids must be unique")
        if self.security_ids and self.delivery is not None:
            raise ValueError("use either security_ids or a consumer delivery profile, not both")
        return self


class HistoricalReplayPartition(StrictModel):
    year: int = Field(ge=1900, le=2200)
    minimum_asof_date: date
    maximum_asof_date: date
    row_count: int = Field(ge=1)
    date_count: int = Field(ge=1)
    security_count: int = Field(ge=1)
    delivery_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    path: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dates(self) -> HistoricalReplayPartition:
        if self.minimum_asof_date > self.maximum_asof_date:
            raise ValueError("partition minimum date must not exceed maximum date")
        if (
            self.minimum_asof_date.year != self.year
            or self.maximum_asof_date.year != self.year
        ):
            raise ValueError("partition dates must be inside its calendar year")
        return self


class HistoricalReplayManifest(StrictModel):
    contract: Literal["facdigger.historical_factor_replay"] = (
        HISTORICAL_REPLAY_CONTRACT
    )
    status: Literal["complete"] = "complete"
    history_id: str
    created_at: datetime
    purpose: Literal["backtest_only"] = "backtest_only"
    replay_mode: Literal["fixed_release_full_history"] = (
        "fixed_release_full_history"
    )
    strict_out_of_sample: Literal[False] = False
    model_parameters_may_use_future_data: Literal[True] = True
    scaler_may_use_future_data: Literal[True] = True
    source_data_may_include_later_revisions: Literal[True] = True
    paper_allowed: Literal[False] = False
    factor_batch_source_kind: Literal["evaluation_predictions"] = (
        "evaluation_predictions"
    )
    release_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    security_scope: Literal["all_eligible", "explicit_ids", "delivery_profile"]
    requested_security_ids: list[str]
    requested_start_date: date | None
    requested_end_date: date | None
    minimum_asof_date: date
    maximum_asof_date: date
    row_count: int = Field(ge=1)
    date_count: int = Field(ge=1)
    security_count: int = Field(ge=1)
    omitted_date_count: int = Field(ge=0)
    partitions: list[HistoricalReplayPartition] = Field(min_length=1)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
            raise ValueError("created_at must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def validate_totals(self) -> HistoricalReplayManifest:
        if self.minimum_asof_date > self.maximum_asof_date:
            raise ValueError("minimum_asof_date must not exceed maximum_asof_date")
        years = [partition.year for partition in self.partitions]
        if years != sorted(years) or len(years) != len(set(years)):
            raise ValueError("historical partitions must have unique ascending years")
        if sum(partition.row_count for partition in self.partitions) != self.row_count:
            raise ValueError("historical partition row counts disagree with manifest")
        if sum(partition.date_count for partition in self.partitions) != self.date_count:
            raise ValueError("historical partition date counts disagree with manifest")
        if self.partitions[0].minimum_asof_date != self.minimum_asof_date:
            raise ValueError("historical minimum date disagrees with first partition")
        if self.partitions[-1].maximum_asof_date != self.maximum_asof_date:
            raise ValueError("historical maximum date disagrees with last partition")
        if self.security_scope == "all_eligible" and self.requested_security_ids:
            raise ValueError("all_eligible scope must not declare security IDs")
        if self.security_scope == "explicit_ids" and not self.requested_security_ids:
            raise ValueError("explicit_ids scope requires security IDs")
        return self


HistoricalReplayProgress = Callable[
    [Literal["scoring", "published", "verified"], int, HistoricalReplayPartition | None],
    None,
]


@dataclass(frozen=True)
class _ResolvedReplay:
    config: HistoricalReplayConfig
    runtime: FactorInferenceRuntime
    snapshot_manifest: dict[str, Any]
    rows: pl.DataFrame
    plan: dict[str, Any]
    delivery: DeliverySelection


def load_historical_replay_config(
    path: str | Path,
    *,
    release_dir: str | Path | None = None,
    inference_snapshot_dir: str | Path | None = None,
    output_root: str | Path | None = None,
) -> HistoricalReplayConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"historical replay configuration not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    for key, value in {
        "release_dir": release_dir, "inference_snapshot_dir": inference_snapshot_dir,
        "output_root": output_root,
    }.items():
        if value is not None:
            raw[key] = value
    return HistoricalReplayConfig.model_validate(raw)


def _normalized_config(config: HistoricalReplayConfig) -> dict[str, Any]:
    payload = config.model_dump(mode="json")
    if config.delivery is None:
        payload.pop("delivery")
    for name in ("release_dir", "inference_snapshot_dir", "output_root"):
        payload[name] = str(Path(payload[name]).resolve())
    return payload


def _partition_rows(rows: pl.DataFrame) -> list[tuple[int, pl.DataFrame]]:
    years = sorted(set(rows["asof_date"].dt.year().to_list()))
    return [
        (
            year,
            rows.filter(pl.col("asof_date").dt.year() == year).sort(
                "asof_date", "security_id"
            ),
        )
        for year in years
    ]


def _paths_overlap(left: Path, right: Path) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


def _resolve_replay(config: HistoricalReplayConfig) -> _ResolvedReplay:
    release_dir = config.release_dir.resolve()
    snapshot_dir = config.inference_snapshot_dir.resolve()
    if not release_dir.is_dir():
        raise FileNotFoundError(
            f"historical replay release does not exist: {release_dir}; specify --release locally"
        )
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(
            f"historical replay inference snapshot does not exist: {snapshot_dir}; "
            "specify --dataset locally"
        )
    destination = (config.output_root.resolve() / config.history_id).resolve()
    if _paths_overlap(destination, release_dir) or _paths_overlap(
        destination, snapshot_dir
    ):
        raise DataContractError(
            "historical replay output must not overlap its immutable release or snapshot"
        )
    runtime = load_factor_inference_runtime(
        release_dir,
        device=config.device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    snapshot_manifest, frames = load_inference_snapshot(snapshot_dir, runtime.release)
    index = frames["inference_index"]
    available_start = cast(date, index["asof_date"].min())
    available_end = cast(date, index["asof_date"].max())
    requested_start = config.start_date or available_start
    requested_end = config.end_date or available_end
    if requested_start < available_start or requested_start > available_end:
        raise DataContractError(
            "historical replay start_date is outside the scorable snapshot range"
        )
    if requested_end < available_start or requested_end > available_end:
        raise DataContractError(
            "historical replay end_date is outside the scorable snapshot range"
        )
    if requested_start > requested_end:
        raise DataContractError("historical replay date range contains no ordered dates")

    range_rows = index.filter(
        pl.col("asof_date").is_between(requested_start, requested_end)
    )
    rows = range_rows
    if config.security_ids:
        available_ids = set(range_rows["security_id"].unique().to_list())
        unknown = sorted(set(config.security_ids) - available_ids)
        if unknown:
            raise DataContractError(
                "historical replay security IDs are absent from the requested date "
                f"range: {unknown}"
            )
        rows = rows.filter(pl.col("security_id").is_in(config.security_ids))
    candidates = frames["delivery_universe"].filter(
        pl.col("asof_date").is_between(requested_start, requested_end)
    )
    if config.security_ids:
        candidates = candidates.filter(pl.col("security_id").is_in(config.security_ids))
    delivery = resolve_delivery(candidates, config.delivery, eligible_only=True)
    rows = rows.join(
        delivery.source_candidates.select("security_id", "asof_date"),
        on=["security_id", "asof_date"], how="semi",
    ).sort("asof_date", "security_id")
    if rows.is_empty():
        raise DataContractError("historical replay selection contains no scorable rows")
    if not rows["eligible"].all():
        raise DataContractError("historical inference index contains non-eligible rows")

    if rows.height != delivery.candidates.height:
        raise DataContractError("historical delivery is missing eligible inference rows")
    selected_dates = rows["asof_date"].n_unique()
    available_dates = range_rows["asof_date"].n_unique()
    partition_plan = [
        {
            "year": year,
            "minimum_asof_date": partition["asof_date"].min().isoformat(),
            "maximum_asof_date": partition["asof_date"].max().isoformat(),
            "row_count": partition.height,
            "date_count": partition["asof_date"].n_unique(),
            "security_count": partition["security_id"].n_unique(),
        }
        for year, partition in _partition_rows(delivery.candidates)
    ]
    plan = {
        "contract": HISTORICAL_PLAN_CONTRACT,
        "history_id": config.history_id,
        "purpose": "backtest_only",
        "replay_mode": "fixed_release_full_history",
        "strict_out_of_sample": False,
        "model_parameters_may_use_future_data": True,
        "scaler_may_use_future_data": True,
        "source_data_may_include_later_revisions": True,
        "paper_allowed": False,
        "factor_batch_source_kind": "evaluation_predictions",
        "release_id": runtime.release.release_id,
        "model_id": runtime.release.model_id,
        "checkpoint_sha256": runtime.release.artifacts["checkpoint"].sha256,
        "snapshot_id": str(snapshot_manifest["snapshot_id"]),
        "snapshot_manifest_sha256": sha256_file(snapshot_dir / "manifest.json"),
        "available_start_date": available_start.isoformat(),
        "available_end_date": available_end.isoformat(),
        "requested_start_date": (
            None if config.start_date is None else config.start_date.isoformat()
        ),
        "requested_end_date": (
            None if config.end_date is None else config.end_date.isoformat()
        ),
        "minimum_asof_date": rows["asof_date"].min().isoformat(),
        "maximum_asof_date": rows["asof_date"].max().isoformat(),
        "security_scope": (
            "delivery_profile" if config.delivery is not None
            else "explicit_ids" if config.security_ids else "all_eligible"
        ),
        "requested_security_ids": list(config.security_ids),
        "row_count": rows.height,
        "date_count": selected_dates,
        "security_count": delivery.candidates["security_id"].n_unique(),
        "omitted_date_count": available_dates - selected_dates,
        "partitions": partition_plan,
    }
    if delivery.audit is not None:
        plan["delivery_audit"] = delivery.audit
    return _ResolvedReplay(
        config=config,
        runtime=runtime,
        snapshot_manifest=snapshot_manifest,
        rows=rows,
        plan=plan,
        delivery=delivery,
    )


def plan_historical_replay(config: HistoricalReplayConfig) -> dict[str, Any]:
    """Validate inputs and return an explicitly non-strict-OOS plan without scoring."""

    return _resolve_replay(config).plan


def _write_json(path: Path, payload: Any, *, atomic: bool = False) -> None:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    ) + "\n"
    if not atomic:
        path.write_text(serialized, encoding="utf-8")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"historical replay {label} is unreadable") from exc


def _prepare_export(resolved: _ResolvedReplay) -> tuple[Path, dict[str, str]]:
    config = resolved.config
    destination = (config.output_root.resolve() / config.history_id).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.mkdir()
    elif not destination.is_dir():
        raise DataContractError("historical replay destination is not a directory")
    elif any(path.is_symlink() for path in destination.iterdir()):
        raise DataContractError(
            "historical replay destination entries must not be symbolic links"
        )
    unexpected = {path.name for path in destination.iterdir()} - EXPORT_FILES
    if unexpected:
        raise DataContractError(
            f"historical replay destination contains undeclared entries: {sorted(unexpected)}"
        )

    normalized = _normalized_config(config)
    config_path = destination / "resolved_config.yaml"
    if config_path.exists():
        observed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        observed_config = HistoricalReplayConfig.model_validate(observed)
        locators = {"release_dir", "inference_snapshot_dir", "output_root"}
        if observed_config.model_dump(exclude=locators) != config.model_dump(exclude=locators):
            raise DataContractError(
                "existing historical replay uses a different resolved configuration"
            )
    else:
        if any(destination.iterdir()):
            raise DataContractError(
                "non-empty historical replay destination has no resolved configuration"
            )
        _write_yaml(config_path, normalized)

    plan_path = destination / "plan.json"
    if plan_path.exists():
        if _read_json(plan_path, "plan") != resolved.plan:
            raise DataContractError("existing historical replay plan differs from inputs")
    else:
        _write_json(plan_path, resolved.plan)

    factor_root = destination / "factor_batches"
    if factor_root.exists():
        if factor_root.is_symlink() or not factor_root.is_dir():
            raise DataContractError("historical factor root must be a real directory")
        if any(
            path.is_symlink() or not path.is_dir() for path in factor_root.iterdir()
        ):
            raise DataContractError(
                "historical factor root may contain only delivery directories"
            )
    else:
        factor_root.mkdir()
    state_path = destination / "state.json"
    if state_path.exists():
        state = _read_json(state_path, "state")
        if not isinstance(state, dict) or set(state) != {"history_id", "completed"}:
            raise DataContractError("historical replay state has invalid fields")
        if state["history_id"] != config.history_id or not isinstance(
            state["completed"], dict
        ):
            raise DataContractError("historical replay state disagrees with configuration")
        completed = state["completed"]
        if any(
            not str(year).isdigit()
            or not isinstance(delivery_id, str)
            or re.fullmatch(r"[0-9a-f]{64}", delivery_id) is None
            for year, delivery_id in completed.items()
        ):
            raise DataContractError("historical replay completed partition state is invalid")
        planned_years = {
            str(partition["year"]) for partition in resolved.plan["partitions"]
        }
        if not set(completed).issubset(planned_years) or len(
            set(completed.values())
        ) != len(completed):
            raise DataContractError("historical replay state contains invalid years")
    else:
        completed = {}
        _write_json(
            state_path,
            {"history_id": config.history_id, "completed": completed},
            atomic=True,
        )
    return destination, cast(dict[str, str], completed)


def _validate_partition_bundle(
    bundle_dir: Path,
    *,
    resolved: _ResolvedReplay,
    expected_rows: pl.DataFrame,
) -> HistoricalReplayPartition:
    (
        candidate_universe,
        expected_source,
        expected_model,
        expected_input,
        expected_time,
    ) = _historical_batch_contract(resolved, expected_rows)
    manifest = load_factor_batch(bundle_dir)
    expected_keys = candidate_universe.select(
        "security_id", "symbol", "asof_date"
    )
    factors = pl.read_parquet(bundle_dir / "factors.parquet")
    if not factors["eligible"].all() or factors["score"].null_count():
        raise DataContractError("historical FactorBatch must contain only scored eligible rows")
    if not factors.select("security_id", "symbol", "asof_date").equals(expected_keys):
        raise DataContractError("historical FactorBatch keys differ from replay plan")
    if manifest.source != expected_source or manifest.model != expected_model:
        raise DataContractError("historical FactorBatch release lineage differs from replay plan")
    if manifest.input != expected_input:
        raise DataContractError("historical FactorBatch input differs from replay plan")
    if manifest.time != expected_time:
        raise DataContractError("historical FactorBatch time semantics differ from replay plan")
    minimum = cast(date, expected_rows["asof_date"].min())
    maximum = cast(date, expected_rows["asof_date"].max())
    year = minimum.year
    if maximum.year != year:
        raise DataContractError("historical FactorBatch spans more than one calendar year")
    return HistoricalReplayPartition(
        year=year,
        minimum_asof_date=minimum,
        maximum_asof_date=maximum,
        row_count=expected_rows.height,
        date_count=expected_rows["asof_date"].n_unique(),
        security_count=candidate_universe["security_id"].n_unique(),
        delivery_id=manifest.delivery_id,
        path=f"factor_batches/{manifest.delivery_id}",
    )


def _historical_batch_contract(
    resolved: _ResolvedReplay,
    rows: pl.DataFrame,
) -> tuple[
    pl.DataFrame,
    FactorBatchSource,
    FactorBatchModel,
    FactorBatchInput,
    FactorBatchTime,
]:
    candidate_universe = resolved.delivery.for_dates(
        rows["asof_date"].unique().to_list()
    ).candidates
    source, model = factor_batch_metadata(
        resolved.runtime.release,
        source_kind="evaluation_predictions",
    )
    input_metadata = FactorBatchInput(
        snapshot_id=str(resolved.snapshot_manifest["snapshot_id"]),
        snapshot_manifest_sha256=sha256_file(
            resolved.config.inference_snapshot_dir.resolve() / "manifest.json"
        ),
        universe_semantics="eligible_scored_cross_section",
        universe_sha256=factor_universe_sha256(candidate_universe),
        identity_policy=delivery_identity_policy(candidate_universe),
    )
    time = FactorBatchTime(
        calendar_version=CALENDAR_VERSION,
        minimum_asof_date=cast(date, rows["asof_date"].min()),
        maximum_asof_date=cast(date, rows["asof_date"].max()),
    )
    return candidate_universe, source, model, input_metadata, time


def _publish_partition(
    resolved: _ResolvedReplay,
    *,
    factor_root: Path,
    rows: pl.DataFrame,
) -> tuple[Path, HistoricalReplayPartition]:
    scored = score_inference_rows(
        resolved.runtime,
        snapshot_dir=resolved.config.inference_snapshot_dir,
        snapshot_manifest=resolved.snapshot_manifest,
        rows=rows,
    )
    (
        candidate_universe,
        source_metadata,
        model_metadata,
        input_metadata,
        time_metadata,
    ) = _historical_batch_contract(resolved, rows)
    factors = build_factor_frame(
        candidate_universe,
        resolved.delivery.for_dates(rows["asof_date"].unique().to_list()).project_scores(
            scored.select("security_id", "asof_date", "score")
        ),
    )
    destination, _ = publish_factor_batch(
        factors,
        factor_root,
        source=source_metadata,
        model=model_metadata,
        input_metadata=input_metadata,
        time_metadata=time_metadata,
    )
    partition = _validate_partition_bundle(
        destination,
        resolved=resolved,
        expected_rows=rows,
    )
    return destination, partition


def _build_manifest(
    resolved: _ResolvedReplay,
    partitions: list[HistoricalReplayPartition],
) -> HistoricalReplayManifest:
    plan = resolved.plan
    return HistoricalReplayManifest(
        history_id=resolved.config.history_id,
        created_at=datetime.now(timezone.utc),
        release_id=plan["release_id"],
        checkpoint_sha256=plan["checkpoint_sha256"],
        snapshot_id=plan["snapshot_id"],
        snapshot_manifest_sha256=plan["snapshot_manifest_sha256"],
        security_scope=plan["security_scope"],
        requested_security_ids=plan["requested_security_ids"],
        requested_start_date=plan["requested_start_date"],
        requested_end_date=plan["requested_end_date"],
        minimum_asof_date=plan["minimum_asof_date"],
        maximum_asof_date=plan["maximum_asof_date"],
        row_count=plan["row_count"],
        date_count=plan["date_count"],
        security_count=plan["security_count"],
        omitted_date_count=plan["omitted_date_count"],
        partitions=partitions,
    )


def run_historical_replay(
    config: HistoricalReplayConfig,
    *,
    on_partition: HistoricalReplayProgress | None = None,
) -> tuple[Path, HistoricalReplayManifest]:
    """Score and publish annual FactorBatch partitions, resuming completed years."""

    resolved = _resolve_replay(config)
    destination, completed = _prepare_export(resolved)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = verify_historical_replay(
            destination, release_dir=config.release_dir,
            inference_snapshot_dir=config.inference_snapshot_dir,
        )
        return destination, manifest

    factor_root = destination / "factor_batches"
    state_path = destination / "state.json"
    partitions: list[HistoricalReplayPartition] = []
    for year, rows in _partition_rows(resolved.rows):
        delivery_id = completed.get(str(year))
        if delivery_id is None:
            if on_partition is not None:
                on_partition("scoring", year, None)
            _, partition = _publish_partition(
                resolved,
                factor_root=factor_root,
                rows=rows,
            )
            completed[str(year)] = partition.delivery_id
            _write_json(
                state_path,
                {"history_id": config.history_id, "completed": completed},
                atomic=True,
            )
            if on_partition is not None:
                on_partition("published", year, partition)
        else:
            partition = _validate_partition_bundle(
                factor_root / delivery_id,
                resolved=resolved,
                expected_rows=rows,
            )
            if on_partition is not None:
                on_partition("verified", year, partition)
        partitions.append(partition)

    manifest = _build_manifest(resolved, partitions)
    _write_json(manifest_path, manifest.model_dump(mode="json"), atomic=True)
    return destination, verify_historical_replay(
        destination, release_dir=config.release_dir,
        inference_snapshot_dir=config.inference_snapshot_dir,
    )


def verify_historical_replay(
    export_dir: str | Path,
    *,
    release_dir: str | Path | None = None,
    inference_snapshot_dir: str | Path | None = None,
) -> HistoricalReplayManifest:
    """Verify the replay request and every declared annual FactorBatch."""

    root = Path(export_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"historical replay export does not exist: {root}")
    actual_entries = {path.name for path in root.iterdir()}
    if actual_entries != EXPORT_FILES:
        raise DataContractError("historical replay export has undeclared or missing entries")
    if any(path.is_symlink() for path in root.iterdir()):
        raise DataContractError("historical replay root entries must not be symbolic links")

    config_payload = yaml.safe_load(
        (root / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    if not isinstance(config_payload, dict):
        raise DataContractError("historical replay resolved configuration is invalid")
    config = load_historical_replay_config(
        root / "resolved_config.yaml", release_dir=release_dir,
        inference_snapshot_dir=inference_snapshot_dir, output_root=root.parent,
    )
    if root.name != config.history_id:
        raise DataContractError("historical replay directory name differs from history_id")
    resolved = _resolve_replay(config)
    if _read_json(root / "plan.json", "plan") != resolved.plan:
        raise DataContractError("historical replay plan no longer matches its inputs")

    manifest = HistoricalReplayManifest.model_validate(
        _read_json(root / "manifest.json", "manifest")
    )
    if manifest.history_id != config.history_id:
        raise DataContractError("historical replay manifest ID differs from configuration")
    factor_root = root / "factor_batches"
    if factor_root.is_symlink() or not factor_root.is_dir():
        raise DataContractError("historical factor root must be a real directory")
    if any(path.is_symlink() or not path.is_dir() for path in factor_root.iterdir()):
        raise DataContractError("historical factor root may contain only delivery directories")
    planned_partitions = _partition_rows(resolved.rows)
    if len(manifest.partitions) != len(planned_partitions):
        raise DataContractError("historical replay partition count differs from plan")
    verified_partitions: list[HistoricalReplayPartition] = []
    for (year, rows), declared in zip(
        planned_partitions,
        manifest.partitions,
        strict=True,
    ):
        expected_path = f"factor_batches/{declared.delivery_id}"
        if declared.year != year or declared.path != expected_path:
            raise DataContractError(
                "historical replay partition year or path differs from plan"
            )
        verified_partitions.append(
            _validate_partition_bundle(
                factor_root / declared.delivery_id,
                resolved=resolved,
                expected_rows=rows,
            )
        )
    expected_manifest = _build_manifest(
        resolved,
        verified_partitions,
    )
    observed_payload = manifest.model_dump(mode="json")
    expected_payload = expected_manifest.model_dump(mode="json")
    observed_payload.pop("created_at")
    expected_payload.pop("created_at")
    if observed_payload != expected_payload:
        raise DataContractError("historical replay manifest differs from verified partitions")

    expected_delivery_ids = {
        partition.delivery_id for partition in manifest.partitions
    }
    if {path.name for path in factor_root.iterdir()} != expected_delivery_ids:
        raise DataContractError("historical factor root differs from declared partitions")
    state = _read_json(root / "state.json", "state")
    expected_state = {
        "history_id": config.history_id,
        "completed": {
            str(partition.year): partition.delivery_id
            for partition in manifest.partitions
        },
    }
    if state != expected_state:
        raise DataContractError("historical replay state differs from completed manifest")
    return manifest
