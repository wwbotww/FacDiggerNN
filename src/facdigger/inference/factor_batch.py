"""Strict cross-project factor delivery contract and atomic publisher."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import uuid
from collections.abc import Callable, Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import polars as pl
from pydantic import Field, field_validator, model_validator

from facdigger.data.config import StrictModel
from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.experiments.manifest import sha256_json
from facdigger.inference.delivery import (
    DeliveryConfig,
    delivery_identity_policy,
    resolve_delivery,
)

if TYPE_CHECKING:
    from facdigger.inference.releases import ModelReleaseManifest

FACTOR_BATCH_CONTRACT = "facdigger.factor_batch"
FACTOR_COLUMNS = ["security_id", "symbol", "asof_date", "score", "eligible"]
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class FactorBatchSource(StrictModel):
    kind: Literal["evaluation_predictions", "signal_inference"]
    repository: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    run_id: str = Field(min_length=1)
    run_manifest_sha256: str

    @field_validator("run_manifest_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("run_manifest_sha256 must be a lowercase SHA-256 digest")
        return value


class FactorBatchModel(StrictModel):
    release_id: str
    model_id: str = Field(min_length=1)
    model_type: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    checkpoint_sha256: str
    training_dataset_id: str = Field(min_length=1)
    higher_score_is_better: Literal[True] = True
    forecast_horizon_sessions: int = Field(ge=1)
    score_semantics: Literal["raw_cross_sectional_rank_score"]

    @field_validator("release_id", "checkpoint_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("model digest fields must be lowercase SHA-256 digests")
        return value


class FactorBatchInput(StrictModel):
    snapshot_id: str = Field(min_length=1)
    snapshot_manifest_sha256: str
    universe_semantics: Literal[
        "eligible_scored_cross_section", "complete_candidate_cross_section"
    ]
    universe_sha256: str
    identity_policy: Literal["provider_neutral_security_id", "eodhd_isin_only"]

    @field_validator("snapshot_manifest_sha256", "universe_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("input digest fields must be lowercase SHA-256 digests")
        return value


class FactorBatchTime(StrictModel):
    calendar: Literal["US_EQUITIES_REGULAR"] = "US_EQUITIES_REGULAR"
    calendar_version: str = Field(min_length=1)
    timezone: Literal["America/New_York"] = "America/New_York"
    minimum_asof_date: date
    maximum_asof_date: date
    signal_available: Literal["after_regular_session_close"] = (
        "after_regular_session_close"
    )
    earliest_execution: Literal["next_regular_session_open"] = (
        "next_regular_session_open"
    )

    @model_validator(mode="after")
    def validate_range(self) -> FactorBatchTime:
        if self.minimum_asof_date > self.maximum_asof_date:
            raise ValueError("minimum_asof_date must not exceed maximum_asof_date")
        return self


class FactorBatchCoverage(StrictModel):
    candidate_rows: int = Field(ge=1)
    actual_rows: int = Field(ge=1)
    expected_eligible_rows: int = Field(ge=0)
    scored_eligible_rows: int = Field(ge=0)
    missing_eligible_rows: Literal[0] = 0
    ratio: Literal[1.0] = 1.0

    @model_validator(mode="after")
    def validate_counts(self) -> FactorBatchCoverage:
        if self.candidate_rows != self.actual_rows:
            raise ValueError("candidate_rows must equal actual_rows")
        if self.expected_eligible_rows != self.scored_eligible_rows:
            raise ValueError("every expected eligible row must be scored")
        if self.expected_eligible_rows > self.candidate_rows:
            raise ValueError("expected eligible rows cannot exceed candidate rows")
        return self


class FactorBatchArtifact(StrictModel):
    file: Literal["factors.parquet"] = "factors.parquet"
    sha256: str
    bytes: int = Field(ge=1)
    row_count: int = Field(ge=1)
    date_count: int = Field(ge=1)

    @field_validator("sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("artifact.sha256 must be a lowercase SHA-256 digest")
        return value


class FactorBatchManifest(StrictModel):
    contract: Literal["facdigger.factor_batch"] = FACTOR_BATCH_CONTRACT
    status: Literal["complete"] = "complete"
    delivery_id: str
    created_at: datetime
    source: FactorBatchSource
    model: FactorBatchModel
    input: FactorBatchInput
    time: FactorBatchTime
    coverage: FactorBatchCoverage
    artifact: FactorBatchArtifact

    @field_validator("delivery_id")
    @classmethod
    def validate_delivery_id(cls, value: str) -> str:
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("delivery_id must be a lowercase SHA-256 digest")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
            raise ValueError("created_at must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def validate_source_semantics(self) -> FactorBatchManifest:
        required = {
            "evaluation_predictions": "eligible_scored_cross_section",
            "signal_inference": "complete_candidate_cross_section",
        }[self.source.kind]
        if self.input.universe_semantics != required:
            raise ValueError("source kind and universe semantics disagree")
        if (
            self.source.kind == "signal_inference"
            and self.time.minimum_asof_date != self.time.maximum_asof_date
        ):
            raise ValueError("signal_inference must contain exactly one as-of date")
        if (
            self.source.kind == "evaluation_predictions"
            and self.coverage.candidate_rows != self.coverage.expected_eligible_rows
        ):
            raise ValueError("evaluation_predictions may contain only eligible scored rows")
        return self


def _factor_schema() -> dict[str, pl.DataType]:
    return {
        "security_id": pl.String,
        "symbol": pl.String,
        "asof_date": pl.Date,
        "score": pl.Float64,
        "eligible": pl.Boolean,
    }


def validate_factor_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Return the canonical FactorBatch frame or fail closed."""

    if frame.columns != FACTOR_COLUMNS:
        raise DataContractError(f"factor columns must exactly equal {FACTOR_COLUMNS}")
    expected_schema = _factor_schema()
    if frame.schema != expected_schema:
        raise DataContractError(
            f"factor schema mismatch: expected={expected_schema}, actual={dict(frame.schema)}"
        )
    if frame.is_empty():
        raise DataContractError("factor frame must not be empty")
    required_nulls = (
        frame.select("security_id", "symbol", "asof_date", "eligible")
        .null_count()
        .sum_horizontal()
        .item()
    )
    if required_nulls:
        raise DataContractError("factor identity, date and eligibility fields must be non-null")
    if frame.filter(pl.col("security_id").str.len_chars() == 0).height:
        raise DataContractError("factor security_id must be non-empty")
    if frame.filter(pl.col("symbol").str.len_chars() == 0).height:
        raise DataContractError("factor symbol must be non-empty")
    duplicates = frame.group_by("security_id", "asof_date").len().filter(pl.col("len") > 1)
    if duplicates.height:
        raise DataContractError("factor frame has duplicate (security_id, asof_date) keys")
    eligible = frame.filter(pl.col("eligible"))
    if eligible["score"].null_count() or not all(
        math.isfinite(float(value)) for value in eligible["score"].to_list()
    ):
        raise DataContractError("eligible factor scores must be finite and non-null")
    if frame.filter(~pl.col("eligible") & pl.col("score").is_not_null()).height:
        raise DataContractError("ineligible factor scores must be null")
    canonical = frame.sort("asof_date", "security_id")
    if not frame.equals(canonical, null_equal=True):
        raise DataContractError("factor frame must be sorted by asof_date and security_id")
    return canonical


def _identity_payload(manifest: FactorBatchManifest) -> dict[str, Any]:
    payload = manifest.model_dump(mode="json")
    payload.pop("created_at")
    payload.pop("delivery_id")
    return payload


def factor_batch_delivery_id(manifest: FactorBatchManifest) -> str:
    """Hash every semantic field and the factor artifact, excluding wall-clock metadata."""

    return sha256_json(_identity_payload(manifest))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_factor_frame(
    candidate_universe: pl.DataFrame,
    scored_rows: pl.DataFrame,
) -> pl.DataFrame:
    """Join scored eligible rows onto the complete declared delivery universe."""
    candidate_columns = ["security_id", "symbol", "asof_date", "eligible"]
    score_columns = ["security_id", "asof_date", "score"]
    if candidate_universe.columns != candidate_columns:
        raise DataContractError(
            f"candidate universe columns must exactly equal {candidate_columns}"
        )
    if scored_rows.columns != score_columns:
        raise DataContractError(f"scored row columns must exactly equal {score_columns}")
    if candidate_universe.is_empty():
        raise DataContractError("candidate universe must not be empty")
    if candidate_universe.schema != {
        "security_id": pl.String,
        "symbol": pl.String,
        "asof_date": pl.Date,
        "eligible": pl.Boolean,
    }:
        raise DataContractError("candidate universe schema is not canonical")
    if scored_rows.schema != {
        "security_id": pl.String,
        "asof_date": pl.Date,
        "score": pl.Float64,
    }:
        raise DataContractError("scored row schema is not canonical")
    candidate_nulls = candidate_universe.null_count().sum_horizontal().item()
    if candidate_nulls:
        raise DataContractError("candidate universe fields must be non-null")
    score_key_nulls = (
        scored_rows.select("security_id", "asof_date")
        .null_count()
        .sum_horizontal()
        .item()
    )
    if score_key_nulls:
        raise DataContractError("scored row keys must be non-null")
    candidate_keys = candidate_universe.select("security_id", "asof_date")
    if candidate_keys.n_unique() != candidate_universe.height:
        raise DataContractError("candidate universe has duplicate identity/date keys")
    if scored_rows.select("security_id", "asof_date").n_unique() != scored_rows.height:
        raise DataContractError("scored rows have duplicate identity/date keys")
    unexpected = scored_rows.join(
        candidate_keys,
        on=["security_id", "asof_date"],
        how="anti",
    )
    if unexpected.height:
        raise DataContractError("scored rows contain keys outside the candidate universe")
    joined = candidate_universe.join(
        scored_rows,
        on=["security_id", "asof_date"],
        how="left",
        validate="1:1",
    )
    missing_eligible = joined.filter(pl.col("eligible") & pl.col("score").is_null())
    if missing_eligible.height:
        raise DataContractError(
            f"{missing_eligible.height} eligible candidate rows have no model score"
        )
    ineligible_scores = joined.filter(~pl.col("eligible") & pl.col("score").is_not_null())
    if ineligible_scores.height:
        raise DataContractError("ineligible candidate rows must not receive model scores")
    return validate_factor_frame(
        joined.select(FACTOR_COLUMNS).sort("asof_date", "security_id")
    )


def factor_universe_sha256(candidate_universe: pl.DataFrame) -> str:
    """Hash the canonical candidate identity/date/eligibility rows without buffering JSON."""

    columns = ["security_id", "symbol", "asof_date", "eligible"]
    if candidate_universe.columns != columns:
        raise DataContractError(f"candidate universe columns must exactly equal {columns}")
    canonical = candidate_universe.sort("asof_date", "security_id")
    if not candidate_universe.equals(canonical, null_equal=True):
        raise DataContractError("candidate universe must be canonically sorted before hashing")
    digest = hashlib.sha256()
    for security_id, symbol, asof_date, eligible in canonical.iter_rows():
        line = json.dumps(
            [security_id, symbol, asof_date.isoformat(), eligible],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def publish_factor_batch(
    frame: pl.DataFrame,
    output_root: str | Path,
    *,
    source: FactorBatchSource | Mapping[str, Any],
    model: FactorBatchModel | Mapping[str, Any],
    input_metadata: FactorBatchInput | Mapping[str, Any],
    time_metadata: FactorBatchTime | Mapping[str, Any],
    created_at: datetime | None = None,
    before_commit: Callable[[], None] | None = None,
    delivery_audit: Mapping[str, Any] | None = None,
) -> tuple[Path, FactorBatchManifest]:
    """Validate and atomically publish one immutable FactorBatch directory."""

    factors = validate_factor_frame(frame)
    source_value = source if isinstance(source, FactorBatchSource) else (
        FactorBatchSource.model_validate(source)
    )
    model_value = (
        model if isinstance(model, FactorBatchModel) else FactorBatchModel.model_validate(model)
    )
    input_value = (
        input_metadata
        if isinstance(input_metadata, FactorBatchInput)
        else FactorBatchInput.model_validate(input_metadata)
    )
    time_value = (
        time_metadata
        if isinstance(time_metadata, FactorBatchTime)
        else FactorBatchTime.model_validate(time_metadata)
    )
    observed_universe_hash = factor_universe_sha256(
        factors.select("security_id", "symbol", "asof_date", "eligible")
    )
    if input_value.universe_sha256 != observed_universe_hash:
        raise DataContractError("factor universe hash disagrees with input metadata")
    observed_minimum = cast(date, factors["asof_date"].min())
    observed_maximum = cast(date, factors["asof_date"].max())
    if (
        time_value.minimum_asof_date != observed_minimum
        or time_value.maximum_asof_date != observed_maximum
    ):
        raise DataContractError("factor date range disagrees with time metadata")

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".tmp-factor-batch-{uuid.uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        factors_path = temporary / "factors.parquet"
        factors.write_parquet(factors_path)
        written = validate_factor_frame(pl.read_parquet(factors_path))
        if not written.equals(factors, null_equal=True):
            raise DataContractError("written factor artifact differs from validated input")
        artifact = FactorBatchArtifact(
            sha256=sha256_file(factors_path),
            bytes=factors_path.stat().st_size,
            row_count=factors.height,
            date_count=factors["asof_date"].n_unique(),
        )
        eligible_rows = factors.filter(pl.col("eligible")).height
        coverage = FactorBatchCoverage(
            candidate_rows=factors.height,
            actual_rows=factors.height,
            expected_eligible_rows=eligible_rows,
            scored_eligible_rows=eligible_rows,
        )
        provisional = FactorBatchManifest(
            delivery_id="0" * 64,
            created_at=created_at or datetime.now(timezone.utc),
            source=source_value,
            model=model_value,
            input=input_value,
            time=time_value,
            coverage=coverage,
            artifact=artifact,
        )
        manifest = provisional.model_copy(
            update={"delivery_id": factor_batch_delivery_id(provisional)}
        )
        destination = root / manifest.delivery_id
        if delivery_audit is not None:
            # Operational receipt is outside the immutable two-file delivery.
            # Historical replay instead embeds the profile in its resolved request.
            audit_root = root.parent / f"{root.name}_delivery_audits"
            audit_root.mkdir(parents=True, exist_ok=True)
            receipt = {
                "delivery_id": manifest.delivery_id,
                "snapshot_id": input_value.snapshot_id,
                "release_id": model_value.release_id,
                **delivery_audit,
            }
            audit_temporary = audit_root / f".tmp-{uuid.uuid4().hex}.json"
            try:
                _write_json(audit_temporary, receipt)
                audit_temporary.replace(audit_root / f"{manifest.delivery_id}.json")
            finally:
                audit_temporary.unlink(missing_ok=True)
        if destination.exists():
            existing = load_factor_batch(destination)
            if existing.delivery_id != manifest.delivery_id:
                raise DataContractError("existing FactorBatch identity does not match")
            shutil.rmtree(temporary)
            return destination, existing
        _write_json(temporary / "manifest.json", manifest.model_dump(mode="json"))
        if before_commit is not None:
            before_commit()
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination, manifest


def factor_batch_metadata(
    release: ModelReleaseManifest,
    *,
    source_kind: Literal["evaluation_predictions", "signal_inference"],
) -> tuple[FactorBatchSource, FactorBatchModel]:
    """Derive cross-project lineage from a fully verified internal ModelRelease."""

    return (
        FactorBatchSource(
            kind=source_kind,
            repository=release.source.repository,
            commit=release.source.commit,
            run_id=release.source.run_id,
            run_manifest_sha256=release.source.run_manifest_sha256,
        ),
        FactorBatchModel(
            release_id=release.release_id,
            model_id=release.model_id,
            model_type=release.model_type,
            checkpoint_sha256=release.artifacts["checkpoint"].sha256,
            training_dataset_id=release.training_data.dataset_id,
            forecast_horizon_sessions=release.forecast_horizon_sessions,
            score_semantics="raw_cross_sectional_rank_score",
        ),
    )


def publish_evaluation_factor_batch(
    predictions_path: str | Path,
    release_dir: str | Path,
    output_root: str | Path,
    *,
    created_at: datetime | None = None,
    delivery: DeliveryConfig | None = None,
) -> tuple[Path, FactorBatchManifest]:
    """Convert verified research predictions into the sole external factor contract."""

    from facdigger.data.market_calendar import CALENDAR_VERSION
    from facdigger.evaluation.contracts import validate_predictions
    from facdigger.inference.releases import load_model_release

    prediction_path = Path(predictions_path).resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(f"predictions file does not exist: {prediction_path}")
    release_path = Path(release_dir).resolve()
    release = load_model_release(release_path)
    predictions = validate_predictions(pl.read_parquet(prediction_path))
    if predictions.is_empty():
        raise DataContractError("evaluation predictions must not be empty")
    if sha256_file(prediction_path) != release.source.predictions_sha256:
        raise DataContractError(
            "predictions file differs from the artifact bound by ModelRelease"
        )
    if not predictions["eligible"].all():
        raise DataContractError(
            "evaluation predictions may contain only eligible scored rows"
        )
    if predictions["model_id"].item(0) != release.model_id:
        raise DataContractError("predictions model_id differs from ModelRelease")
    if predictions["dataset_id"].item(0) != release.training_data.dataset_id:
        raise DataContractError("predictions dataset_id differs from ModelRelease")
    if predictions["checkpoint_hash"].n_unique() != 1 or (
        predictions["checkpoint_hash"].item(0)
        != release.artifacts["checkpoint"].sha256
    ):
        raise DataContractError("predictions checkpoint differs from ModelRelease")

    candidate_universe = predictions.select(
        "security_id", "symbol", "asof_date", "eligible"
    ).cast(
        {
            "security_id": pl.String,
            "symbol": pl.String,
            "asof_date": pl.Date,
            "eligible": pl.Boolean,
        }
    ).sort("asof_date", "security_id")
    scored_rows = predictions.select(
        "security_id",
        "asof_date",
        pl.col("score_raw").cast(pl.Float64).alias("score"),
    ).sort("asof_date", "security_id")
    build_factor_frame(candidate_universe, scored_rows)
    selection = resolve_delivery(
        candidate_universe, delivery, eligible_only=True
    )
    factors = build_factor_frame(selection.candidates, selection.project_scores(scored_rows))
    source_metadata, model_metadata = factor_batch_metadata(
        release, source_kind="evaluation_predictions"
    )
    input_metadata = FactorBatchInput(
        snapshot_id=release.training_data.dataset_id,
        snapshot_manifest_sha256=release.training_data.dataset_manifest_sha256,
        universe_semantics="eligible_scored_cross_section",
        universe_sha256=factor_universe_sha256(selection.candidates),
        identity_policy=delivery_identity_policy(selection.candidates),
    )
    time_metadata = FactorBatchTime(
        calendar_version=CALENDAR_VERSION,
        minimum_asof_date=cast(date, factors["asof_date"].min()),
        maximum_asof_date=cast(date, factors["asof_date"].max()),
    )
    return publish_factor_batch(
        factors,
        output_root,
        source=source_metadata,
        model=model_metadata,
        input_metadata=input_metadata,
        time_metadata=time_metadata,
        created_at=created_at,
        delivery_audit=selection.audit,
    )


def load_factor_batch(bundle_dir: str | Path) -> FactorBatchManifest:
    """Validate a published FactorBatch directory, manifest, artifact and identity."""

    root = Path(bundle_dir).resolve()
    manifest_path = root / "manifest.json"
    factor_path = root / "factors.parquet"
    if not manifest_path.is_file() or not factor_path.is_file():
        raise FileNotFoundError(f"FactorBatch is missing manifest or factors: {root}")
    entries = {path.name for path in root.iterdir()}
    if entries != {"manifest.json", "factors.parquet"}:
        raise DataContractError("FactorBatch directory must contain exactly two contract files")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataContractError("FactorBatch manifest is unreadable") from exc
    manifest = FactorBatchManifest.model_validate(payload)
    if root.name != manifest.delivery_id:
        raise DataContractError("FactorBatch directory name must equal delivery_id")
    if manifest.delivery_id != factor_batch_delivery_id(manifest):
        raise DataContractError("FactorBatch semantic identity does not match delivery_id")
    if sha256_file(factor_path) != manifest.artifact.sha256:
        raise DataContractError("FactorBatch factors hash does not match manifest")
    if factor_path.stat().st_size != manifest.artifact.bytes:
        raise DataContractError("FactorBatch factors byte size does not match manifest")
    factors = validate_factor_frame(pl.read_parquet(factor_path))
    if manifest.input.universe_sha256 != factor_universe_sha256(
        factors.select("security_id", "symbol", "asof_date", "eligible")
    ):
        raise DataContractError("FactorBatch universe hash does not match factors")
    if factors.height != manifest.artifact.row_count:
        raise DataContractError("FactorBatch row count does not match manifest")
    if factors["asof_date"].n_unique() != manifest.artifact.date_count:
        raise DataContractError("FactorBatch date count does not match manifest")
    eligible_rows = factors.filter(pl.col("eligible")).height
    if manifest.coverage != FactorBatchCoverage(
        candidate_rows=factors.height,
        actual_rows=factors.height,
        expected_eligible_rows=eligible_rows,
        scored_eligible_rows=eligible_rows,
    ):
        raise DataContractError("FactorBatch coverage does not match factors")
    if (
        factors["asof_date"].min() != manifest.time.minimum_asof_date
        or factors["asof_date"].max() != manifest.time.maximum_asof_date
    ):
        raise DataContractError("FactorBatch date range does not match manifest")
    return manifest
