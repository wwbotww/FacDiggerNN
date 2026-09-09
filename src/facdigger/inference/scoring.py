"""Shared fixed-release scoring for every supported factor model family."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from facdigger.data.contracts import DataContractError
from facdigger.inference.backends import CheckpointBackend, load_checkpoint_backend

if TYPE_CHECKING:
    from facdigger.inference.releases import ModelReleaseManifest

SCORING_INDEX_COLUMNS = [
    "sample_id",
    "security_id",
    "symbol",
    "asof_date",
    "feature_start",
    "feature_end",
    "eligible",
]
SCORE_SCHEMA = {
    "security_id": pl.String,
    "symbol": pl.String,
    "asof_date": pl.Date,
    "score": pl.Float64,
}


@dataclass
class FactorInferenceRuntime:
    """A fixed verified release and its lazily loaded model-family adapter."""

    release: ModelReleaseManifest
    backend: CheckpointBackend


def load_factor_inference_runtime(
    release_dir: str | Path,
    *,
    device: Literal["auto", "cpu", "cuda"] = "cpu",
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> FactorInferenceRuntime:
    from facdigger.inference.releases import release_runtime

    release, config, checkpoint = release_runtime(release_dir)
    backend = load_checkpoint_backend(
        release.model_type,
        config_payload=config,
        checkpoint_path=checkpoint,
        training_dataset_id=release.training_data.dataset_id,
        context_length=release.feature_contract.context_length,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if backend.config.channels != release.feature_contract.channels:
        raise DataContractError("release channels differ from resolved model configuration")
    return FactorInferenceRuntime(release=release, backend=backend)


def validate_scoring_rows(rows: pl.DataFrame) -> pl.DataFrame:
    missing = sorted(set(SCORING_INDEX_COLUMNS) - set(rows.columns))
    if missing:
        raise DataContractError(f"inference rows are missing required fields: {missing}")
    if any(c.startswith("target") or c in {"label", "split"} for c in rows.columns):
        raise DataContractError("scoring input must not contain labels, targets or split")
    canonical = rows.select(SCORING_INDEX_COLUMNS).sort("asof_date", "security_id")
    if canonical.null_count().sum_horizontal().item():
        raise DataContractError("scoring identity and window fields must be non-null")
    if not canonical["eligible"].all():
        raise DataContractError("scoring rows must all be eligible")
    if canonical["sample_id"].n_unique() != canonical.height:
        raise DataContractError("inference rows contain duplicate sample IDs")
    if canonical.select("security_id", "asof_date").n_unique() != canonical.height:
        raise DataContractError("inference rows contain duplicate security/date keys")
    return canonical


def score_inference_rows(
    runtime: FactorInferenceRuntime,
    *,
    snapshot_dir: str | Path,
    snapshot_manifest: dict[str, Any],
    rows: pl.DataFrame,
) -> pl.DataFrame:
    """Compute the model universe before projecting a requested delivery subset."""
    if rows.is_empty():
        return pl.DataFrame(schema=SCORE_SCHEMA)
    requested = validate_scoring_rows(rows)
    snapshot_path = Path(snapshot_dir).resolve()
    index_path = snapshot_path / snapshot_manifest["artifacts"]["inference_index"]
    dates = requested["asof_date"].unique().to_list()
    computational = validate_scoring_rows(
        pl.scan_parquet(index_path).filter(pl.col("asof_date").is_in(dates)).collect()
    )
    matching = requested.join(computational, on=SCORING_INDEX_COLUMNS, how="anti")
    if matching.height:
        raise DataContractError("requested rows differ from the snapshot scoring universe")
    feature_config = {
        "channels": runtime.release.feature_contract.channels,
        "market_channels": getattr(runtime.release.feature_contract, "market_channels", []),
    }
    scored = runtime.backend.predict(
        snapshot_path,
        {"config": {"features": feature_config}, "artifacts": snapshot_manifest["artifacts"]},
        computational,
    )
    if (
        scored.schema != SCORE_SCHEMA
        or scored.select("security_id", "asof_date").n_unique() != scored.height
    ):
        raise DataContractError("model backend returned a non-canonical score table")
    expected = computational.select("security_id", "symbol", "asof_date")
    if not scored.select(expected.columns).equals(expected):
        raise DataContractError("model backend did not score the complete computational universe")
    if scored["score"].null_count() or not scored["score"].is_finite().all():
        raise DataContractError("model backend returned missing or non-finite scores")
    return scored.join(
        requested.select("security_id", "asof_date"),
        on=["security_id", "asof_date"],
        how="semi",
    ).sort("asof_date", "security_id")
