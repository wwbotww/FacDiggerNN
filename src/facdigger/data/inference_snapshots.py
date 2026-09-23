"""Target-free inference snapshots bound to one immutable ModelRelease."""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from facdigger.data.adapters import StandardParquetAdapter
from facdigger.data.config import InferenceSnapshotConfig
from facdigger.data.contracts import DataContractError
from facdigger.data.paths import artifact_path
from facdigger.data.snapshots import sha256_file
from facdigger.datasets.index import build_inference_index
from facdigger.experiments.manifest import sha256_json
from facdigger.features.pipeline import (
    apply_feature_scaler,
    build_raw_feature_tables,
    validate_feature_scaler,
)
from facdigger.inference.releases import (
    ModelReleaseManifest,
    load_model_release,
)

INFERENCE_SNAPSHOT_CONTRACT = "facdigger.inference_snapshot"
INFERENCE_SNAPSHOT_ARTIFACTS = {
    "features",
    "inference_index",
    "delivery_universe",
    "audit",
    "scaler",
    "source_manifest",
}


def _source_hashes(config: InferenceSnapshotConfig) -> dict[str, str | None]:
    paths = {
        "bars": config.sources.bars,
        "universe": config.sources.universe,
        "corporate_actions": config.sources.corporate_actions,
        "delistings": config.sources.delistings,
        "source_manifest": config.sources.source_manifest,
    }
    missing = [name for name, path in paths.items() if path is not None and not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Configured inference source files do not exist: {missing}")
    return {name: sha256_file(path) if path is not None else None for name, path in paths.items()}


def _feature_audit(features: pl.DataFrame) -> dict[str, Any]:
    observed = [column for column in features.columns if column.startswith("observed_")]
    return {
        "rows": features.height,
        "observed_ratio": {
            column.removeprefix("observed_"): float(features[column].mean() or 0.0)
            for column in observed
        },
    }


def _validate_snapshot_files(
    snapshot_dir: Path,
    release: ModelReleaseManifest,
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Validate one immutable inference snapshot without loading its feature table."""

    snapshot_dir = snapshot_dir.resolve()
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.is_file():
        raise DataContractError("existing inference snapshot has no manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataContractError("inference snapshot manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise DataContractError("inference snapshot manifest must be a mapping")
    if manifest.get("contract") != INFERENCE_SNAPSHOT_CONTRACT:
        raise DataContractError("inference snapshot contract is unsupported")
    if manifest.get("status") != "complete":
        raise DataContractError("inference snapshot is not complete")
    identity = {
        key: manifest.get(key)
        for key in ("contract", "config", "feature_contract", "input_file_hashes")
    }
    snapshot_id = sha256_json(identity)
    if manifest.get("snapshot_id") != snapshot_id or snapshot_dir.name != snapshot_id:
        raise DataContractError("inference snapshot semantic identity does not match")
    if expected_identity is not None and any(
        identity.get(key) != value for key, value in expected_identity.items()
    ):
        raise DataContractError("existing inference snapshot identity does not match")
    expected_feature_contract = {
        "release_id": release.release_id,
        **release.feature_contract.model_dump(mode="json"),
    }
    if manifest.get("feature_contract") != expected_feature_contract:
        raise DataContractError("inference snapshot feature contract differs from ModelRelease")
    artifacts = manifest.get("artifacts") or {}
    artifact_hashes = manifest.get("artifact_hashes") or {}
    expected_artifacts = INFERENCE_SNAPSHOT_ARTIFACTS | (
        {"market_features"}
        if release.feature_contract.feature_set == "finance_transformer"
        else set()
    )
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise DataContractError("inference snapshot artifact declarations are invalid")
    required_artifacts = expected_artifacts - {"source_manifest"}
    if any(not isinstance(artifacts[name], str) for name in required_artifacts):
        raise DataContractError("required inference snapshot artifacts must name files")
    if artifacts["source_manifest"] is not None and not isinstance(
        artifacts["source_manifest"], str
    ):
        raise DataContractError("inference source manifest artifact must name a file or be null")
    expected_hashed = {name for name, relative in artifacts.items() if relative is not None}
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != expected_hashed:
        raise DataContractError("inference snapshot artifact hash declarations are incomplete")
    paths: dict[str, Path] = {}
    for name, expected_hash in artifact_hashes.items():
        relative = artifacts.get(name)
        if not isinstance(relative, str):
            raise DataContractError(f"inference snapshot artifact is undeclared: {name}")
        path = artifact_path(snapshot_dir, relative, name, require_file=False)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise DataContractError(f"inference snapshot artifact integrity failure: {name}")
        paths[name] = path
    expected_files = {"manifest.json"} | {
        artifact_path(snapshot_dir, relative, "inference artifact").relative_to(
            snapshot_dir
        ).as_posix()
        for relative in artifacts.values() if relative is not None
    }
    actual_entries = {
        path.relative_to(snapshot_dir).as_posix() for path in snapshot_dir.rglob("*")
    }
    if actual_entries != expected_files:
        raise DataContractError("inference snapshot contains undeclared or missing entries")
    if sha256_file(paths["scaler"]) != release.feature_contract.scaler_sha256:
        raise DataContractError("inference snapshot scaler differs from ModelRelease")
    return manifest, paths


def _validated_inference_tables(
    inference_index: pl.DataFrame,
    delivery_universe: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if any(c.startswith("target") or c in {"label", "split"} for c in inference_index.columns):
        raise DataContractError("inference index must not contain target or split")
    required_index = {
        "sample_id",
        "security_id",
        "symbol",
        "asof_date",
        "feature_start",
        "feature_end",
        "eligible",
    }
    missing = sorted(required_index - set(inference_index.columns))
    if missing:
        raise DataContractError(f"inference index is missing required columns: {missing}")
    expected_index_schema = {
        "sample_id": pl.String,
        "security_id": pl.String,
        "symbol": pl.String,
        "asof_date": pl.Date,
        "feature_start": pl.Date,
        "feature_end": pl.Date,
        "eligible": pl.Boolean,
    }
    if any(
        inference_index.schema[column] != expected_type
        for column, expected_type in expected_index_schema.items()
    ):
        raise DataContractError("inference index required field schema is not canonical")
    if inference_index.null_count().select(sorted(required_index)).sum_horizontal().item():
        raise DataContractError("inference index required fields must be non-null")
    if not inference_index["eligible"].all():
        raise DataContractError("inference index may contain only model-scorable eligible rows")
    if inference_index["sample_id"].n_unique() != inference_index.height:
        raise DataContractError("inference index contains duplicate sample IDs")
    expected_sample_ids = inference_index.select(
        pl.concat_str(
            "security_id",
            pl.col("asof_date").dt.strftime("%Y-%m-%d"),
            separator="|",
        ).alias("sample_id")
    )["sample_id"]
    if not inference_index["sample_id"].equals(expected_sample_ids):
        raise DataContractError("inference index sample IDs do not match identity/date keys")
    if inference_index.filter(pl.col("feature_end") != pl.col("asof_date")).height:
        raise DataContractError("inference index contains invalid feature window bounds")
    if inference_index.filter(pl.col("feature_start") > pl.col("feature_end")).height:
        raise DataContractError("inference index contains reversed feature window bounds")
    if delivery_universe.columns != ["security_id", "symbol", "asof_date", "eligible"]:
        raise DataContractError("delivery universe columns are not canonical")
    expected_schema = {
        "security_id": pl.String,
        "symbol": pl.String,
        "asof_date": pl.Date,
        "eligible": pl.Boolean,
    }
    if delivery_universe.schema != expected_schema:
        raise DataContractError("delivery universe schema is not canonical")
    if delivery_universe.null_count().sum_horizontal().item():
        raise DataContractError("delivery universe fields must be non-null")
    canonical_universe = delivery_universe.sort("asof_date", "security_id")
    if not delivery_universe.equals(canonical_universe, null_equal=True):
        raise DataContractError("delivery universe must be canonically sorted")
    if delivery_universe.select("security_id", "asof_date").n_unique() != (
        delivery_universe.height
    ):
        raise DataContractError("delivery universe contains duplicate identity/date keys")
    canonical_index = inference_index.sort("asof_date", "security_id")
    if not inference_index.equals(canonical_index, null_equal=True):
        raise DataContractError("inference index must be canonically sorted")
    if inference_index.select("security_id", "asof_date").n_unique() != inference_index.height:
        raise DataContractError("inference index contains duplicate identity/date keys")
    eligible_candidates = delivery_universe.filter(pl.col("eligible")).select(
        "security_id", "symbol", "asof_date"
    )
    scored_candidates = inference_index.select("security_id", "symbol", "asof_date")
    if not eligible_candidates.equals(scored_candidates, null_equal=True):
        raise DataContractError("delivery eligibility does not exactly match the inference index")
    return inference_index, delivery_universe


def load_inference_snapshot(
    snapshot_dir: str | Path,
    release: ModelReleaseManifest,
) -> tuple[dict[str, Any], dict[str, pl.DataFrame]]:
    """Verify an inference snapshot and load only its lightweight scoring indexes."""

    root = Path(snapshot_dir).resolve()
    manifest, paths = _validate_snapshot_files(root, release)
    inference_index, delivery_universe = _validated_inference_tables(
        pl.read_parquet(paths["inference_index"]),
        pl.read_parquet(paths["delivery_universe"]),
    )
    if not inference_index.is_empty():
        # Both supported feature contracts observe range only from a real OHLC
        # bar on that date. Check cached snapshots too, not only new builds.
        observed_bars = (
            pl.scan_parquet(paths["features"]).filter(pl.col("observed_range"))
            .select("security_id", pl.col("trade_date").alias("asof_date"))
        )
        absent = inference_index.select("security_id", "asof_date").lazy().join(
            observed_bars, on=["security_id", "asof_date"], how="anti",
        ).limit(1).collect()
        if absent.height:
            raise DataContractError("inference index contains an unobserved as-of bar")
    return manifest, {
        "inference_index": inference_index,
        "delivery_universe": delivery_universe,
    }


def _release_scaler(
    release_dir: Path,
) -> tuple[ModelReleaseManifest, dict[str, Any], str]:
    release = load_model_release(release_dir)
    if release.feature_contract.scaler_contract != "train_global_robust":
        raise DataContractError("inference builder does not support the release scaler contract")
    scaler_artifact = release.artifacts["scaler"]
    scaler_path = release_dir / scaler_artifact.file
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    if not isinstance(scaler, Mapping):
        raise DataContractError("release scaler must be a JSON mapping")
    validate_feature_scaler(
        scaler,
        feature_set=release.feature_contract.feature_set,
        channels=release.feature_contract.channels,
        market_channels=getattr(release.feature_contract, "market_channels", []),
    )
    return release, dict(scaler), scaler_artifact.sha256


def _require_source_provenance(
    config: InferenceSnapshotConfig, release_dir: Path, release: ModelReleaseManifest,
) -> None:
    """Preserve source-proof requirements independently of consumer identity policy."""
    training_manifest = json.loads(artifact_path(
        release_dir, release.artifacts["training_dataset_manifest"].file,
        "release training dataset manifest",
    ).read_text(encoding="utf-8"))
    if (training_manifest.get("artifacts") or {}).get("source_manifest") is not None:
        if config.sources.source_manifest is None:
            raise DataContractError(
                "inference requires source provenance because the training snapshot used it"
            )


def _delivery_universe(
    universe: pl.DataFrame,
    inference_index: pl.DataFrame,
) -> pl.DataFrame:
    scorable = inference_index.select("security_id", "asof_date").with_columns(
        pl.lit(True).alias("_has_model_window")
    )
    return (
        universe.select(
            "security_id",
            "symbol",
            pl.col("trade_date").alias("asof_date"),
            pl.col("eligible").alias("_source_eligible"),
        )
        .join(scorable, on=["security_id", "asof_date"], how="left")
        .with_columns(
            (pl.col("_source_eligible") & pl.col("_has_model_window").fill_null(False)).alias(
                "eligible"
            )
        )
        .select("security_id", "symbol", "asof_date", "eligible")
        .sort(["asof_date", "security_id"])
    )


def describe_unscorable(
    universe: pl.DataFrame, bars: pl.DataFrame, candidates: pl.DataFrame,
) -> list[dict[str, Any]]:
    """Explain input eligibility; never turn a failed model score into missing data."""
    missing = candidates.filter(~pl.col("eligible"))
    if missing.is_empty():
        return []
    dates = missing["asof_date"].unique()
    metadata = universe.filter(pl.col("trade_date").is_in(dates.implode())).select(
        "security_id", pl.col("trade_date").alias("asof_date"),
        pl.col("eligible").alias("_source_eligible"),
        (
            pl.col("is_delisted") if "is_delisted" in universe.columns else pl.lit(False)
        ).alias("_delisted"),
        (
            pl.col("adv20_usd").is_null() if "adv20_usd" in universe.columns else pl.lit(False)
        ).alias("_missing_liquidity_history"),
        (
            pl.col("trade_status_quality") == "source_quality_quarantined"
            if "trade_status_quality" in universe.columns else pl.lit(False)
        ).alias("_quarantined"),
    )
    observed = bars.filter(pl.col("trade_date").is_in(dates.implode())).select(
        "security_id", pl.col("trade_date").alias("asof_date"),
        pl.lit(True).alias("_observed_bar"),
    )
    reasons = (
        missing.join(metadata, on=["security_id", "asof_date"], validate="1:1")
        .join(observed, on=["security_id", "asof_date"], how="left", validate="1:1")
        .with_columns(
            pl.when(pl.col("_quarantined")).then(pl.lit("source_quality_quarantined"))
            .when(pl.col("_delisted")).then(pl.lit("not_active"))
            .when(pl.col("_observed_bar").is_null()).then(pl.lit("missing_target_bar"))
            .when(pl.col("_source_eligible")).then(pl.lit("insufficient_model_history"))
            .when(pl.col("_missing_liquidity_history"))
            .then(pl.lit("insufficient_liquidity_history"))
            .otherwise(pl.lit("outside_model_universe")).alias("reason"),
            pl.col("asof_date").dt.strftime("%Y-%m-%d"),
        )
        .select("security_id", "symbol", "asof_date", "reason")
        .sort("asof_date", "security_id")
    )
    return reasons.to_dicts()


def build_inference_snapshot(
    config: InferenceSnapshotConfig,
    release_dir: str | Path,
    *,
    asof_date: date | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Build a target-free snapshot using the scaler frozen in one ModelRelease."""

    release_path = Path(release_dir).resolve()
    release, scaler, scaler_hash = _release_scaler(release_path)
    _require_source_provenance(config, release_path, release)
    feature_contract = release.feature_contract.model_dump(mode="json")
    adapter = StandardParquetAdapter(config.sources)
    semantic_config = config.model_dump(mode="json")
    source_paths = semantic_config.pop("sources")
    semantic_config.pop("output_root")
    if asof_date is not None:
        semantic_config["asof_date"] = asof_date.isoformat()
    identity = {
        "contract": INFERENCE_SNAPSHOT_CONTRACT,
        "config": semantic_config,
        "feature_contract": {"release_id": release.release_id, **feature_contract},
        "input_file_hashes": _source_hashes(config),
    }
    snapshot_id = sha256_json(identity)
    base_output_root = config.output_root.resolve()
    output_root = (
        base_output_root / asof_date.isoformat() if asof_date is not None else base_output_root
    )
    final_dir = output_root / snapshot_id
    if final_dir.exists():
        manifest, _ = _validate_snapshot_files(
            final_dir,
            release,
            expected_identity=identity,
        )
        return final_dir, manifest

    bundle = adapter.load()
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".tmp-{snapshot_id[:12]}-{uuid.uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        source_audit = adapter.audit(bundle)
        universe = bundle.universe
        feature_bars = bundle.bars
        latest_source_date = asof_date or universe["trade_date"].max()
        audit_bars = bundle.bars.filter(pl.col("trade_date") == latest_source_date).select(
            "security_id", "trade_date",
        )
        feature_universe = universe
        if asof_date is not None and release.feature_contract.feature_set == "price_volume_v1":
            context_length = int(feature_contract["context_length"])
            session_dates = sorted(feature_bars["trade_date"].unique().to_list())
            eligible_dates = [day for day in session_dates if day <= asof_date]
            if len(eligible_dates) < context_length:
                raise DataContractError(
                    "inference source has insufficient history for the release context"
                )
            # Twenty extra sessions preserve all rolling feature warm-up before
            # the first window observation while avoiding a full-history rebuild.
            start_index = max(0, len(eligible_dates) - context_length - 20)
            feature_start = eligible_dates[start_index]
            feature_bars = feature_bars.filter(
                pl.col("trade_date").is_between(feature_start, asof_date)
            )
            target_universe = universe.filter(pl.col("trade_date") == asof_date)
            if target_universe.is_empty():
                raise DataContractError(
                    f"inference source has no target session {asof_date.isoformat()}"
                )
            scorable_ids = (
                target_universe.filter(pl.col("eligible"))["security_id"].unique().to_list()
            )
            feature_bars = feature_bars.filter(
                pl.col("trade_date").is_between(feature_start, asof_date)
                & pl.col("security_id").is_in(scorable_ids)
            )
            first_observations = feature_bars.group_by("security_id").agg(
                pl.col("trade_date").min().alias("_first_observation")
            )
            identities = target_universe.filter(pl.col("security_id").is_in(scorable_ids)).select(
                "security_id",
                "symbol",
                "eligible",
                "industry_code",
                "float_market_cap",
            )
            feature_calendar = pl.DataFrame(
                {"trade_date": [day for day in eligible_dates if day >= feature_start]},
                schema={"trade_date": pl.Date},
            )
            feature_universe = (
                identities.join(feature_calendar, how="cross")
                .join(first_observations, on="security_id", how="left", validate="m:1")
                .filter(pl.col("trade_date") >= pl.col("_first_observation"))
                .drop("_first_observation")
            )
        elif asof_date is not None:
            # Cross-sectional features require actual membership on every past date,
            # including stocks which are no longer eligible on the target date.
            dates = (
                universe["trade_date"]
                .filter(universe["trade_date"] <= asof_date)
                .unique()
                .sort()
                .to_list()
            )
            history_sessions = int(feature_contract["context_length"]) + 20
            if len(dates) < history_sessions:
                raise DataContractError("inference source has insufficient market history")
            feature_start = dates[-history_sessions]
            target_universe = universe.filter(pl.col("trade_date") == asof_date)
            if target_universe.is_empty():
                raise DataContractError("inference source has no target session")
            feature_universe = universe.filter(
                pl.col("trade_date").is_between(feature_start, asof_date)
            )
            feature_bars = feature_bars.filter(
                pl.col("trade_date").is_between(feature_start, asof_date)
            )
        raw_features, raw_market = build_raw_feature_tables(
            feature_bars, feature_universe, feature_set=release.feature_contract.feature_set
        )
        features, market_features = apply_feature_scaler(raw_features, raw_market, scaler)
        del raw_features, raw_market, bundle
        inference_index = build_inference_index(
            features,
            feature_universe,
            int(feature_contract["context_length"]),
        )
        # A source eligibility flag cannot manufacture a target-day observation.
        # Keep masked historical holes, but never score a stock with no actual D bar.
        inference_index = inference_index.join(
            feature_bars.select("security_id", pl.col("trade_date").alias("asof_date")),
            on=["security_id", "asof_date"], how="semi",
        ).sort("asof_date", "security_id")
        if market_features is not None:
            context_length = int(feature_contract["context_length"])
            if market_features.height < context_length:
                raise DataContractError("inference source has insufficient market context")
            first_scorable_date = market_features["trade_date"].sort()[context_length - 1]
            inference_index = inference_index.filter(pl.col("asof_date") >= first_scorable_date)
        if asof_date is not None:
            inference_index = inference_index.filter(pl.col("asof_date") == asof_date)
        maximum_date = inference_index["asof_date"].max()
        if maximum_date is None and asof_date is None:
            raise DataContractError("inference snapshot contains no eligible feature windows")
        # An exact-day snapshot may truthfully contain zero scorable windows.
        # Operational gates decide whether to wait; never fabricate a model score.
        audit_date = asof_date or universe["trade_date"].max()
        assert audit_date is not None
        delivery_source = target_universe if asof_date is not None else universe
        delivery_universe = _delivery_universe(delivery_source, inference_index)
        artifacts = {
            "features": "features.parquet",
            "inference_index": "inference_index.parquet",
            "delivery_universe": "delivery_universe.parquet",
            "audit": "audit.json",
            "scaler": "scaler.json",
            "source_manifest": (
                "source_manifest.json" if config.sources.source_manifest is not None else None
            ),
        }
        if market_features is not None:
            artifacts["market_features"] = "market_features.parquet"
            market_features.write_parquet(temporary / artifacts["market_features"])
        features.write_parquet(temporary / artifacts["features"])
        inference_index.write_parquet(temporary / artifacts["inference_index"])
        delivery_universe.write_parquet(temporary / artifacts["delivery_universe"])
        audit = {
            "sources": source_audit,
            "features": _feature_audit(features),
            "inference_index": {
                "rows": inference_index.height,
                "minimum_asof_date": (
                    inference_index["asof_date"].min().isoformat()
                    if inference_index.height else None
                ),
                "maximum_asof_date": maximum_date.isoformat() if maximum_date else None,
                "latest_cross_section_rows": inference_index.filter(
                    pl.col("asof_date") == audit_date
                ).height,
                "contains_target": False,
            },
            "delivery_universe": {
                "rows": delivery_universe.height,
                "latest_cross_section_rows": delivery_universe.filter(
                    pl.col("asof_date") == audit_date
                ).height,
                "latest_unscorable": describe_unscorable(
                    delivery_source, audit_bars,
                    delivery_universe.filter(pl.col("asof_date") == audit_date),
                ),
            },
            "scaler": {"origin": "model_release", "sha256": scaler_hash},
        }
        if market_features is not None:
            audit["market_features"] = _feature_audit(market_features)
        (temporary / artifacts["audit"]).write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copyfile(
            release_path / release.artifacts["scaler"].file, temporary / artifacts["scaler"]
        )
        if config.sources.source_manifest is not None:
            shutil.copyfile(config.sources.source_manifest, temporary / "source_manifest.json")
        manifest = {
            **identity,
            "status": "complete",
            "snapshot_id": snapshot_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_paths": source_paths,
            "artifacts": artifacts,
            "artifact_hashes": {
                name: sha256_file(temporary / relative)
                for name, relative in artifacts.items()
                if relative is not None
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(final_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final_dir, manifest
