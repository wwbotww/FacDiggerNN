"""Unified E0-E3 checkpoint loading, replay verification and signal inference."""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import yaml

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.datasets.window import SnapshotWindowDataset
from facdigger.environment import collect_environment
from facdigger.evaluation.contracts import prediction_coverage
from facdigger.evaluation.metrics import evaluate_predictions
from facdigger.evaluation.report import write_evaluation_report
from facdigger.experiments.manifest import sha256_json
from facdigger.inference.scoring import (
    build_patchtst_model,
    load_e3_inference_runtime,
    patch_config,
    score_e3_inference_rows,
)
from facdigger.models.baselines import (
    TabularPreprocessor,
    build_multiscale_features,
    load_mlp_checkpoint,
    predict_lightgbm_checkpoint,
    predict_mlp,
)
from facdigger.training.common import (
    apply_source_readiness_gate,
    build_prediction_frame,
    load_source_provenance,
    load_training_snapshot,
)
from facdigger.training.e0_config import E0ExperimentConfig
from facdigger.training.e1_engine import predict_e1, select_device

InferenceSplit = Literal["train", "valid", "test"]
SUPPORTED_MODEL_TYPES = {
    "mlp",
    "lightgbm",
    "random_patchtst",
    "etth1_transferred_patchtst",
    "financial_pretrained_patchtst",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _resolve_inside(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise DataContractError(f"{label} escapes source run directory") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _load_source_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    root = run_dir.resolve()
    manifest_path = root / "manifest.json"
    config_path = root / "resolved_config.yaml"
    if not manifest_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"run is missing manifest or resolved config: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status", "complete") != "complete":
        raise ValueError("inference requires a complete source run")
    model_type = str(manifest.get("model_type"))
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"unsupported source run model_type: {model_type}")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise ValueError("source resolved configuration must be a mapping")
    if sha256_json(config_payload) != manifest["config_hash"]:
        raise DataContractError("source resolved configuration hash does not match manifest")
    checkpoint_info = manifest.get("checkpoint") or {}
    checkpoint_path = _resolve_inside(root, str(checkpoint_info["file"]), "checkpoint")
    if sha256_file(checkpoint_path) != checkpoint_info["sha256"]:
        raise DataContractError("source checkpoint hash does not match manifest")
    return manifest, config_payload, checkpoint_path


def _validate_dataset(
    manifest: dict[str, Any], dataset_path: Path
) -> tuple[dict[str, Any], dict[str, pl.DataFrame]]:
    dataset_manifest, frames = load_training_snapshot(dataset_path)
    if dataset_manifest["dataset_id"] != manifest["dataset_id"]:
        raise DataContractError("inference dataset_id does not match source run")
    expected_hash = manifest["dataset_manifest_hash"]
    if sha256_file(dataset_path / "manifest.json") != expected_hash:
        raise DataContractError("inference dataset manifest hash does not match source run")
    return dataset_manifest, frames


def _load_preprocessor(
    run_dir: Path, manifest: dict[str, Any], checkpoint_path: Path
) -> tuple[TabularPreprocessor, dict[str, Any]]:
    declared = (manifest.get("checkpoint") or {}).get("preprocessing")
    if declared:
        path = _resolve_inside(run_dir, str(declared["file"]), "preprocessing checkpoint")
        if sha256_file(path) != declared["sha256"]:
            raise DataContractError("preprocessing checkpoint hash does not match manifest")
    else:
        path = checkpoint_path.with_suffix(".preprocessing.json")
        if not path.is_file():
            raise FileNotFoundError(f"LightGBM preprocessing checkpoint is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TabularPreprocessor.from_dict(payload), {
        "file": str(path),
        "sha256": sha256_file(path),
        "declared_in_source_manifest": declared is not None,
    }


def _predict_e0(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    config_payload: dict[str, Any],
    checkpoint_path: Path,
    frames: dict[str, pl.DataFrame],
    context_length: int,
    split: InferenceSplit,
    device_preference: str,
) -> tuple[np.ndarray, pl.DataFrame, dict[str, Any]]:
    config = E0ExperimentConfig.model_validate(config_payload)
    tabular, feature_columns = build_multiscale_features(
        frames["features"],
        frames["sample_index"],
        channels=config.channels,
        windows=config.windows,
        context_length=context_length,
    )
    rows = tabular.filter(pl.col("split") == split)
    if rows.is_empty():
        raise DataContractError(f"dataset contains no inference rows for split={split}")
    if feature_columns != list(manifest["feature_columns"]):
        raise DataContractError("reconstructed E0 feature columns differ from source run")
    if manifest["model_type"] == "mlp":
        device = select_device(device_preference)
        model, preprocessor, payload = load_mlp_checkpoint(
            checkpoint_path, device=device
        )
        scores = predict_mlp(model, preprocessor.transform(rows), device)
        audit = {
            "loader": "embedded_mlp_checkpoint",
            "device": device,
            "preprocessing": "embedded",
            "checkpoint_seed": payload.get("seed"),
        }
    else:
        preprocessor, preprocessing_audit = _load_preprocessor(
            run_dir, manifest, checkpoint_path
        )
        scores = predict_lightgbm_checkpoint(checkpoint_path, preprocessor.transform(rows))
        audit = {
            "loader": "isolated_lightgbm_checkpoint",
            "device": "cpu",
            "preprocessing": preprocessing_audit,
        }
    if preprocessor.feature_columns != feature_columns:
        raise DataContractError("E0 preprocessing feature columns differ from reconstruction")
    return scores, rows, audit


def _predict_patchtst(
    *,
    manifest: dict[str, Any],
    config_payload: dict[str, Any],
    checkpoint_path: Path,
    frames: dict[str, pl.DataFrame],
    context_length: int,
    split: InferenceSplit,
    device_preference: str,
) -> tuple[np.ndarray, pl.DataFrame, dict[str, Any]]:
    import torch

    config, inference_config = patch_config(manifest["model_type"], config_payload)
    dataset = SnapshotWindowDataset(
        features=frames["features"],
        sample_index=frames["sample_index"],
        channels=config.channels,
        context_length=context_length,
        split=split,
    )
    device = select_device(device_preference)
    model = build_patchtst_model(
        config.model,
        context_length=context_length,
        num_channels=len(config.channels),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_dataset_id = checkpoint.get("dataset_id")
    if checkpoint_dataset_id != manifest["dataset_id"]:
        raise DataContractError("PatchTST checkpoint dataset_id does not match source run")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    precision = inference_config.precision if device == "cuda" else "fp32"
    scores = predict_e1(
        model,
        dataset,
        batch_size=inference_config.batch_size,
        device=device,
        precision=precision,
        num_workers=inference_config.num_workers,
    )
    return scores, dataset.sample_rows, {
        "loader": "strict_patchtst_state_dict",
        "device": device,
        "precision": precision,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
    }


def _verify_replay(
    run_dir: Path,
    manifest: dict[str, Any],
    split: InferenceSplit,
    predictions: pl.DataFrame,
    *,
    require_match: bool,
) -> dict[str, Any]:
    original_path = run_dir / "predictions.parquet"
    expected = split == manifest["evaluation_split"]
    if not expected:
        return {
            "applicable": False,
            "reason": "requested split differs from source run evaluation split",
        }
    if not original_path.is_file():
        if require_match:
            raise FileNotFoundError("source run predictions are missing for replay verification")
        return {"applicable": True, "available": False, "matched": None}
    keys = ["security_id", "asof_date", "target"]
    original = pl.read_parquet(original_path).sort(["asof_date", "security_id"])
    replayed = predictions.sort(["asof_date", "security_id"])
    if not original.select(keys).equals(replayed.select(keys), null_equal=True):
        raise DataContractError("replayed prediction keys or targets differ from source run")
    left = original["score_raw"].to_numpy().astype(np.float64)
    right = replayed["score_raw"].to_numpy().astype(np.float64)
    difference = np.abs(left - right)
    matched = bool(np.allclose(left, right, rtol=1e-6, atol=1e-8, equal_nan=False))
    audit = {
        "applicable": True,
        "available": True,
        "matched": matched,
        "rows": len(left),
        "maximum_absolute_difference": float(difference.max()) if len(difference) else 0.0,
        "mean_absolute_difference": float(difference.mean()) if len(difference) else 0.0,
        "rtol": 1e-6,
        "atol": 1e-8,
        "source_predictions_sha256": sha256_file(original_path),
    }
    if require_match and not matched:
        raise RuntimeError(
            "checkpoint replay scores differ from source predictions: "
            f"max_abs={audit['maximum_absolute_difference']}"
        )
    return audit


def _select_signal_inputs(
    index: pl.DataFrame,
    delivery_universe: pl.DataFrame,
    *,
    asof: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Select one declared candidate date without silently falling back to stale scores."""

    selected_date = (
        delivery_universe["asof_date"].max()
        if asof == "latest"
        else date.fromisoformat(asof)
    )
    candidates = delivery_universe.filter(
        pl.col("asof_date") == selected_date
    ).sort("security_id")
    if candidates.is_empty():
        raise DataContractError("requested signal date is absent from the delivery universe")
    rows = index.filter(pl.col("asof_date") == selected_date).sort("security_id")
    return rows, candidates


def run_signal_inference(
    release_dir: str | Path,
    *,
    output_root: str | Path = "artifacts/factor_batches",
    dataset_dir: str | Path,
    asof: str = "latest",
    device: Literal["auto", "cpu", "cuda"] = "cpu",
    before_publish: Callable[[], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Score one target-free inference snapshot and publish a FactorBatch."""

    from facdigger.data.inference_snapshots import load_inference_snapshot
    from facdigger.data.providers.eodhd.market_calendar import CALENDAR_VERSION
    from facdigger.inference.factor_batch import (
        FactorBatchInput,
        FactorBatchTime,
        build_factor_frame,
        factor_batch_metadata,
        factor_universe_sha256,
        publish_factor_batch,
    )
    runtime = load_e3_inference_runtime(release_dir, device=device)
    release = runtime.release
    dataset_path = Path(dataset_dir).resolve()
    dataset_manifest, frames = load_inference_snapshot(dataset_path, release)
    inference_index = frames["inference_index"]
    rows, candidate_universe = _select_signal_inputs(
        inference_index,
        frames["delivery_universe"],
        asof=asof,
    )
    if rows.is_empty():
        scored_rows = pl.DataFrame(
            schema={
                "security_id": pl.String,
                "asof_date": pl.Date,
                "score": pl.Float64,
            }
        )
    else:
        scored_rows = score_e3_inference_rows(
            runtime,
            snapshot_dir=dataset_path,
            snapshot_manifest=dataset_manifest,
            rows=rows,
        ).select("security_id", "asof_date", "score")
    candidate_universe = candidate_universe.select(
        "security_id", "symbol", "asof_date", "eligible"
    )
    factors = build_factor_frame(candidate_universe, scored_rows)
    source_metadata, model_metadata = factor_batch_metadata(
        release, source_kind="signal_inference"
    )
    input_metadata = FactorBatchInput(
        snapshot_id=str(dataset_manifest["snapshot_id"]),
        snapshot_manifest_sha256=sha256_file(dataset_path / "manifest.json"),
        universe_semantics="complete_candidate_cross_section",
        universe_sha256=factor_universe_sha256(candidate_universe),
        identity_policy=release.feature_contract.identity_policy,
    )
    time_metadata = FactorBatchTime(
        calendar_version=CALENDAR_VERSION,
        minimum_asof_date=factors["asof_date"].min(),
        maximum_asof_date=factors["asof_date"].max(),
    )
    destination, factor_manifest = publish_factor_batch(
        factors,
        output_root,
        source=source_metadata,
        model=model_metadata,
        input_metadata=input_metadata,
        time_metadata=time_metadata,
        before_commit=before_publish,
    )
    return destination, factor_manifest.model_dump(mode="json")


def run_inference(
    run_dir: str | Path,
    *,
    split: InferenceSplit | None = None,
    output_dir: str | Path | None = None,
    dataset_dir: str | Path | None = None,
    device: Literal["auto", "cpu", "cuda"] = "cpu",
    unlock_test: bool = False,
    require_replay_match: bool = True,
) -> tuple[Path, dict[str, Any]]:
    source_run = Path(run_dir).resolve()
    manifest, config_payload, checkpoint_path = _load_source_run(source_run)
    requested_split = split or str(manifest["evaluation_split"])
    if requested_split not in {"train", "valid", "test"}:
        raise ValueError("inference split must be train, valid or test")
    if requested_split == "test" and not unlock_test:
        raise ValueError("test inference requires unlock_test=true")
    dataset_path = Path(dataset_dir or manifest["dataset_path"]).resolve()
    dataset_manifest, frames = _validate_dataset(manifest, dataset_path)
    context_length = int(dataset_manifest["config"]["features"]["context_length"])
    channels = list(dataset_manifest["config"]["features"]["channels"])
    configured_channels = list(config_payload["channels"])
    if configured_channels != channels:
        raise DataContractError("source run channels differ from inference dataset")

    created_at = datetime.now(timezone.utc)
    identity = {
        "source_run": str(source_run),
        "source_manifest": sha256_file(source_run / "manifest.json"),
        "checkpoint": sha256_file(checkpoint_path),
        "dataset": manifest["dataset_id"],
        "split": requested_split,
        "created_at": created_at.isoformat(),
        "nonce": uuid.uuid4().hex,
    }
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else source_run
        / "replays"
        / f"{requested_split}-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{sha256_json(identity)[:8]}"
    ).resolve()
    if destination.exists():
        raise FileExistsError(f"inference output already exists: {destination}")

    if manifest["model_type"] in {"mlp", "lightgbm"}:
        scores, rows, loader_audit = _predict_e0(
            run_dir=source_run,
            manifest=manifest,
            config_payload=config_payload,
            checkpoint_path=checkpoint_path,
            frames=frames,
            context_length=context_length,
            split=requested_split,
            device_preference=device,
        )
    else:
        scores, rows, loader_audit = _predict_patchtst(
            manifest=manifest,
            config_payload=config_payload,
            checkpoint_path=checkpoint_path,
            frames=frames,
            context_length=context_length,
            split=requested_split,
            device_preference=device,
        )
    checkpoint_hash = sha256_file(checkpoint_path)
    predictions, neutralization_audit = build_prediction_frame(
        rows,
        frames["sample_metadata"],
        scores,
        model_id=manifest["model_id"],
        checkpoint_hash=checkpoint_hash,
        dataset_id=manifest["dataset_id"],
    )
    coverage = prediction_coverage(
        predictions,
        frames["sample_index"],
        split=requested_split,
        minimum=float(config_payload["minimum_coverage"]),
    )
    replay_audit = _verify_replay(
        source_run,
        manifest,
        requested_split,
        predictions,
        require_match=require_replay_match,
    )
    source_provenance = load_source_provenance(dataset_path, dataset_manifest)
    factor_metrics = apply_source_readiness_gate(
        evaluate_predictions(predictions, list(config_payload["costs_bps"])),
        source_provenance,
    )
    metrics = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "model_id": manifest["model_id"],
        "dataset_id": manifest["dataset_id"],
        "evaluation_split": requested_split,
        "coverage": coverage,
        "neutralization": neutralization_audit,
        "source_provenance": source_provenance,
        "metrics": factor_metrics,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".tmp-{destination.name}-{uuid.uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    predictions_path = temporary / "predictions.parquet"
    metrics_path = temporary / "metrics.json"
    try:
        predictions.write_parquet(predictions_path)
        _write_json(metrics_path, metrics)
        write_evaluation_report(metrics, temporary / "report.html")
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    inference_manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at": created_at.isoformat(),
        "source_run": str(source_run),
        "source_run_id": manifest["run_id"],
        "source_run_manifest_sha256": sha256_file(source_run / "manifest.json"),
        "source_model_type": manifest["model_type"],
        "source_config_hash": manifest["config_hash"],
        "dataset_id": manifest["dataset_id"],
        "dataset_path": str(dataset_path),
        "dataset_manifest_sha256": sha256_file(dataset_path / "manifest.json"),
        "checkpoint": {
            "file": str(checkpoint_path),
            "sha256": checkpoint_hash,
        },
        "split": requested_split,
        "test_unlocked": requested_split == "test" and unlock_test,
        "row_count": len(predictions),
        "coverage": coverage,
        "loader": loader_audit,
        "replay_verification": replay_audit,
        "environment": collect_environment(include_model_dependencies=True),
        "artifacts": {
            "predictions": {
                "file": "predictions.parquet",
                "sha256": sha256_file(predictions_path),
            },
            "metrics": {"file": "metrics.json", "sha256": sha256_file(metrics_path)},
            "report": "report.html",
        },
    }
    try:
        _write_json(temporary / "manifest.json", inference_manifest)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination, inference_manifest
