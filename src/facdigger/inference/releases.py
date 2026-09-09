"""Immutable model releases for reproducible target-free inference."""

from __future__ import annotations

import json
import logging
import re
import shutil
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import Field, field_validator, model_validator

from facdigger.data.config import StrictModel
from facdigger.data.contracts import DataContractError
from facdigger.data.paths import artifact_path
from facdigger.data.snapshots import sha256_file
from facdigger.experiments.manifest import collect_git_state, sha256_json
from facdigger.inference.backends import (
    PATCHTST_MODEL_TYPES,
    RELEASABLE_MODEL_TYPES,
    ReleasableModelType,
)
from facdigger.inference.source import _load_source_run, resolve_training_snapshot
from facdigger.training.ranking import RANKING_OBJECTIVE, TARGET_TRANSFORM

logger = logging.getLogger(__name__)
MODEL_RELEASE_CONTRACT = "facdigger.model_release"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IdentityPolicy = Literal["provider_neutral_security_id", "eodhd_isin_only"]
REQUIRED_ARTIFACTS = {
    "checkpoint",
    "checkpoint_protocol",
    "resolved_config",
    "scaler",
    "training_dataset_manifest",
    "source_run_manifest",
}


class ReleaseArtifact(StrictModel):
    file: str = Field(min_length=1)
    sha256: str
    bytes: int = Field(ge=1)

    @field_validator("sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("artifact digest must be a lowercase SHA-256")
        return value


class ReleaseSource(StrictModel):
    repository: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    run_id: str = Field(min_length=1)
    run_manifest_sha256: str
    predictions_sha256: str
    git_clean: bool = Field(strict=True)

    @field_validator("run_manifest_sha256", "predictions_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("source digest must be a lowercase SHA-256")
        return value


class ReleaseTrainingData(StrictModel):
    dataset_id: str = Field(min_length=1)
    dataset_manifest_sha256: str

    @field_validator("dataset_manifest_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("dataset manifest digest must be a lowercase SHA-256")
        return value


class ReleaseFeatureContract(StrictModel):
    feature_set: str = Field(min_length=1)
    channels: list[str] = Field(min_length=1)
    context_length: int = Field(ge=1)
    scaler_sha256: str
    scaler_contract: str = Field(min_length=1)
    identity_policy: IdentityPolicy

    @field_validator("scaler_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("scaler digest must be a lowercase SHA-256")
        return value


class FinanceReleaseFeatureContract(ReleaseFeatureContract):
    """Extra inputs only present for the finance-native family; old identities stay intact."""

    feature_set: Literal["finance_transformer"] = "finance_transformer"
    market_channels: list[str] = Field(min_length=1)


class ModelReleaseManifest(StrictModel):
    contract: Literal["facdigger.model_release"] = MODEL_RELEASE_CONTRACT
    status: Literal["complete"] = "complete"
    release_id: str
    created_at: datetime
    model_id: str = Field(min_length=1)
    model_type: ReleasableModelType
    forecast_horizon_sessions: int = Field(ge=1)
    higher_score_is_better: Literal[True] = True
    objective: str = Field(min_length=1)
    target_transform: str = Field(min_length=1)
    source: ReleaseSource
    training_data: ReleaseTrainingData
    feature_contract: FinanceReleaseFeatureContract | ReleaseFeatureContract
    artifacts: dict[str, ReleaseArtifact]

    @field_validator("release_id")
    @classmethod
    def validate_release_id(cls, value: str) -> str:
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("release_id must be a lowercase SHA-256")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
            raise ValueError("created_at must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def validate_artifacts(self) -> ModelReleaseManifest:
        if (self.model_type == "finance_patch_transformer") != isinstance(
            self.feature_contract, FinanceReleaseFeatureContract
        ):
            raise ValueError("release model family and feature contract disagree")
        if set(self.artifacts) != REQUIRED_ARTIFACTS:
            raise ValueError(
                f"model release artifact names must exactly equal {sorted(REQUIRED_ARTIFACTS)}"
            )
        files = [artifact.file for artifact in self.artifacts.values()]
        if len(files) != len(set(files)):
            raise ValueError("model release artifact paths must be unique")
        return self


def _identity_payload(manifest: ModelReleaseManifest) -> dict[str, Any]:
    payload = manifest.model_dump(mode="json")
    payload.pop("created_at")
    payload.pop("release_id")
    return payload


def model_release_id(manifest: ModelReleaseManifest) -> str:
    return sha256_json(_identity_payload(manifest))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact(path: Path, relative: str) -> ReleaseArtifact:
    return ReleaseArtifact(file=relative, sha256=sha256_file(path), bytes=path.stat().st_size)


def _checkpoint_protocol(
    checkpoint_path: Path,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - model extra is required by the CLI
        raise RuntimeError("creating a model release requires the model dependencies") from exc
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise DataContractError("model checkpoint payload must be a mapping")
    protocol: dict[str, Any] = {
        "schema_version": payload.get("schema_version"),
        "model_type": payload.get("model_type", payload.get("experiment_family")),
        "objective": payload.get("objective"),
        "target_transform": payload.get("target_transform"),
    }
    optimization = payload.get("optimization_protocol")
    if optimization is not None:
        protocol["optimization_protocol"] = optimization
    dataset_id = payload.get("dataset_id")
    if dataset_id is not None:
        protocol["training_dataset_id"] = dataset_id
    if payload.get("model_type") == "finance_patch_transformer":
        protocol["contract"] = payload.get("contract")
        protocol["config_hash"] = sha256_json(payload.get("config"))
    return protocol


def _validate_checkpoint_release_protocol(
    model_type: str,
    protocol: Mapping[str, Any],
    training_dataset_id: str,
) -> None:
    if model_type == "finance_patch_transformer":
        if (
            protocol.get("schema_version") != 4
            or protocol.get("contract") != "finance_patch_transformer_checkpoint"
            or protocol.get("model_type") != model_type
            or protocol.get("objective") != "multi_horizon_full_date_rank_correlation"
            or (protocol.get("optimization_protocol") or {}).get("method")
            != "exact_leaf_embedding_replay"
        ):
            raise DataContractError(
                "finance Transformer release requires its full-date checkpoint protocol"
            )
    else:
        if protocol.get("schema_version") != 3:
            raise DataContractError(
                "PatchTST release requires checkpoint schema 3 with full-date ranking"
            )
        if protocol.get("objective") != RANKING_OBJECTIVE:
            raise DataContractError("PatchTST release requires the current ranking objective")
    if protocol.get("target_transform") != TARGET_TRANSFORM:
        raise DataContractError("PatchTST release requires the current target transform")
    expected_checkpoint_family = PATCHTST_MODEL_TYPES.get(model_type, model_type)
    if protocol.get("model_type") != expected_checkpoint_family:
        raise DataContractError("checkpoint experiment family differs from the source model type")
    optimization = protocol.get("optimization_protocol") or {}
    if optimization.get("unit") != "complete_date":
        raise DataContractError("PatchTST release requires complete-date optimization")
    if protocol.get("training_dataset_id") != training_dataset_id:
        raise DataContractError("checkpoint training dataset ID differs from source run")


def _validate_finance_inputs(
    run: Mapping[str, Any],
    dataset: Mapping[str, Any],
    protocol: Mapping[str, Any],
    config_payload: dict[str, Any],
) -> None:
    if run["model_type"] != "finance_patch_transformer":
        return
    from facdigger.training.finance_transformer_config import FinanceTransformerExperimentConfig

    config = FinanceTransformerExperimentConfig.model_validate(config_payload)
    features = dataset["config"]["features"]
    label = dataset["config"]["label"]
    inputs = run.get("input") or {}
    if (
        features.get("name") != "finance_transformer"
        or features.get("channels") != config.channels
        or features.get("market_channels") != config.market_channels
        or inputs.get("market_channels") != config.market_channels
        or inputs.get("horizons") != config.horizons
        or inputs.get("primary_horizon") != config.primary_horizon
        or label.get("horizon") != config.primary_horizon
        or sorted([label["horizon"], *label.get("auxiliary_horizons", [])]) != config.horizons
    ):
        raise DataContractError(
            "finance Transformer inputs or horizons differ from training contract"
        )
    if protocol.get("config_hash") != sha256_json(config_payload):
        raise DataContractError(
            "finance Transformer checkpoint configuration differs from source run"
        )


def create_model_release(
    run_dir: str | Path,
    output_root: str | Path,
    *,
    repository_root: str | Path,
    created_at: datetime | None = None,
    dataset_dir: str | Path | None = None,
    allow_dirty: bool = False,
) -> tuple[Path, ModelReleaseManifest]:
    """Copy and bind every artifact required for immutable inference."""

    source_run = Path(run_dir).resolve()
    source_run_manifest_path = source_run / "manifest.json"
    source_run_manifest_hash = sha256_file(source_run_manifest_path)
    run_manifest, config_payload, checkpoint_path = _load_source_run(source_run)
    if run_manifest.get("status") != "complete":
        raise DataContractError("model release requires an explicitly complete source run")
    if sha256_file(source_run_manifest_path) != source_run_manifest_hash:
        raise DataContractError("source run manifest changed while creating the release")
    repository = Path(repository_root).resolve()
    current_git = collect_git_state(repository)
    if current_git["commit"] is None or not re.fullmatch(
        r"[0-9a-f]{40}", str(current_git["commit"])
    ):
        raise DataContractError("model release requires a readable current Git commit")
    if not isinstance(current_git.get("dirty"), bool):
        raise DataContractError("model release publisher Git state is unavailable")
    if current_git["dirty"]:
        if not allow_dirty:
            raise DataContractError("model release publisher requires a clean Git worktree")
        logger.warning("Allowing dirty publisher worktree for integration testing")
    run_git = run_manifest.get("git") or {}
    if not re.fullmatch(r"[0-9a-f]{40}", str(run_git.get("commit", ""))):
        raise DataContractError("source run must record a readable Git commit")
    if not isinstance(run_git.get("dirty"), bool):
        raise DataContractError("source run Git state is unavailable")
    if run_git["dirty"]:
        if not allow_dirty:
            raise DataContractError("source run must have been created from a clean Git worktree")
        logger.warning("Allowing dirty source run; unrecorded code changes cannot be reconstructed")
    predictions_relative = (run_manifest.get("artifacts") or {}).get("predictions")
    if not isinstance(predictions_relative, str):
        raise DataContractError("source run does not declare its predictions artifact")
    predictions_path = artifact_path(
        source_run, predictions_relative, "source predictions artifact"
    )
    predictions_hash = sha256_file(predictions_path)
    if run_manifest.get("predictions_sha256") != predictions_hash:
        raise DataContractError(
            "source run does not bind its predictions artifact; old or modified runs "
            "cannot be released"
        )
    raw_model_type = str(run_manifest["model_type"])
    if raw_model_type not in RELEASABLE_MODEL_TYPES:
        raise DataContractError(f"ModelRelease has no inference adapter for {raw_model_type}")
    model_type = cast(ReleasableModelType, raw_model_type)
    checkpoint_protocol = _checkpoint_protocol(checkpoint_path)
    _validate_checkpoint_release_protocol(
        model_type, checkpoint_protocol, str(run_manifest["dataset_id"])
    )
    dataset_path = resolve_training_snapshot(run_manifest, dataset_dir)
    dataset_manifest_path = dataset_path / "manifest.json"
    scaler_path = dataset_path / "scaler.json"
    if not dataset_manifest_path.is_file() or not scaler_path.is_file():
        raise FileNotFoundError("source training snapshot is missing manifest.json or scaler.json")
    if sha256_file(dataset_manifest_path) != run_manifest["dataset_manifest_hash"]:
        raise DataContractError("source training dataset manifest hash has changed")
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("dataset_id") != run_manifest["dataset_id"]:
        raise DataContractError("source run and training snapshot dataset IDs disagree")
    run_input = run_manifest.get("input") or {}
    scaler_hash = sha256_file(scaler_path)
    if run_input.get("feature_scaler_sha256") != scaler_hash:
        raise DataContractError(
            "source run does not bind the current training scaler; old or modified runs "
            "cannot be released"
        )
    feature_config = dataset_manifest.get("config", {}).get("features", {})
    if not isinstance(feature_config, dict):
        raise DataContractError("training snapshot has no feature configuration")
    if run_input.get("feature_set") != feature_config.get("name"):
        raise DataContractError("source run feature set differs from training snapshot")
    if run_input.get("context_length") != feature_config.get("context_length"):
        raise DataContractError("source run context length differs from training snapshot")
    if run_input.get("channels") != feature_config.get("channels"):
        raise DataContractError("source run channels differ from training snapshot")
    if run_input.get("feature_scaler") != feature_config.get("scaler"):
        raise DataContractError("source run scaler contract differs from training snapshot")
    horizon = int(dataset_manifest.get("config", {}).get("label", {}).get("horizon", 0))
    if horizon < 1:
        raise DataContractError("training snapshot has no valid forecast horizon")

    _validate_finance_inputs(run_manifest, dataset_manifest, checkpoint_protocol, config_payload)
    feature_type = (
        FinanceReleaseFeatureContract
        if model_type == "finance_patch_transformer"
        else ReleaseFeatureContract
    )
    feature_contract = feature_type(
        feature_set=str(feature_config.get("name")),
        channels=[str(value) for value in feature_config.get("channels", [])],
        context_length=int(feature_config.get("context_length", 0)),
        scaler_sha256=scaler_hash,
        scaler_contract=str(feature_config.get("scaler")),
        # Retain the field for existing immutable releases. Cross-project identity
        # is a delivery property, not a promise about every model input security.
        identity_policy="provider_neutral_security_id",
        **(
            {"market_channels": feature_config.get("market_channels")}
            if model_type == "finance_patch_transformer"
            else {}
        ),
    )
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".tmp-model-release-{uuid.uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        checkpoint_target = temporary / "checkpoint" / checkpoint_path.name
        checkpoint_target.parent.mkdir()
        shutil.copyfile(checkpoint_path, checkpoint_target)
        config_target = temporary / "resolved_config.yaml"
        shutil.copyfile(source_run / "resolved_config.yaml", config_target)
        scaler_target = temporary / "scaler.json"
        shutil.copyfile(scaler_path, scaler_target)
        dataset_manifest_target = temporary / "training_dataset_manifest.json"
        shutil.copyfile(dataset_manifest_path, dataset_manifest_target)
        run_manifest_target = temporary / "source_run_manifest.json"
        shutil.copyfile(source_run / "manifest.json", run_manifest_target)
        if sha256_file(run_manifest_target) != source_run_manifest_hash:
            raise DataContractError("source run manifest changed during release copy")
        artifacts = {
            "checkpoint": _artifact(
                checkpoint_target, checkpoint_target.relative_to(temporary).as_posix()
            ),
            "resolved_config": _artifact(config_target, config_target.name),
            "scaler": _artifact(scaler_target, scaler_target.name),
            "training_dataset_manifest": _artifact(
                dataset_manifest_target, dataset_manifest_target.name
            ),
            "source_run_manifest": _artifact(run_manifest_target, run_manifest_target.name),
        }
        checkpoint_protocol_target = temporary / "checkpoint_protocol.json"
        _write_json(checkpoint_protocol_target, checkpoint_protocol)
        artifacts["checkpoint_protocol"] = _artifact(
            checkpoint_protocol_target, checkpoint_protocol_target.name
        )
        provisional = ModelReleaseManifest(
            release_id="0" * 64,
            created_at=created_at or datetime.now(timezone.utc),
            model_id=str(run_manifest["model_id"]),
            model_type=model_type,
            forecast_horizon_sessions=horizon,
            objective=str(checkpoint_protocol["objective"]),
            target_transform=str(checkpoint_protocol["target_transform"]),
            source=ReleaseSource(
                repository=repository.name,
                commit=str(run_git["commit"]),
                run_id=str(run_manifest["run_id"]),
                run_manifest_sha256=artifacts["source_run_manifest"].sha256,
                predictions_sha256=predictions_hash,
                git_clean=not run_git["dirty"],
            ),
            training_data=ReleaseTrainingData(
                dataset_id=str(run_manifest["dataset_id"]),
                dataset_manifest_sha256=artifacts["training_dataset_manifest"].sha256,
            ),
            feature_contract=feature_contract,
            artifacts=artifacts,
        )
        manifest = provisional.model_copy(update={"release_id": model_release_id(provisional)})
        destination = root / manifest.release_id
        if destination.exists():
            existing = load_model_release(destination)
            if existing.release_id != manifest.release_id:
                raise DataContractError("existing model release identity does not match")
            shutil.rmtree(temporary)
            return destination, existing
        _write_json(temporary / "manifest.json", manifest.model_dump(mode="json"))
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination, manifest


def load_model_release(release_dir: str | Path) -> ModelReleaseManifest:
    """Validate an immutable release before any inference artifact is consumed."""

    root = Path(release_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"model release manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataContractError("model release manifest is unreadable") from exc
    manifest = ModelReleaseManifest.model_validate(payload)
    if root.name != manifest.release_id:
        raise DataContractError("model release directory name must equal release_id")
    if manifest.release_id != model_release_id(manifest):
        raise DataContractError("model release semantic identity does not match release_id")
    actual_entries = {path.relative_to(root).as_posix() for path in root.rglob("*")}
    expected_files = {"manifest.json"}
    for artifact in manifest.artifacts.values():
        relative = artifact_path(root, artifact.file, "release artifact").relative_to(root)
        expected_files.add(relative.as_posix())
        expected_files.update(
            parent.as_posix() for parent in relative.parents if parent != Path(".")
        )
    if actual_entries != expected_files:
        raise DataContractError("model release contains undeclared or missing entries")
    paths: dict[str, Path] = {}
    for name, artifact in manifest.artifacts.items():
        path = artifact_path(root, artifact.file, f"release artifact {name}")
        paths[name] = path
        if sha256_file(path) != artifact.sha256 or path.stat().st_size != artifact.bytes:
            raise DataContractError(f"model release artifact integrity failure: {name}")
    source_manifest = json.loads(
        paths["source_run_manifest"].read_text(encoding="utf-8")
    )
    if manifest.artifacts["source_run_manifest"].sha256 != manifest.source.run_manifest_sha256:
        raise DataContractError("release source run artifact identity differs from manifest")
    if source_manifest.get("status") != "complete":
        raise DataContractError("release source run is not complete")
    if source_manifest.get("run_id") != manifest.source.run_id:
        raise DataContractError("release source run ID differs from manifest")
    if source_manifest.get("predictions_sha256") != manifest.source.predictions_sha256:
        raise DataContractError("release source predictions identity differs from manifest")
    if source_manifest.get("model_id") != manifest.model_id:
        raise DataContractError("release source model ID differs from manifest")
    if source_manifest.get("model_type") != manifest.model_type:
        raise DataContractError("release source model type differs from manifest")
    if source_manifest.get("dataset_id") != manifest.training_data.dataset_id:
        raise DataContractError("release source training dataset differs from manifest")
    source_input = source_manifest.get("input") or {}
    if source_input.get("feature_set") != manifest.feature_contract.feature_set:
        raise DataContractError("release source feature set differs from manifest")
    if source_input.get("context_length") != manifest.feature_contract.context_length:
        raise DataContractError("release source context length differs from manifest")
    if source_input.get("channels") != manifest.feature_contract.channels:
        raise DataContractError("release source channels differ from manifest")
    if source_input.get("feature_scaler") != manifest.feature_contract.scaler_contract:
        raise DataContractError("release source scaler contract differs from manifest")
    if source_input.get("feature_scaler_sha256") != manifest.feature_contract.scaler_sha256:
        raise DataContractError("release source scaler hash differs from manifest")
    source_git = source_manifest.get("git") or {}
    if (
        source_git.get("commit") != manifest.source.commit
        or not isinstance(source_git.get("dirty"), bool)
        or (not source_git["dirty"]) != manifest.source.git_clean
    ):
        raise DataContractError("release source Git lineage differs from manifest")
    if manifest.artifacts["checkpoint"].sha256 != source_manifest["checkpoint"]["sha256"]:
        raise DataContractError("release checkpoint differs from source run manifest")
    if manifest.artifacts["scaler"].sha256 != manifest.feature_contract.scaler_sha256:
        raise DataContractError("release scaler differs from feature contract")
    if (
        manifest.artifacts["training_dataset_manifest"].sha256
        != manifest.training_data.dataset_manifest_sha256
    ):
        raise DataContractError("release dataset manifest differs from training data contract")
    if (
        source_manifest.get("dataset_manifest_hash")
        != manifest.training_data.dataset_manifest_sha256
    ):
        raise DataContractError("release source run dataset manifest identity differs")
    training_dataset_manifest = json.loads(
        paths["training_dataset_manifest"].read_text(encoding="utf-8")
    )
    if training_dataset_manifest.get("dataset_id") != manifest.training_data.dataset_id:
        raise DataContractError("release training dataset ID differs from copied manifest")
    dataset_config = training_dataset_manifest.get("config") or {}
    feature_config = dataset_config.get("features") or {}
    expected_feature_contract = {
        "feature_set": feature_config.get("name"),
        "channels": feature_config.get("channels"),
        "context_length": feature_config.get("context_length"),
        "scaler_contract": feature_config.get("scaler"),
    }
    observed_feature_contract = {
        "feature_set": manifest.feature_contract.feature_set,
        "channels": manifest.feature_contract.channels,
        "context_length": manifest.feature_contract.context_length,
        "scaler_contract": manifest.feature_contract.scaler_contract,
    }
    if observed_feature_contract != expected_feature_contract:
        raise DataContractError("release feature contract differs from training dataset")
    if (dataset_config.get("label") or {}).get("horizon") != (manifest.forecast_horizon_sessions):
        raise DataContractError("release forecast horizon differs from training dataset")
    copied_checkpoint_protocol = json.loads(
        paths["checkpoint_protocol"].read_text(encoding="utf-8")
    )
    actual_checkpoint_protocol = _checkpoint_protocol(paths["checkpoint"])
    if copied_checkpoint_protocol != actual_checkpoint_protocol:
        raise DataContractError("release checkpoint protocol differs from checkpoint")
    _validate_checkpoint_release_protocol(
        manifest.model_type,
        actual_checkpoint_protocol,
        manifest.training_data.dataset_id,
    )
    if actual_checkpoint_protocol.get("objective") != manifest.objective:
        raise DataContractError("release checkpoint objective differs from manifest")
    if actual_checkpoint_protocol.get("target_transform") != manifest.target_transform:
        raise DataContractError("release checkpoint target transform differs from manifest")
    config_payload = yaml.safe_load(
        paths["resolved_config"].read_text(encoding="utf-8")
    )
    if sha256_json(config_payload) != source_manifest["config_hash"]:
        raise DataContractError("release resolved configuration differs from source run")
    _validate_finance_inputs(
        source_manifest, training_dataset_manifest, actual_checkpoint_protocol, config_payload
    )
    if isinstance(manifest.feature_contract, FinanceReleaseFeatureContract):
        if feature_config.get("market_channels") != manifest.feature_contract.market_channels:
            raise DataContractError("release market channels differ from training dataset")
    return manifest


def release_runtime(
    release_dir: str | Path,
) -> tuple[ModelReleaseManifest, dict[str, Any], Path]:
    """Return verified release metadata, resolved model config and checkpoint path."""

    root = Path(release_dir).resolve()
    release = load_model_release(root)
    config = yaml.safe_load(
        artifact_path(root, release.artifacts["resolved_config"].file, "release config").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(config, dict):
        raise DataContractError("release resolved configuration must be a mapping")
    return release, config, artifact_path(root, release.artifacts["checkpoint"].file, "checkpoint")
