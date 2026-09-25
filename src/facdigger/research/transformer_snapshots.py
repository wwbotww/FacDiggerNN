"""CPU-only fold preparation shared by independent deployment and the training runner."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from facdigger.data.config import load_dataset_build_config
from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import build_dataset_snapshot, sha256_file
from facdigger.experiments.manifest import sha256_json
from facdigger.research.transformer_config import TransformerComparisonConfig
from facdigger.training.common import load_source_provenance
from facdigger.training.runtime import (
    DatasetLocation,
    TrainingRuntimeConfig,
    atomic_write_text,
    load_training_runtime,
    resolve_dataset,
    run_lock,
    snapshot_checksums,
    write_json,
)


def _iter_snapshots(
    config: TransformerComparisonConfig,
    locations: dict[str, DatasetLocation],
    *,
    allow_build: bool,
) -> Iterator[dict[str, Any]]:
    base = load_dataset_build_config(config.base_dataset_config)
    if base.features.name != "finance_transformer":
        raise DataContractError("streamlined comparison requires finance_transformer features")
    fold_ids = {fold.fold_id for fold in config.folds}
    if set(locations) - fold_ids or (not allow_build and set(locations) != fold_ids):
        raise DataContractError("prebuilt snapshots must specify exactly the configured folds")
    source_hashes = None
    for fold in config.folds:
        split = fold.model_dump(mode="json", exclude={"fold_id"})
        fold_config = base.model_copy(
            update={
                "output_root": config.snapshot_output_root,
                "split": base.split.model_validate(split),
            }
        )
        location = locations.get(fold.fold_id)
        if location is None:
            snapshot, manifest = build_dataset_snapshot(fold_config)
        else:
            manifest = json.loads((location.path / "manifest.json").read_text(encoding="utf-8"))
            dataset_id = str(manifest["dataset_id"])
            snapshot = resolve_dataset(
                location.path,
                TrainingRuntimeConfig(dataset_overrides={dataset_id: location}),
                dataset_id=dataset_id,
            )
            expected = fold_config.model_dump(mode="json", exclude={"sources", "output_root"})
            if manifest["config"] != expected:
                raise DataContractError("prebuilt snapshot does not match fold dataset protocol")
            identity = {
                key: manifest[key] for key in ("schema_version", "config", "input_file_hashes")
            }
            if int(manifest["schema_version"]) < 4 or sha256_json(identity) != dataset_id:
                raise DataContractError("prebuilt snapshot identity does not match")
            load_source_provenance(snapshot, manifest)
        if not isinstance(manifest["input_file_hashes"], dict) or not manifest["input_file_hashes"]:
            raise DataContractError("fold snapshot source hashes must be a nonempty mapping")
        if source_hashes is not None and source_hashes != manifest["input_file_hashes"]:
            raise DataContractError("fold snapshots must use the same source files and revision")
        source_hashes = manifest["input_file_hashes"]
        yield {
            "fold_id": fold.fold_id,
            "split": split,
            "dataset_id": manifest["dataset_id"],
            "dataset_path": str(snapshot.resolve()),
            "dataset_manifest_sha256": sha256_file(snapshot / "manifest.json"),
        }


def build_transformer_snapshots(
    config: TransformerComparisonConfig, runtime: TrainingRuntimeConfig | None = None
) -> list[dict[str, Any]]:
    locations = (runtime or TrainingRuntimeConfig()).fold_snapshots
    return list(_iter_snapshots(config, locations, allow_build=not bool(locations)))


def prepare_transformer_snapshots(
    config: TransformerComparisonConfig,
    output: Path,
    runtime: TrainingRuntimeConfig | None = None,
) -> dict[str, Any]:
    """Commit each complete fold and reuse verified folds after a CPU job interruption.

    Feature computation inside an unfinished fold restarts. No model, benchmark,
    provider credential or bronze data is needed when verifying completed inputs.
    """
    runtime = runtime or TrainingRuntimeConfig()
    base = load_dataset_build_config(config.base_dataset_config)
    binding = {
        "research": config.model_dump(mode="json"),
        "dataset": base.model_dump(mode="json"),
        "runtime": runtime.model_dump(mode="json"),
    }
    root = output.resolve()
    if root.is_relative_to(config.snapshot_output_root.resolve()) or any(
        root.is_relative_to(location.path.resolve()) for location in runtime.fold_snapshots.values()
    ):
        raise ValueError("preparation output must be outside immutable snapshots")
    with run_lock(root / ".prepare.lock"):
        state_path = root / "preparation.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["binding"] != binding:
                raise DataContractError("preparation configuration changed; use a new output")
        else:
            state = {"status": "preparing", "binding": binding, "locations": {}}
            write_json(state_path, state)
        locations = dict(runtime.fold_snapshots)
        locations.update(
            {
                key: DatasetLocation.model_validate(value)
                for key, value in state["locations"].items()
            }
        )
        completed = state["status"] == "complete"
        plans = []
        # A supplied mapping is complete by contract; partial progress from this
        # preparation is the only case that may build missing folds.
        if runtime.fold_snapshots and set(runtime.fold_snapshots) != {
            fold.fold_id for fold in config.folds
        }:
            raise DataContractError("prebuilt snapshots must specify exactly the configured folds")
        for plan in _iter_snapshots(config, locations, allow_build=not completed):
            fold_id = plan["fold_id"]
            snapshot = Path(plan["dataset_path"])
            checksums = root / f"{fold_id}.checksums.json"
            if checksums.is_relative_to(snapshot):
                raise ValueError("preparation output must be outside immutable snapshots")
            # The iterator already checked every byte of an existing location.
            inventory = (
                json.loads(locations[fold_id].checksums.read_text(encoding="utf-8"))
                if fold_id in locations
                else snapshot_checksums(snapshot)
            )
            if checksums.exists():
                if json.loads(checksums.read_text(encoding="utf-8")) != inventory:
                    raise DataContractError("prepared snapshot differs from committed checksums")
            elif completed:
                raise DataContractError("completed preparation checksum inventory is missing")
            else:
                write_json(checksums, inventory)
            locations[fold_id] = DatasetLocation(path=snapshot, checksums=checksums)
            plans.append(plan)
            if not completed:
                state["locations"][fold_id] = locations[fold_id].model_dump(mode="json")
                write_json(state_path, state)
        generated_runtime = runtime.model_copy(update={"fold_snapshots": locations})
        runtime_path = root / "runtime.yaml"
        folds_path = root / "folds.json"
        if completed:
            if load_training_runtime(runtime_path) != generated_runtime or (
                json.loads(folds_path.read_text(encoding="utf-8")) != plans
            ):
                raise DataContractError("completed preparation outputs changed")
        else:
            atomic_write_text(
                runtime_path,
                yaml.safe_dump(generated_runtime.model_dump(mode="json"), sort_keys=True),
            )
            write_json(folds_path, plans)
            state["status"] = "complete"
            write_json(state_path, state)
        return {
            "status": "complete",
            "folds": plans,
            "runtime": str(runtime_path),
            "preparation": str(state_path),
        }
