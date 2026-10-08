"""Freeze input boundaries before any complete Finance cell or V evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.experiments.manifest import collect_git_state, sha256_json
from facdigger.research.finance_diagnostics import diagnostic_inputs
from facdigger.research.finance_experiment_config import (
    FinanceExperimentConfig,
    cell_config,
    experiment_cells,
)
from facdigger.training.finance_transformer_config import load_finance_transformer_config
from facdigger.training.runtime import TrainingRuntimeConfig, run_lock, write_json


def clean_git(repository_root: Path, expected_commit: str | None = None) -> dict:
    git = collect_git_state(repository_root)
    if git["dirty"] is not False:
        raise DataContractError("complete experiments require a committed clean checkout")
    if expected_commit is not None and git["commit"] != expected_commit:
        raise DataContractError("experiment code commit changed; use the frozen checkout")
    return git


def disjoint_output(output: Path, source: Path) -> None:
    output, source = output.resolve(), source.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise DataContractError("output and protected input must be disjoint")


def _fold_inputs(location: dict, config, fold: dict, *, phases=("F", "S")):
    snapshot, checksums = Path(location["path"]), Path(location["checksums"])
    manifest, protocol, labelled, pools = diagnostic_inputs(
        snapshot, checksums, config, phases=phases,
    )
    if set(labelled) != set(phases) or set(pools) != set(phases):
        raise DataContractError("experiment reader exceeded the requested phases")
    if manifest["config"]["split"] != {k: v for k, v in fold.items() if k != "fold_id"}:
        raise DataContractError("snapshot does not match the declared fold")
    hashes = manifest.get("input_file_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise DataContractError("fold snapshot requires source content hashes")
    expected_id = sha256_json({
        key: manifest[key] for key in ("schema_version", "config", "input_file_hashes")
    })
    if manifest["dataset_id"] != expected_id:
        raise DataContractError("snapshot content identity differs")
    return manifest, protocol, labelled, pools


def freeze_experiment_plan(
    config: FinanceExperimentConfig, runtime: TrainingRuntimeConfig, output: Path, *,
    repository_root: Path,
) -> dict:
    """Read F/S and metadata only; never build/download data or instantiate a model."""
    git = clean_git(repository_root)
    expected = {fold.fold_id for fold in config.folds}
    if set(runtime.fold_snapshots) != expected:
        raise DataContractError("runtime must bind exactly the configured fold snapshots")
    base = load_finance_transformer_config(config.supervised_config).model_dump(mode="json")
    experiment = cell_config(base, config.seeds[0])
    bindings = {}
    source_hashes = common_config = None
    for fold in config.folds:
        location = runtime.fold_snapshots[fold.fold_id].model_dump(mode="json")
        location = {key: str(Path(value).resolve()) for key, value in location.items()}
        disjoint_output(output, Path(location["path"]))
        if Path(location["checksums"]).resolve().is_relative_to(output.resolve()):
            raise DataContractError("input checksum inventory must be outside experiment output")
        manifest, protocol, labelled, pools = _fold_inputs(
            location, experiment, fold.model_dump(mode="json"),
        )
        shared = {key: value for key, value in manifest["config"].items() if key != "split"}
        if source_hashes is not None and (
            source_hashes != manifest["input_file_hashes"] or common_config != shared
        ):
            raise DataContractError("folds must share source revision and non-split data semantics")
        source_hashes, common_config = manifest["input_file_hashes"], shared
        bindings[fold.fold_id] = {
            **location, "dataset_id": manifest["dataset_id"],
            "checksums_sha256": sha256_file(Path(location["checksums"])),
            "data_protocol": protocol,
            "support": {
                phase: {
                    "dates": labelled[phase]["asof_date"].n_unique(),
                    "computational_rows": pools[phase].height,
                    "labelled_rows": labelled[phase].height,
                }
                for phase in ("F", "S")
            },
        }
    identity = {
        "research": config.model_dump(mode="json"), "supervised_config": base,
        "folds": bindings, "cells": experiment_cells(config),
        "observation": {"phases": ["F", "S"], "precision": "fp32", "full_fit": True},
        "code_commit": git["commit"],
    }
    plan = {
        "identity": identity, "plan_hash": sha256_json(identity),
        "git": git, "holdout_used": False, "validation_requires_seal": True,
    }
    with run_lock(output / ".plan.lock"):
        path = output / "plan.json"
        if path.exists():
            previous = read_experiment_plan(output)
            if previous["identity"] != identity:
                raise DataContractError("frozen experiment plan differs; use a new output")
            return previous
        if any(output.iterdir()) and set(p.name for p in output.iterdir()) != {".plan.lock"}:
            raise DataContractError("new experiment output is not empty")
        write_json(path, plan)
    return plan


def read_experiment_plan(output: Path) -> dict:
    plan = json.loads((output / "plan.json").read_text())
    identity = plan["identity"]
    if plan["plan_hash"] != sha256_json(identity) or plan.get("holdout_used") is not False:
        raise DataContractError("frozen experiment plan is corrupt")
    config = FinanceExperimentConfig.model_validate(identity["research"])
    if identity["cells"] != experiment_cells(config):
        raise DataContractError("frozen cell list differs from the research matrix")
    cell_config(identity["supervised_config"], config.seeds[0])
    return plan


def get_cell(plan: dict, cell_id: str) -> dict:
    for cell in plan["identity"]["cells"]:
        if cell["cell_id"] == cell_id:
            return cell
    raise DataContractError("cell is not in the frozen experiment plan")


def load_fold_inputs(plan: dict, fold_id: str, *, seed: int, phases=("F", "S")):
    identity = plan["identity"]
    fold = next(row for row in identity["research"]["folds"] if row["fold_id"] == fold_id)
    location = identity["folds"][fold_id]
    if sha256_file(Path(location["checksums"])) != location["checksums_sha256"]:
        raise DataContractError("bound snapshot checksum inventory changed")
    result = _fold_inputs(
        location, cell_config(identity["supervised_config"], seed), fold, phases=phases,
    )
    if result[1] != location["data_protocol"]:
        raise DataContractError("fold data protocol changed")
    return result


def experiment_status(output: Path) -> dict:
    plan = read_experiment_plan(output)
    cells = []
    for cell in plan["identity"]["cells"]:
        root = output / "cells" / cell["cell_id"]
        path = root / "cell.json"
        audit = json.loads(path.read_text()) if path.is_file() else {}
        cells.append({
            **cell, "status": audit.get("status", "pending"),
            "epoch": audit.get("epoch"), "global_step": audit.get("global_step"),
            "best_epoch": audit.get("best_epoch"),
            "validation_used": False,
        })
    return {
        "research_id": plan["identity"]["research"]["research_id"],
        "plan_hash": plan["plan_hash"], "cells": cells,
        "complete_cells": sum(row["status"] == "complete" for row in cells),
        "selections_sealed": (output / "selection-seal.json").is_file(),
        "holdout_used": False,
    }
