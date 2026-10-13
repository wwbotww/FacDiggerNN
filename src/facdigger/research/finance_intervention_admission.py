"""Read-only real-scale cache, market-state and epoch-zero pairing admission."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from facdigger.data.contracts import DataContractError
from facdigger.datasets.finance_statistics import FinanceStatisticsDataset
from facdigger.environment import collect_environment
from facdigger.research.finance_comparison import read_comparison
from facdigger.research.finance_experiment_cache import verify_conditional_market
from facdigger.research.finance_experiment_config import cell_config
from facdigger.research.finance_experiment_plan import clean_git, disjoint_output, load_fold_inputs
from facdigger.research.finance_experiment_state import (
    build_cell_model,
    execution_identity,
    protocol_hash,
)
from facdigger.research.finance_experiment_summary import checked_file
from facdigger.training.e1_engine import seed_everything
from facdigger.training.runtime import write_json


def admit_interventions(
    comparison_root: Path,
    caches: dict[str, Path],
    output: Path,
    *,
    repository_root: Path,
) -> dict:
    git = clean_git(repository_root)
    comparison, plans = read_comparison(comparison_root)
    if output.exists():
        raise DataContractError("admission requires a new output directory")
    disjoint_output(output, comparison_root)
    reference = plans["reference"]
    source = Path(comparison["identity"]["sources"]["reference"]["path"])
    for binding in comparison["identity"]["sources"].values():
        disjoint_output(output, Path(binding["path"]))
    if set(caches) != set(reference["identity"]["folds"]):
        raise DataContractError("admission requires every fold cache")
    old_execution = json.loads((source / "execution.json").read_text())
    config = cell_config(reference["identity"]["supervised_config"], 17)
    current_execution = execution_identity(config, collect_environment())
    if current_execution != old_execution:
        raise DataContractError(
            "admission numerical environment differs from the original controls"
        )
    hashes, market_audit, paired = {}, {}, []
    for fold_id, location in reference["identity"]["folds"].items():
        disjoint_output(output, Path(location["path"]))
        manifest, protocol, _, pools = load_fold_inputs(reference, fold_id, seed=17)
        cache = caches[fold_id]
        disjoint_output(output, cache)
        cached = {
            phase: FinanceStatisticsDataset(
                cache / phase,
                identity=protocol,
                windows=tuple(config.model.statistics_windows),
                expected_rows=pools[phase],
            )
            for phase in ("F", "S")
        }
        market_audit[fold_id] = verify_conditional_market(
            Path(location["path"]),
            manifest,
            config,
            pools,
            cached,
        )
        context = manifest["config"]["features"]["context_length"]
        for cell in reference["identity"]["cells"]:
            if cell["fold_id"] != fold_id:
                continue
            root = source / "cells" / cell["cell_id"]
            audit = json.loads((root / "cell.json").read_text())
            name = "observations/epoch-0.pt"
            weights = torch.load(
                checked_file(root, name, audit["artifacts"][name], hashes),
                map_location="cpu",
                weights_only=False,
            )
            if weights["epoch"] != 0 or weights["identity"] != audit["identity"]:
                raise DataContractError("reference epoch-zero weight binding differs")
            old_config = cell_config(reference["identity"]["supervised_config"], cell["seed"])
            best = torch.load(root / "checkpoints/best.pt", map_location="cpu", weights_only=False)
            if protocol_hash(old_config, protocol, cell["candidate"]) != best["protocol_hash"]:
                raise DataContractError("original serialized protocol changed")
            variants = [("reference", cell["candidate"])]
            if cell["candidate"] == "statistics_linear":
                variants.append(("conditional", "statistics_conditional_linear"))
            elif cell["candidate"] in {"statistics_mlp_dropout03", "finance"}:
                variants.append(("low_lr", cell["candidate"]))
            for role, candidate in variants:
                new_config = cell_config(plans[role]["identity"]["supervised_config"], cell["seed"])
                seed_everything(cell["seed"])
                model = build_cell_model(new_config, candidate, context, "cpu")
                actual = model.state_dict()
                if candidate == "statistics_conditional_linear":
                    if torch.count_nonzero(actual.pop("interaction")):
                        raise DataContractError("conditional interaction is not zero at epoch zero")
                expected = weights["model_state"]
                if actual.keys() != expected.keys() or any(
                    not torch.equal(actual[key], expected[key]) for key in actual
                ):
                    raise DataContractError(
                        "epoch-zero weights differ from the paired saved control"
                    )
                paired.append(
                    {
                        "source": role,
                        "fold_id": fold_id,
                        "seed": cell["seed"],
                        "candidate": candidate,
                        "control": cell["cell_id"],
                        "exact_match": True,
                    }
                )
                del model
        del cached
    result = {
        "comparison_hash": comparison["comparison_hash"],
        "code_commit": git["commit"],
        "execution": current_execution,
        "market": market_audit,
        "epoch_zero": paired,
        "caches": {key: str(value.resolve()) for key, value in caches.items()},
        "source_files": hashes,
        "optimizer_updates": 0,
        "validation_used": False,
        "holdout_used": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "admission.json", result)
    return {
        key: value for key, value in result.items() if key not in {"epoch_zero", "source_files"}
    }
