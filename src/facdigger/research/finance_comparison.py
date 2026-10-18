"""Frozen cross-plan controls and the complete October static comparison family."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.experiments.manifest import sha256_json
from facdigger.research.finance_experiment_plan import (
    clean_git,
    disjoint_output,
    read_experiment_plan,
)
from facdigger.research.finance_experiment_summary import read_completed_cell, summarize_experiments
from facdigger.training.runtime import write_json

BASE_METHODS = ("linear", "mlp01", "mlp03", "finance", "conditional_linear")
STATIC_COMPARISONS = (
    [
        {"name": f"{left}-minus-{right}", "candidate": left, "control": right}
        for right, left in combinations(BASE_METHODS, 2)
    ]
    + [
        {
            "name": f"{model}_{change}-minus-{model}",
            "candidate": f"{model}_{change}",
            "control": model,
        }
        for change in ("lr03", "epochs30", "no_aux")
        for model in ("mlp03", "finance")
    ]
    + [
        {
            "name": "finance_no_stock_interaction-minus-finance",
            "candidate": "finance_no_stock_interaction",
            "control": "finance",
        },
        {"name": "finance_ssl3-minus-finance", "candidate": "finance_ssl3", "control": "finance"},
        {"name": "finance_ssl10-minus-finance", "candidate": "finance_ssl10", "control": "finance"},
        {
            "name": "finance_ssl10-minus-finance_ssl3",
            "candidate": "finance_ssl10",
            "control": "finance_ssl3",
        },
    ]
)
PACKAGE_METHODS = frozenset(
    row[arm] for row in STATIC_COMPARISONS for arm in ("candidate", "control")
)
METHOD_BINDINGS = {
    "linear": {"source": "reference", "candidate": "statistics_linear"},
    "mlp01": {"source": "reference", "candidate": "statistics_mlp"},
    "mlp03": {"source": "reference", "candidate": "statistics_mlp_dropout03"},
    "finance": {"source": "reference", "candidate": "finance"},
    "conditional_linear": {"source": "conditional", "candidate": "statistics_conditional_linear"},
    "mlp03_lr03": {"source": "low_lr", "candidate": "statistics_mlp_dropout03"},
    "finance_lr03": {"source": "low_lr", "candidate": "finance"},
}
PRIMARY_PAIRS = (
    "conditional_linear-minus-linear",
    "mlp03_lr03-minus-mlp03",
    "finance_lr03-minus-finance",
)
ABLATION_METHOD_BINDINGS = {
    **METHOD_BINDINGS,
    "mlp03_no_aux": {"source": "no_aux", "candidate": "statistics_mlp_dropout03"},
    "finance_no_aux": {"source": "no_aux", "candidate": "finance"},
    "finance_no_stock_interaction": {"source": "no_stock", "candidate": "finance"},
    "mlp03_epochs30": {"source": "epochs30", "candidate": "statistics_mlp_dropout03"},
    "finance_epochs30": {"source": "epochs30", "candidate": "finance"},
}
ABLATION_PAIRS = PRIMARY_PAIRS + (
    "mlp03_no_aux-minus-mlp03", "finance_no_aux-minus-finance",
    "finance_no_stock_interaction-minus-finance",
    "mlp03_epochs30-minus-mlp03", "finance_epochs30-minus-finance",
)


def comparison_methods(sources: dict) -> dict:
    if set(sources) == {"reference", "conditional", "low_lr"}:
        return METHOD_BINDINGS
    if set(sources) == {"reference", "conditional", "low_lr", "no_aux", "no_stock", "epochs30"}:
        return ABLATION_METHOD_BINDINGS
    raise DataContractError("comparison requires a complete declared intervention batch")


def validate_intervention_plans(plans: dict) -> None:
    methods = comparison_methods(plans)
    reference = plans["reference"]["identity"]
    for role, plan in plans.items():
        identity = plan["identity"]
        research = identity["research"]
        expected_candidates = [
            v["candidate"] for v in methods.values() if v["source"] == role
        ]
        if (
            research["candidates"] != expected_candidates
            or research["seeds"] != [17, 42, 73, 101, 137]
            or [row["fold_id"] for row in research["folds"]] != ["wf1", "wf2", "wf3"]
            or research["folds"] != reference["research"]["folds"]
            or identity["folds"] != reference["folds"]
            or identity["observation"] != reference["observation"]
        ):
            raise DataContractError(
                "intervention matrix, fold inputs or observation protocol differs"
            )
        expected = json.loads(json.dumps(reference["supervised_config"]))
        if role == "low_lr":
            for field in ("encoder_learning_rate", "head_learning_rate"):
                expected["training"][field] *= 0.3
                if (
                    abs(
                        expected["training"][field]
                        - identity["supervised_config"]["training"][field]
                    )
                    > 1e-15
                ):
                    raise DataContractError("low LR must be exactly the declared 0.3 strategy")
                expected["training"][field] = identity["supervised_config"]["training"][field]
        if role in {"no_aux", "no_stock"}:
            expected["research_ablation"] = (
                "no_auxiliary_rank" if role == "no_aux" else "no_stock_interaction"
            )
        if role == "epochs30":
            expected["training"].update(max_epochs=30, minimum_epochs=30)
        if identity["supervised_config"] != expected:
            raise DataContractError("undeclared scientific configuration difference")


def freeze_intervention_comparison(
    sources: dict[str, Path],
    output: Path,
    *,
    repository_root: Path,
) -> dict:
    git = clean_git(repository_root)
    if output.exists():
        raise DataContractError("comparison must use a new directory")
    for path in sources.values():
        disjoint_output(output, path)
    plans = {role: read_experiment_plan(path) for role, path in sources.items()}
    validate_intervention_plans(plans)
    hashes = {}
    protected = ["reference"]
    if "no_aux" in plans:
        protected += ["conditional", "low_lr"]
    for role in protected:
        for cell in plans[role]["identity"]["cells"]:
            read_completed_cell(sources[role], plans[role], cell, hashes)
    identity = {
        "analysis_commit": git["commit"],
        "sources": {
            role: {
                "path": str(path.resolve()),
                "plan_hash": plans[role]["plan_hash"],
                "training_commit": plans[role]["identity"]["code_commit"],
            }
            for role, path in sources.items()
        },
        "methods": comparison_methods(sources),
        "static_comparisons": STATIC_COMPARISONS,
        "primary_fs_comparisons": list(ABLATION_PAIRS if "no_aux" in plans else PRIMARY_PAIRS),
        "reference_files": hashes,
        "validation_policy": {
            "required_methods": sorted(PACKAGE_METHODS),
            "required_supervised_cells": 210,
            "required_ssl_stages": 30,
            "holm_family_size": 20,
            "practical_delta": 0.001,
            "positive_folds": 2,
            "positive_cross_fold_seeds": 4,
            "worst_fold_positive": True,
            "hac_lags": [5, 20, 60],
            "nonoverlap_stride": 5,
            "nonoverlap_offsets": [0, 1, 2, 3, 4],
            "bootstrap_blocks": [20, 60],
        },
    }
    result = {
        "identity": identity,
        "comparison_hash": sha256_json(identity),
        "validation_used": False,
        "holdout_used": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "comparison.json", result)
    return result


def read_comparison(output: Path) -> tuple[dict, dict]:
    comparison = json.loads((output / "comparison.json").read_text())
    identity = comparison["identity"]
    if (
        comparison["comparison_hash"] != sha256_json(identity)
        or identity["static_comparisons"] != STATIC_COMPARISONS
        or identity["methods"] != comparison_methods(identity["sources"])
        or identity["primary_fs_comparisons"] != list(
            ABLATION_PAIRS if "no_aux" in identity["sources"] else PRIMARY_PAIRS
        )
        or comparison.get("validation_used") is not False
        or comparison.get("holdout_used") is not False
    ):
        raise DataContractError("frozen comparison binding changed")
    plans = {}
    for role, binding in identity["sources"].items():
        plan = read_experiment_plan(Path(binding["path"]))
        if (
            plan["plan_hash"] != binding["plan_hash"]
            or plan["identity"]["code_commit"] != binding["training_commit"]
        ):
            raise DataContractError("comparison source plan changed")
        plans[role] = plan
    validate_intervention_plans(plans)
    for path, expected in identity["reference_files"].items():
        if sha256_file(Path(path)) != expected:
            raise DataContractError("protected reference artifact changed")
    return comparison, plans


def summarize_comparison(comparison_root: Path, output: Path, *, repository_root: Path) -> dict:
    comparison, _ = read_comparison(comparison_root)
    identity = comparison["identity"]
    pairs = [
        {
            "name": row["name"],
            **{arm: identity["methods"][row[arm]] for arm in ("candidate", "control")},
        }
        for row in STATIC_COMPARISONS
        if row["name"] in identity["primary_fs_comparisons"]
    ]
    disjoint_output(output, comparison_root)
    return summarize_experiments(
        {role: Path(binding["path"]) for role, binding in identity["sources"].items()},
        output,
        repository_root=repository_root,
        comparisons=pairs,
    )


def seal_comparison_selections(comparison_root: Path, *, repository_root: Path) -> dict:
    """Seal native selections in the NEW comparison directory, never in controls."""
    git = clean_git(repository_root)
    comparison, plans = read_comparison(comparison_root)
    hashes, selections, environments = {}, {}, set()
    for role, plan in plans.items():
        source = Path(comparison["identity"]["sources"][role]["path"])
        for cell in plan["identity"]["cells"]:
            audit, _ = read_completed_cell(source, plan, cell, hashes)
            environments.add(sha256_json(audit["identity"]["execution"]))
            selections[f"{role}/{cell['cell_id']}"] = {
                "best_epoch": audit["best_epoch"],
                "native_selection_score": audit["best_selection_score"],
                "best_checkpoint_sha256": audit["artifacts"]["checkpoints/best.pt"],
            }
    if len(environments) != 1:
        raise DataContractError("comparison selection environments differ")
    seal = {
        "comparison_hash": comparison["comparison_hash"],
        "analysis_commit": git["commit"],
        "selections": selections,
        "files": hashes,
        "validation_used": False,
        "holdout_used": False,
    }
    path = comparison_root / "selection-seal.json"
    if path.exists() and json.loads(path.read_text()) != seal:
        raise DataContractError("comparison selection seal changed")
    if not path.exists():
        write_json(path, seal)
    return seal


def require_complete_validation_package(comparison_root: Path) -> None:
    """Check the full family BEFORE a caller can read any V data. No subset override."""
    comparison, _ = read_comparison(comparison_root)
    missing = sorted(PACKAGE_METHODS - set(comparison["identity"]["methods"]))
    if missing:
        raise DataContractError(
            "V remains closed; missing original package methods: " + ", ".join(missing)
        )
    # Future arms need their scientific implementation and a new frozen full-package binding.
    # Neither the 105- nor 180-cell binding is a complete 210-cell declaration.
    raise DataContractError("V requires a separately frozen complete 210-cell/30-SSL package")
