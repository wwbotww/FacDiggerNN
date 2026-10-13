from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from test_finance_complete_experiments import last, setup_cells
from test_finance_diagnostic_prefix import _cached
from test_finance_transformer_training import _datasets

from facdigger.data.contracts import DataContractError
from facdigger.research import finance_comparison as comparison
from facdigger.research.finance_experiment_config import (
    experiment_cells,
    load_finance_experiment_config,
)
from facdigger.research.finance_experiment_state import validate_checkpoint
from facdigger.research.finance_experiment_summary import summarize_experiments
from facdigger.training.finance_transformer_config import load_finance_transformer_config

ROOT = Path(__file__).resolve().parents[2]


def intervention_plans():
    plans = {}
    for role, name in (
        ("reference", "finance_complete_reference"),
        ("conditional", "finance_market_conditional"),
        ("low_lr", "finance_low_lr"),
    ):
        config = load_finance_experiment_config(ROOT / f"configs/research/{name}.yaml")
        base = load_finance_transformer_config(config.supervised_config)
        plans[role] = {
            "identity": {
                "research": config.model_dump(mode="json"),
                "supervised_config": base.model_dump(mode="json"),
                "cells": experiment_cells(config),
                "folds": {
                    fold.fold_id: {"data_protocol": {"fixture": fold.fold_id}}
                    for fold in config.folds
                },
                "observation": {"phases": ["F", "S"], "precision": "fp32", "full_fit": True},
            }
        }
    return plans


def test_frozen_intervention_matrices_and_full_comparison_family():
    plans = intervention_plans()
    assert [len(plans[role]["identity"]["cells"]) for role in plans] == [60, 15, 30]
    comparison.validate_intervention_plans(plans)
    assert len(comparison.STATIC_COMPARISONS) == 20
    assert len({row["name"] for row in comparison.STATIC_COMPARISONS}) == 20
    assert len(comparison.PACKAGE_METHODS) == 14
    assert len(comparison.METHOD_BINDINGS) == 7
    assert set(comparison.PRIMARY_PAIRS).issubset(
        {row["name"] for row in comparison.STATIC_COMPARISONS}
    )


@pytest.mark.parametrize("change", ["lr", "objective", "seed", "fold", "data", "candidate"])
def test_comparison_rejects_undeclared_changes(change):
    plans = intervention_plans()
    arm = plans["low_lr"]["identity"]
    if change == "lr":
        arm["supervised_config"]["training"]["head_learning_rate"] *= 2
    elif change == "objective":
        arm["supervised_config"]["training"]["objective"]["scale_regularization"] = 0
    elif change == "seed":
        arm["research"]["seeds"].pop()
    elif change == "fold":
        arm["research"]["folds"].pop()
    elif change == "data":
        arm["folds"]["wf3"]["data_protocol"]["fixture"] = "different"
    else:
        arm["research"]["candidates"].pop()
    with pytest.raises(DataContractError):
        comparison.validate_intervention_plans(plans)


def test_real_v_gate_refuses_partial_package_before_any_target_read(tmp_path, monkeypatch):
    bound = {"identity": {"methods": comparison.METHOD_BINDINGS}}
    monkeypatch.setattr(comparison, "read_comparison", lambda *a: (bound, {}))
    with pytest.raises(DataContractError, match="missing original package methods"):
        comparison.require_complete_validation_package(tmp_path)


@pytest.mark.parametrize("failure", ["missing", "nonfinite", "wrong", "channel_order"])
def test_conditional_cache_does_not_substitute_an_earlier_market_observation(tmp_path, failure):
    train, _ = _datasets()
    data = _cached(train, tmp_path / "cache")
    store = copy.deepcopy(train.market_store)
    assert data.verify_current_market(store)["exact_float32_match"]
    day = np.datetime64(data.sample_rows["asof_date"][0], "D")
    index = int(np.searchsorted(store.dates, day))
    assert store.observed[index - 1, 0]
    if failure == "missing":
        store.observed[index, 0] = False
    elif failure == "nonfinite":
        store.values[index, 0] = np.nan
    elif failure == "wrong":
        store.values[index, 0] += 1
    else:
        store.channels = tuple(reversed(store.channels))
    with pytest.raises(DataContractError, match="current market"):
        data.verify_current_market(store)


def test_low_lr_uses_same_initial_weights_clock_and_scaled_group_trajectory(tmp_path, monkeypatch):
    records = []
    for role, ratio in (("base", 1), ("low", 0.3)):
        folder = tmp_path / role
        folder.mkdir()
        setup = setup_cells(folder, monkeypatch, "finance", lr_ratio=ratio)
        root = setup["plan"]("run")
        setup["run"](root)
        cell = last(root, "finance").parents[1]
        weights = torch.load(cell / "observations/epoch-0.pt", weights_only=False)["model_state"]
        state = torch.load(last(root, "finance"), weights_only=False)
        records.append((setup, weights, state))
    base, low = records
    for key, tensor in base[1].items():
        assert torch.equal(tensor, low[1][key])
    assert base[2]["global_step"] == low[2]["global_step"]
    assert base[2]["scheduler_state"]["last_epoch"] == low[2]["scheduler_state"]["last_epoch"]
    for a, b in zip(
        base[2]["optimizer_state"]["param_groups"],
        low[2]["optimizer_state"]["param_groups"],
        strict=True,
    ):
        assert b["lr"] == pytest.approx(a["lr"] * 0.3)
        assert b["initial_lr"] == pytest.approx(a["initial_lr"] * 0.3)
        assert a["weight_decay"] == b["weight_decay"]
    with pytest.raises(DataContractError, match="protocol differs"):
        validate_checkpoint(base[2], low[0]["config"], low[0]["protocol"], "finance")


def test_summary_reuses_committed_results_and_preserves_sources(tmp_path, monkeypatch):
    setup = setup_cells(tmp_path, monkeypatch, epochs=3)
    root = setup["plan"]("source")
    setup["run"](root)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}

    def forbidden(*a, **kw):
        raise AssertionError("summary must not read targets or train")

    from facdigger.research import finance_experiment_plan as planning
    from facdigger.training import finance_transformer_engine as engine

    monkeypatch.setattr(planning, "_fold_inputs", forbidden)
    monkeypatch.setattr(engine, "train_finance_transformer", forbidden)
    result = summarize_experiments({"old": root}, tmp_path / "summary", repository_root=tmp_path)
    assert result["cells"] == 1 and result["common_epoch_range"] == [0, 3]
    assert result["validation_used"] is False and result["holdout_used"] is False
    curves = pl.read_csv(tmp_path / "summary/curves.csv")
    assert curves["epoch"].unique().sort().to_list() == [0, 1, 2, 3]
    assert curves["seeds_present"].to_list() == [1] * 24
    assert {path: path.read_bytes() for path in before} == before
    assert not (root / "selection-seal.json").exists()
    # The source best is hash-bound, even if a summary would only use daily metrics.
    (last(root).parent / "best.pt").write_bytes(b"changed")
    with pytest.raises(DataContractError, match="checksum differs"):
        summarize_experiments({"old": root}, tmp_path / "tampered", repository_root=tmp_path)
    assert not (tmp_path / "tampered").exists()


def test_summary_rejects_successful_date_intersection(tmp_path, monkeypatch):
    setup = setup_cells(tmp_path, monkeypatch)
    root = setup["plan"]("source")
    setup["run"](root)
    original = pl.read_parquet

    def partial(path, *args, **kwargs):
        table = original(path, *args, **kwargs)
        return table.slice(3) if str(path).endswith("epoch-1-S-daily.parquet") else table

    monkeypatch.setattr(pl, "read_parquet", partial)
    with pytest.raises(DataContractError, match="incomplete F/S daily support"):
        summarize_experiments({"old": root}, tmp_path / "summary", repository_root=tmp_path)


def test_seal_refuses_missing_cells_and_changed_control_before_writing(tmp_path, monkeypatch):
    setup = setup_cells(tmp_path, monkeypatch)
    root = setup["plan"]("source")
    setup["run"](root)
    from facdigger.research.finance_experiment_plan import read_experiment_plan

    plan = read_experiment_plan(root)
    bound = {
        "comparison_hash": "fixture",
        "identity": {"sources": {"reference": {"path": str(root)}}},
    }
    monkeypatch.setattr(comparison, "read_comparison", lambda *a: (bound, {"reference": plan}))
    output = tmp_path / "comparison"
    output.mkdir()
    seal = comparison.seal_comparison_selections(output, repository_root=tmp_path)
    assert len(seal["selections"]) == 1
    assert not (root / "selection-seal.json").exists()
    audit_path = last(root).parents[1] / "cell.json"
    audit = json.loads(audit_path.read_text())
    audit["status"] = "running"
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(DataContractError, match="complete cells"):
        comparison.seal_comparison_selections(output, repository_root=tmp_path)


def test_admission_checks_saved_epoch_zero_and_current_market_without_updates(
    tmp_path, monkeypatch
):
    from test_finance_complete_experiments import ENVIRONMENT

    from facdigger.research import finance_intervention_admission as admission
    from facdigger.research.finance_experiment_plan import read_experiment_plan

    setup = setup_cells(tmp_path, monkeypatch, "statistics_linear")
    root = setup["plan"]("reference")
    setup["run"](root)
    plan = read_experiment_plan(root)
    bound = {
        "comparison_hash": "fixture",
        "identity": {
            "sources": {
                "reference": {"path": str(root)},
                "conditional": {"path": str(tmp_path / "new")},
            }
        },
    }
    monkeypatch.setattr(
        admission,
        "read_comparison",
        lambda *a: (
            bound,
            {
                "reference": plan,
                "conditional": plan,
            },
        ),
    )
    monkeypatch.setattr(admission, "collect_environment", lambda: ENVIRONMENT)
    result = admission.admit_interventions(
        tmp_path / "comparison",
        {"wf": setup["cache"]},
        tmp_path / "proof",
        repository_root=tmp_path,
    )
    assert result["optimizer_updates"] == 0 and result["validation_used"] is False
    proof = json.loads((tmp_path / "proof/admission.json").read_text())
    assert len(proof["epoch_zero"]) == 2
    assert proof["market"]["wf"]["F"]["exact_float32_match"]
