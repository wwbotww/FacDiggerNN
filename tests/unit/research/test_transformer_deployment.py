from __future__ import annotations

from pathlib import Path

import pytest

from facdigger.data.config import load_dataset_build_config
from facdigger.data.contracts import DataContractError
from facdigger.experiments.manifest import sha256_json
from facdigger.research.transformer_config import load_transformer_comparison_config
from facdigger.research.transformer_runner import _load_resource_admission, _snapshots
from facdigger.training.finance_pretrain_config import load_finance_pretraining_config
from facdigger.training.finance_transformer_config import load_finance_transformer_config
from facdigger.training.resources import TrainingResourceBudget
from facdigger.training.runtime import (
    DatasetLocation,
    TrainingRuntimeConfig,
    write_json,
    write_snapshot_checksums,
)

REPOSITORY = Path(__file__).resolve().parents[3]


def _config(tmp_path):
    config = load_transformer_comparison_config(
        REPOSITORY / "configs/research/finance_transformer_streamlined.yaml"
    )
    return config.model_copy(
        update={
            "base_dataset_config": REPOSITORY / config.base_dataset_config,
            "snapshot_output_root": tmp_path / "snapshots",
            "admission_report": tmp_path / "admission.json",
        }
    )


@pytest.mark.parametrize("damage", [None, "protocol", "identity", "files", "missing_fold"])
def test_prebuilt_folds_need_no_bronze_and_reject_changed_inputs(tmp_path, monkeypatch, damage):
    config = _config(tmp_path)
    base = load_dataset_build_config(config.base_dataset_config)
    locations = {}
    for fold in config.folds:
        protocol = base.model_dump(mode="json", exclude={"sources", "output_root"})
        protocol["split"] = fold.model_dump(mode="json", exclude={"fold_id"})
        if damage == "protocol" and fold.fold_id == "wf1":
            protocol["split"]["embargo_sessions"] += 1
        identity = {"schema_version": 4, "config": protocol, "input_file_hashes": {"fixture": "a"}}
        dataset_id = sha256_json(identity)
        if damage == "identity" and fold.fold_id == "wf1":
            dataset_id = "incorrect_identity"
        path = tmp_path / dataset_id
        write_json(path / "manifest.json", {**identity, "dataset_id": dataset_id})
        (path / "features.parquet").write_bytes(b"immutable data")
        checksums = tmp_path / f"{fold.fold_id}.checksums.json"
        write_snapshot_checksums(path, checksums)
        if damage == "files" and fold.fold_id == "wf1":
            (path / "features.parquet").write_bytes(b"changed after transfer")
        locations[fold.fold_id] = DatasetLocation(path=path, checksums=checksums)
    if damage == "missing_fold":
        del locations["wf2"]
    monkeypatch.setattr(
        "facdigger.research.transformer_runner.build_dataset_snapshot",
        lambda *_: pytest.fail("prebuilt folds must not load bronze or rebuild features"),
    )
    runtime = TrainingRuntimeConfig(fold_snapshots=locations)
    if damage:
        with pytest.raises(DataContractError):
            _snapshots(config, runtime)
    else:
        plans = _snapshots(config, runtime)
        assert [plan["fold_id"] for plan in plans] == ["wf1", "wf2", "wf3"]
        assert [plan["dataset_path"] for plan in plans] == [
            str(location.path) for location in locations.values()
        ]


@pytest.mark.parametrize(
    "change", [None, "missing_budget", "budget", "hardware", "ram", "gpu", "time", "fp16"]
)
def test_explicit_resource_admission_binds_budget_device_and_current_allocation(
    tmp_path, monkeypatch, change
):
    config = _config(tmp_path)
    supervised = load_finance_transformer_config(REPOSITORY / config.experiments.scratch)
    pretraining = load_finance_pretraining_config(REPOSITORY / config.experiments.pretraining)
    budget = TrainingResourceBudget(projected_days=20, host_peak_rss_bytes=20 * 1024**3)
    hardware = {
        "device": "cuda",
        "device_name": "allocated MIG",
        "device_memory_bytes": 16 * 1024**3,
    }
    monkeypatch.setattr("facdigger.research.transformer_runner.training_hardware", lambda: hardware)
    monkeypatch.setattr("facdigger.training.resources.cgroup_memory_limit", lambda: 8 * 1024**3)
    payload = {
        "benchmark_optimizer_updates": 100,
        "dataset_id": "largest-fold",
        "supervised_config_hash": sha256_json(supervised.model_dump(mode="json")),
        "pretraining_config_hash": sha256_json(pretraining.model_dump(mode="json")),
        "matrix_projection": {"pretraining_runs": 3, "supervised_cells": 6, "projected_days": 18},
        "resource_budget": budget.model_dump(mode="json"),
        "hardware": dict(hardware),
        "host_peak_rss_bytes": 4 * 1024**3,
        "supervised": {"cuda_peak_reserved_bytes": 4 * 1024**3},
        "pretraining": {"cuda_peak_reserved_bytes": 4 * 1024**3},
        "admission": {
            "cuda_fp16_verified": True,
            "within_memory_budget": True,
            "within_fourteen_days": False,
            "within_time_budget": True,
            "admitted": True,
        },
    }
    if change == "missing_budget":
        budget = None
    elif change == "budget":
        payload["resource_budget"]["projected_days"] = 30
    elif change == "hardware":
        payload["hardware"]["device_name"] = "another GPU"
    elif change == "ram":
        payload["host_peak_rss_bytes"] = 9 * 1024**3
    elif change == "gpu":
        payload["supervised"]["cuda_peak_reserved_bytes"] = 10 * 1024**3
    elif change == "time":
        payload["matrix_projection"]["projected_days"] = 21
    elif change == "fp16":
        payload["admission"]["cuda_fp16_verified"] = False
    write_json(config.admission_report, payload)
    kwargs = dict(supervised=supervised, pretraining=pretraining, resource_budget=budget)
    if change:
        with pytest.raises(DataContractError):
            _load_resource_admission(config, **kwargs)
    else:
        assert _load_resource_admission(config, **kwargs) == payload
