from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import build_dataset_snapshot, sha256_file
from facdigger.experiments.manifest import sha256_json
from facdigger.research import transformer_runner as runner
from facdigger.research.transformer_config import load_transformer_comparison_config
from facdigger.training.finance_data import finance_data_protocol
from facdigger.training.finance_pretrain_config import FinancePretrainingExperimentConfig
from facdigger.training.finance_transformer_config import FinanceTransformerExperimentConfig
from facdigger.training.runtime import TrainingRuntimeConfig, write_json
from tests.integration.test_finance_selection_snapshot import finance_config

REPOSITORY = Path(__file__).resolve().parents[3]


def _bytes(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and not path.name.endswith(".lock")
    }


def _change_child_budget_consistently(record):
    child = Path(record["run_dir"])
    raw = yaml.safe_load((child / "resolved_config.yaml").read_text())
    raw["training"]["max_epochs"] += 1
    (child / "resolved_config.yaml").write_text(yaml.safe_dump(raw))
    config_payload = FinanceTransformerExperimentConfig.model_validate(raw).model_dump(mode="json")
    payload = json.loads((child / "manifest.json").read_text())
    payload["config_hash"] = sha256_json(config_payload)
    checkpoint = child / payload["checkpoint"]["file"]
    envelope = torch.load(checkpoint, map_location="cpu", weights_only=False)
    envelope["protocol_hash"] = sha256_json(
        {
            "config": config_payload,
            "data_protocol": payload["data_protocol"],
        }
    )
    torch.save(envelope, checkpoint)
    payload["checkpoint"]["sha256"] = sha256_file(checkpoint)
    write_json(child / "manifest.json", payload)
    record["manifest_sha256"] = sha256_file(child / "manifest.json")


@pytest.fixture
def matrix_environment(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    dataset, dataset_manifest = build_dataset_snapshot(finance_config(source))
    root = tmp_path / "research"
    root.mkdir()
    scratch = FinanceTransformerExperimentConfig.model_validate(
        {
            "model": {"statistics_windows": [5, 20]},
        }
    )
    pretrained = scratch.model_copy(
        update={
            "initialization": "finance_pretrained",
            "pretrained_checkpoint": tmp_path / "template-encoder.pt",
        }
    )
    pretraining = FinancePretrainingExperimentConfig.model_validate(
        {
            "model": {"statistics_windows": [5, 20]},
            "training": {"probe": {"fit_dates": 5, "selection_dates": 4}},
        }
    )
    templates = {"scratch": scratch, "pretrained": pretrained, "pretraining": pretraining}
    paths = {}
    for stage, template in templates.items():
        path = tmp_path / f"{stage}.yaml"
        path.write_text(yaml.safe_dump(template.model_dump(mode="json")))
        paths[stage] = path
    config = load_transformer_comparison_config(
        REPOSITORY / "configs/research/finance_transformer_streamlined.yaml",
    )
    config = config.model_copy(
        update={
            "experiments": config.experiments.model_validate(paths),
            "admission_report": tmp_path / "admission.json",
        }
    )
    plans = [
        {
            "fold_id": fold.fold_id,
            "split": fold.model_dump(mode="json", exclude={"fold_id"}),
            "dataset_path": str(dataset),
            "dataset_id": dataset_manifest["dataset_id"],
            "dataset_manifest_sha256": sha256_file(dataset / "manifest.json"),
        }
        for fold in config.folds
    ]
    matrix = {"stages": []}
    for plan in plans:
        for stage in ("pretraining", "scratch", "finance_pretrained"):
            output = root / "runs" / plan["fold_id"] / stage
            child = output / "run"
            template = templates["pretrained" if stage == "finance_pretrained" else stage]
            update = {"output_root": output, "seed": 42}
            if stage == "finance_pretrained":
                update["pretrained_checkpoint"] = (
                    root
                    / "runs"
                    / plan["fold_id"]
                    / "pretraining"
                    / "run"
                    / "checkpoints"
                    / "best_encoder.pt"
                )
            experiment = template.model_copy(update=update)
            checkpoint = (
                child / "checkpoints" / ("best_encoder.pt" if stage == "pretraining" else "best.pt")
            )
            checkpoint.parent.mkdir(parents=True)
            protocol = finance_data_protocol(dataset, dataset_manifest, experiment)
            torch.save(
                {
                    "contract": "finance_patch_pretrain_encoder"
                    if stage == "pretraining"
                    else "finance_patch_transformer_checkpoint",
                    "dataset_id": plan["dataset_id"],
                    "data_protocol": protocol,
                    "protocol_hash": sha256_json(
                        {
                            "config": experiment.model_dump(mode="json"),
                            "data_protocol": protocol,
                        }
                    ),
                },
                checkpoint,
            )
            (child / "resolved_config.yaml").write_text(
                yaml.safe_dump(experiment.model_dump(mode="json"))
            )
            payload = {
                "status": "complete",
                "model_type": "finance_patch_pretrain"
                if stage == "pretraining"
                else "finance_patch_transformer",
                "config_hash": sha256_json(experiment.model_dump(mode="json")),
                "dataset_id": plan["dataset_id"],
                "dataset_manifest_hash": plan["dataset_manifest_sha256"],
                "data_protocol": protocol,
                "checkpoint": {
                    "file": str(checkpoint.relative_to(child)),
                    "sha256": sha256_file(checkpoint),
                },
                "artifacts": {"resolved_config": "resolved_config.yaml"},
            }
            write_json(child / "manifest.json", payload)
            matrix["stages"].append(
                {
                    "fold_id": plan["fold_id"],
                    "stage": stage,
                    "status": "complete",
                    "seed": 42,
                    "run_dir": str(child),
                    "manifest_sha256": sha256_file(child / "manifest.json"),
                }
            )
    admission = {
        "dataset_id": dataset_manifest["dataset_id"],
        "data_protocols": {
            "supervised": finance_data_protocol(dataset, dataset_manifest, scratch),
            "pretraining": finance_data_protocol(dataset, dataset_manifest, pretraining),
        },
    }
    write_json(config.admission_report, admission)
    write_json(root / "comparison.json", {"acceptance": {"status": "no_go"}})
    manifest = {
        "status": "complete",
        "config_hash": sha256_json(config.model_dump(mode="json")),
        "comparison_sha256": sha256_file(root / "comparison.json"),
        "resource_admission": {**admission, "report_sha256": sha256_file(config.admission_report)},
        "attempts": [{"started_at": "original"}],
    }
    write_json(root / "manifest.json", manifest)
    write_json(root / "matrix.json", matrix)
    write_json(root / "folds.json", plans)
    (root / "progress.jsonl").write_text('{"event":"original"}\n')
    return root, config, plans, matrix, admission, scratch, pretraining


@pytest.mark.parametrize("damage", [None, "dataset", "supervised_scaler", "pretraining_plan"])
def test_admission_is_bound_to_actual_largest_fold_inputs(matrix_environment, damage):
    _, _, plans, _, original, scratch, pretraining = matrix_environment
    admission = copy.deepcopy(original)
    if damage == "dataset":
        admission["dataset_id"] = "another-snapshot"
    elif damage == "supervised_scaler":
        admission["data_protocols"]["supervised"]["feature_scaler_sha256"] = "incorrect"
    elif damage == "pretraining_plan":
        admission["data_protocols"]["pretraining"]["selection_plan"]["a_end"] = "1900-01-01"
    kwargs = dict(supervised=scratch, pretraining=pretraining, runtime=TrainingRuntimeConfig())
    if damage is None:
        runner._validate_admission_dataset(admission, plans, **kwargs)
    else:
        with pytest.raises(DataContractError):
            runner._validate_admission_dataset(admission, plans, **kwargs)


def test_existing_child_cannot_validate_its_own_different_template(matrix_environment):
    root, config, plans, matrix, *_ = matrix_environment
    record = matrix["stages"][1]
    _change_child_budget_consistently(record)
    write_json(root / "matrix.json", matrix)
    before = _bytes(root)
    with pytest.raises(DataContractError):
        runner._validate_existing_stage_protocols(
            root,
            matrix,
            plans,
            TrainingRuntimeConfig(),
            config=config,
        )
    assert _bytes(root) == before


@pytest.mark.parametrize(
    "damage", [None, "old_child", "admission_scaler", "missing_stage", "duplicate_stage"]
)
def test_completed_matrix_reentry_validates_all_bindings_without_writes(matrix_environment, damage):
    root, config, _, matrix, *_ = matrix_environment
    if damage == "old_child":
        record = matrix["stages"][0]
        child = Path(record["run_dir"])
        payload = json.loads((child / "manifest.json").read_text())
        payload.pop("data_protocol")
        write_json(child / "manifest.json", payload)
        record["manifest_sha256"] = sha256_file(child / "manifest.json")
    elif damage == "admission_scaler":
        payload = json.loads((root / "manifest.json").read_text())
        payload["resource_admission"]["data_protocols"]["supervised"]["feature_scaler_sha256"] = (
            "incorrect"
        )
        write_json(root / "manifest.json", payload)
    elif damage == "missing_stage":
        matrix["stages"].clear()
    elif damage == "duplicate_stage":
        matrix["stages"].append(copy.deepcopy(matrix["stages"][0]))
    write_json(root / "matrix.json", matrix)
    before = _bytes(root)
    kwargs = dict(repository_root=REPOSITORY, run_dir=root)
    if damage is None:
        assert runner.run_transformer_comparison(config, **kwargs)[0] == root
    else:
        with pytest.raises(DataContractError):
            runner.run_transformer_comparison(config, **kwargs)
    assert _bytes(root) == before


def test_paused_matrix_rejects_changed_child_before_recording_attempt(
    matrix_environment, monkeypatch
):
    root, config, _, matrix, admission, *_ = matrix_environment
    payload = json.loads((root / "manifest.json").read_text())
    payload["status"] = "paused"
    write_json(root / "manifest.json", payload)
    _change_child_budget_consistently(matrix["stages"][1])
    write_json(root / "matrix.json", matrix)
    monkeypatch.setattr(runner, "_load_resource_admission", lambda *a, **k: admission)
    before = _bytes(root)
    with pytest.raises(DataContractError):
        runner.run_transformer_comparison(config, repository_root=REPOSITORY, resume_run=root)
    assert _bytes(root) == before


@pytest.mark.parametrize("damage", ["checkpoint_protocol", "checkpoint_bytes"])
def test_paused_matrix_checks_actual_checkpoint_before_any_parent_or_child_write(
    matrix_environment, monkeypatch, damage
):
    root, config, _, matrix, admission, *_ = matrix_environment
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["status"] = "paused"
    write_json(root / "manifest.json", manifest)
    record = matrix["stages"][0]
    child = Path(record["run_dir"])
    child_manifest = json.loads((child / "manifest.json").read_text())
    checkpoint = child / child_manifest["checkpoint"]["file"]
    if damage == "checkpoint_protocol":
        envelope = torch.load(checkpoint, map_location="cpu", weights_only=False)
        envelope["data_protocol"]["feature_scaler_sha256"] = "another-scaler"
        torch.save(envelope, checkpoint)
        # Every outer checksum is consistent; only the actual checkpoint
        # envelope disagrees with the correctly bound child manifest.
        child_manifest["checkpoint"]["sha256"] = sha256_file(checkpoint)
        write_json(child / "manifest.json", child_manifest)
        record["manifest_sha256"] = sha256_file(child / "manifest.json")
        write_json(root / "matrix.json", matrix)
    else:
        checkpoint.write_bytes(b"damaged actual checkpoint")
    monkeypatch.setattr(runner, "_load_resource_admission", lambda *a, **k: admission)
    before = _bytes(root)
    with pytest.raises(DataContractError):
        runner.run_transformer_comparison(config, repository_root=REPOSITORY, resume_run=root)
    assert _bytes(root) == before
