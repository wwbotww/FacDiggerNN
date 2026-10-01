from __future__ import annotations

import json

import pytest

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.experiments.manifest import sha256_json
from facdigger.training.finance_transformer_config import FinanceTransformerExperimentConfig
from facdigger.training.run_state import training_run
from facdigger.training.runtime import TrainingControl


@pytest.mark.parametrize("status", ["paused", "complete"])
def test_old_data_protocol_is_rejected_before_any_run_write(tmp_path, status):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(json.dumps({"dataset_id": "fixture"}))
    root = tmp_path / "run"
    root.mkdir()
    config = FinanceTransformerExperimentConfig(output_root=tmp_path)
    manifest = {
        "status": status,
        "model_type": "finance_patch_transformer",
        "config_hash": sha256_json(config.model_dump(mode="json")),
        "dataset_id": "fixture",
        "dataset_manifest_hash": sha256_file(dataset / "manifest.json"),
        "attempts": [{"started_at": "original"}],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "progress.jsonl").write_text('{"event":"original"}\n')
    (root / "resolved_config.yaml").write_text("original: true\n")
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(DataContractError, match="data protocol"):
        with training_run(
            config,
            dataset,
            repository_root=tmp_path,
            model_type="finance_patch_transformer",
            control=TrainingControl(),
            resume_from=None,
            run_dir=root,
            data_protocol_loader=lambda path, payload: {"computational_universe": "new"},
        ):
            pytest.fail("old run was reused")
    assert {p.name: p.read_bytes() for p in root.iterdir() if p.name != ".training.lock"} == before


def test_invalid_snapshot_does_not_create_attempt_or_progress(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(json.dumps({"dataset_id": "fixture"}))
    root = tmp_path / "run"

    def reject(path, manifest):
        raise DataContractError("actual scaler crosses probe boundary")

    with pytest.raises(DataContractError, match="scaler crosses"):
        with training_run(
            FinanceTransformerExperimentConfig(output_root=tmp_path),
            dataset,
            repository_root=tmp_path,
            model_type="finance_patch_transformer",
            control=TrainingControl(),
            resume_from=None,
            run_dir=root,
            data_protocol_loader=reject,
        ):
            pytest.fail("invalid snapshot reached training")
    assert not (root / "manifest.json").exists()
    assert not (root / "progress.jsonl").exists()
    assert not (root / "resolved_config.yaml").exists()


@pytest.mark.parametrize("damage", ["missing", "different", "embedded_best"])
def test_checkpoint_protocol_rejection_preserves_prior_attempt(tmp_path, damage):
    import torch

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(json.dumps({"dataset_id": "fixture"}))
    root = tmp_path / "run"
    checkpoint = root / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    config = FinanceTransformerExperimentConfig(output_root=tmp_path)
    protocol = {"dataset_id": "fixture", "computational_universe": "target_free_inference_index"}
    manifest = {
        "status": "paused",
        "model_type": "finance_patch_transformer",
        "config_hash": sha256_json(config.model_dump(mode="json")),
        "dataset_id": "fixture",
        "dataset_manifest_hash": sha256_file(dataset / "manifest.json"),
        "data_protocol": protocol,
        "attempts": [{"started_at": "original"}],
    }
    state = {
        "contract": "finance_patch_transformer_training_resume",
        "dataset_id": "fixture",
        "data_protocol": protocol,
        "protocol_hash": sha256_json(
            {"config": config.model_dump(mode="json"), "data_protocol": protocol}
        ),
    }
    if damage == "missing":
        del state["data_protocol"]
    elif damage == "different":
        state["data_protocol"] = {**protocol, "computational_universe": "sample_index"}
    else:
        state["best_checkpoint"] = {**state, "data_protocol": {}}
    torch.save(state, checkpoint)
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "progress.jsonl").write_text('{"event":"original"}\n')
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(DataContractError, match="checkpoint data protocol"):
        with training_run(
            config,
            dataset,
            repository_root=tmp_path,
            model_type="finance_patch_transformer",
            control=TrainingControl(),
            resume_from=checkpoint,
            run_dir=root,
            data_protocol_loader=lambda *args: protocol,
        ):
            pytest.fail("incompatible checkpoint reached training")
    assert {p: p.read_bytes() for p in before} == before
