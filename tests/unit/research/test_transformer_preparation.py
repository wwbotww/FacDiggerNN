from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from facdigger.data.contracts import DataContractError
from facdigger.experiments.manifest import sha256_json
from facdigger.research.transformer_config import load_transformer_comparison_config
from facdigger.research.transformer_snapshots import prepare_transformer_snapshots
from facdigger.training.runtime import (
    TrainingRuntimeConfig,
    load_training_runtime,
    run_lock,
    write_json,
)

REPOSITORY = Path(__file__).resolve().parents[3]


def setup_builder(tmp_path, monkeypatch):
    config = load_transformer_comparison_config(
        REPOSITORY / "configs/research/finance_transformer_streamlined.yaml"
    ).model_copy(
        update={
            "base_dataset_config": REPOSITORY
            / "configs/datasets/eodhd_historical_liquid_transformer.yaml",
            "snapshot_output_root": tmp_path / "snapshots",
        }
    )
    calls = []
    fail = [None]
    revision = ["original"]

    def build(dataset):
        calls.append(dataset.split.train_end)
        if dataset.split.train_end == fail[0]:
            raise RuntimeError("CPU job interrupted during feature building")
        identity = {
            "schema_version": 4,
            "config": dataset.model_dump(mode="json", exclude={"sources", "output_root"}),
            "input_file_hashes": {"bars": revision[0]},
        }
        dataset_id = sha256_json(identity)
        manifest = {**identity, "dataset_id": dataset_id}
        path = dataset.output_root / dataset_id
        write_json(path / "manifest.json", manifest)
        (path / "features.parquet").write_bytes(b"complete immutable snapshot")
        return path, manifest

    monkeypatch.setattr("facdigger.research.transformer_snapshots.build_dataset_snapshot", build)
    return config, calls, fail, revision


def test_partial_prepare_resumes_without_rebuilding_completed_folds(tmp_path, monkeypatch):
    config, calls, fail, _ = setup_builder(tmp_path, monkeypatch)
    output = tmp_path / "prepared"
    runtime = TrainingRuntimeConfig(checkpoint_interval_seconds=600, handle_signals=True)
    fail[0] = config.folds[1].train_end
    with pytest.raises(RuntimeError, match="interrupted"):
        prepare_transformer_snapshots(config, output, runtime)
    state = json.loads((output / "preparation.json").read_text())
    assert state["status"] == "preparing" and set(state["locations"]) == {"wf1"}
    assert not (output / "runtime.yaml").exists()
    fail[0] = None
    calls.clear()
    report = prepare_transformer_snapshots(config, output, runtime)
    assert calls == [fold.train_end for fold in config.folds[1:]]
    generated = load_training_runtime(Path(report["runtime"]))
    assert generated.checkpoint_interval_seconds == 600 and generated.handle_signals
    assert set(generated.fold_snapshots) == {"wf1", "wf2", "wf3"}
    mtimes = {path: path.stat().st_mtime_ns for path in output.iterdir() if path.suffix}
    calls.clear()
    assert prepare_transformer_snapshots(config, output, runtime) == report
    assert calls == []
    assert mtimes == {path: path.stat().st_mtime_ns for path in mtimes}


@pytest.mark.parametrize("damage", ["snapshot", "checksums", "runtime", "protocol"])
def test_repeat_preparation_rejects_changes_without_reblessing_files(tmp_path, monkeypatch, damage):
    config, _, _, _ = setup_builder(tmp_path, monkeypatch)
    output = tmp_path / "prepared"
    report = prepare_transformer_snapshots(config, output)
    if damage == "snapshot":
        (Path(report["folds"][0]["dataset_path"]) / "features.parquet").write_bytes(b"corrupt")
    elif damage == "checksums":
        (output / "wf1.checksums.json").unlink()
    elif damage == "runtime":
        (output / "runtime.yaml").write_text("{}")
    else:
        config = config.model_copy(update={"research_id": "changed"})
    with pytest.raises((DataContractError, FileNotFoundError)):
        prepare_transformer_snapshots(config, output)


def test_prepare_rejects_new_source_revision_after_interruption(tmp_path, monkeypatch):
    config, _, fail, revision = setup_builder(tmp_path, monkeypatch)
    fail[0] = config.folds[1].train_end
    output = tmp_path / "prepared"
    with pytest.raises(RuntimeError):
        prepare_transformer_snapshots(config, output)
    fail[0] = None
    revision[0] = "downloaded again"
    with pytest.raises(DataContractError, match="same source"):
        prepare_transformer_snapshots(config, output)
    assert not (output / "runtime.yaml").exists()


def test_prepare_lock_and_output_cannot_modify_snapshots(tmp_path, monkeypatch):
    config, calls, _, _ = setup_builder(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="outside immutable"):
        prepare_transformer_snapshots(config, config.snapshot_output_root / "oops")
    assert not config.snapshot_output_root.exists()
    output = tmp_path / "prepared"
    with run_lock(output / ".prepare.lock"), pytest.raises(RuntimeError, match="already"):
        prepare_transformer_snapshots(config, output)
    assert not calls


def test_prepare_module_does_not_import_training_model_dependencies():
    code = """
import sys
import facdigger.research.transformer_snapshots
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
