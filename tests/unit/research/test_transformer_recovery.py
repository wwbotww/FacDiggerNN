from __future__ import annotations

import json

import pytest

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.experiments.manifest import sha256_json
from facdigger.research.transformer_runner import _run_stage
from facdigger.training.finance_pretrain_config import FinancePretrainingExperimentConfig
from facdigger.training.runtime import TrainingControl, write_json


def _stage(tmp_path, monkeypatch):
    write_json(tmp_path / "manifest.json", {"dataset_id": "fixture"})
    monkeypatch.setattr(
        "facdigger.research.transformer_runner.finance_data_protocol",
        lambda *args: {"computational_universe": "pretraining_index_before_probe_selection"},
    )
    # These tests isolate matrix binding; actual checkpoint envelopes have their
    # own integration/preflight tests.
    monkeypatch.setattr(
        "facdigger.training.run_state.verify_checkpoint_protocol", lambda *a, **k: None
    )
    output = tmp_path / "stages" / "pretraining"
    config = FinancePretrainingExperimentConfig(output_root=output)
    plan = {"dataset_id": "fixture", "dataset_manifest_sha256": "a" * 64}
    record = {"fold_id": "wf1", "stage": "pretraining", "status": "running"}
    matrix = {"stages": [record]}
    return config, plan, record, matrix, output


def _manifest(child, config, plan, status):
    checkpoint = child / "checkpoints" / "best_encoder.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"selected encoder")
    payload = {
        "status": status,
        "model_type": "finance_patch_pretrain",
        "config_hash": sha256_json(config.model_dump(mode="json")),
        "dataset_id": plan["dataset_id"],
        "dataset_manifest_hash": plan["dataset_manifest_sha256"],
        "data_protocol": {"computational_universe": "pretraining_index_before_probe_selection"},
        "checkpoint": {"file": "checkpoints/best_encoder.pt", "sha256": sha256_file(checkpoint)},
        "artifacts": {},
    }
    write_json(child / "manifest.json", payload)
    (child / "resolved_config.yaml").write_text("{}")


def test_running_child_is_bound_before_training_and_reused(tmp_path, monkeypatch):
    config, plan, record, matrix, output = _stage(tmp_path, monkeypatch)
    child = output / "legacy-run"
    _manifest(child, config, plan, "running")
    (child / "checkpoints" / "last.pt").write_bytes(b"committed resume")
    matrix_path = tmp_path / "matrix.json"

    def trainer(experiment, dataset, **kwargs):
        bound = json.loads(matrix_path.read_text())["stages"][0]
        assert bound["run_dir"] == str(child)
        assert kwargs["run_dir"] == child
        assert (child / "checkpoints" / "last.pt").is_file()
        _manifest(child, config, plan, "complete")
        return child, {}

    assert (
        _run_stage(
            record,
            matrix,
            matrix_path,
            output_root=output,
            experiment=config,
            plan=plan,
            dataset_path=tmp_path,
            repository=tmp_path,
            control=TrainingControl(),
            trainer=trainer,
        )
        == child
    )
    assert record["status"] == "complete"


def test_orphan_complete_is_adopted_and_tampering_is_rejected(tmp_path, monkeypatch):
    config, plan, record, matrix, output = _stage(tmp_path, monkeypatch)
    child = output / "run"
    _manifest(child, config, plan, "complete")
    record["run_dir"] = str(child)

    def unexpected(*args, **kwargs):
        pytest.fail("completed stage must not retrain")

    kwargs = dict(
        output_root=output,
        experiment=config,
        plan=plan,
        dataset_path=tmp_path,
        repository=tmp_path,
        control=TrainingControl(),
        trainer=unexpected,
    )
    _run_stage(record, matrix, tmp_path / "matrix.json", **kwargs)
    assert record["status"] == "complete"
    (child / "checkpoints" / "best_encoder.pt").write_bytes(b"corrupt")
    with pytest.raises(DataContractError, match="artifact changed"):
        _run_stage(record, matrix, tmp_path / "matrix.json", **kwargs)


def test_legacy_ambiguous_runs_never_choose_latest(tmp_path, monkeypatch):
    config, plan, record, matrix, output = _stage(tmp_path, monkeypatch)
    for name in ("older", "newer"):
        _manifest(output / name, config, plan, "failed")
    with pytest.raises(DataContractError, match="ambiguous"):
        _run_stage(
            record,
            matrix,
            tmp_path / "matrix.json",
            output_root=output,
            experiment=config,
            plan=plan,
            dataset_path=tmp_path,
            repository=tmp_path,
            control=TrainingControl(),
            trainer=lambda *a, **k: pytest.fail("must reject"),
        )


def test_external_control_is_not_restarted_for_the_matrix(tmp_path, monkeypatch):
    from facdigger.research import transformer_runner
    from facdigger.training.runtime import TrainingControl, TrainingRuntimeConfig

    control = TrainingControl(TrainingRuntimeConfig(max_walltime_seconds=200))
    calls = []

    def run(config, **kwargs):
        calls.append(kwargs["control"])
        return tmp_path / "run", {}

    monkeypatch.setattr(transformer_runner, "_run_transformer_comparison", run)
    transformer_runner.run_transformer_comparison(
        object(),
        repository_root=tmp_path,
        run_dir=tmp_path / "run",
        control=control,
    )
    assert calls == [control]


def test_old_complete_child_rejected_without_rewriting_matrix(tmp_path, monkeypatch):
    config, plan, record, matrix, output = _stage(tmp_path, monkeypatch)
    child = output / "run"
    _manifest(child, config, plan, "complete")
    record["run_dir"] = str(child)
    path = child / "manifest.json"
    payload = json.loads(path.read_text())
    del payload["data_protocol"]
    write_json(path, payload)
    matrix_path = tmp_path / "matrix.json"
    write_json(matrix_path, matrix)
    before = {path: path.read_bytes(), matrix_path: matrix_path.read_bytes()}
    with pytest.raises(DataContractError, match="protocol or dataset differs"):
        _run_stage(
            record,
            matrix,
            matrix_path,
            output_root=output,
            experiment=config,
            plan=plan,
            dataset_path=tmp_path,
            repository=tmp_path,
            control=TrainingControl(),
            trainer=lambda *a, **k: pytest.fail("must reject"),
        )
    assert {p: p.read_bytes() for p in before} == before
