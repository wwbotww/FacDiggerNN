from __future__ import annotations

import json

import pytest
import torch
from test_finance_factor_delivery import finance_delivery as _delivery_fixture
from typer.testing import CliRunner

from facdigger.cli import app

finance_delivery = _delivery_fixture


def test_pause_resume_and_completed_reentry_keep_run_identity(
    finance_delivery, tmp_path, monkeypatch
):
    original, dataset, *_ = finance_delivery
    run = tmp_path / "fixed-run"
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text("max_walltime_seconds: 0.001\n")
    args = [
        "train",
        "finance-transformer",
        "--config",
        str(original / "resolved_config.yaml"),
        "--dataset",
        str(dataset),
        "--run-dir",
        str(run),
    ]
    runner = CliRunner()
    paused = runner.invoke(app, [*args, "--runtime", str(runtime)])
    assert paused.exit_code == 75, paused.output
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "paused"
    assert manifest["stop_reason"] == "walltime_budget"
    assert (run / manifest["recoverable_checkpoint"]).is_file()
    resumed = runner.invoke(app, args)
    assert resumed.exit_code == 0, resumed.output
    completed_bytes = (run / "manifest.json").read_bytes()
    completed = json.loads(completed_bytes)
    assert completed["run_id"] == manifest["run_id"]
    assert completed["config_hash"] == manifest["config_hash"]
    assert len(completed["attempts"]) == 2
    monkeypatch.setattr(
        "facdigger.training.finance_transformer.train_finance_transformer",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not retrain")),
    )
    again = runner.invoke(app, args)
    assert again.exit_code == 0, again.output
    assert (run / "manifest.json").read_bytes() == completed_bytes


def test_final_prediction_interruption_does_not_repeat_training(
    finance_delivery, tmp_path, monkeypatch
):
    from facdigger.training import finance_transformer, finance_transformer_engine
    from facdigger.training.finance_transformer_config import load_finance_transformer_config
    from facdigger.training.runtime import TrainingControl, TrainingPaused, TrainingRuntimeConfig

    original, dataset, *_ = finance_delivery
    config = load_finance_transformer_config(original / "resolved_config.yaml")
    run = tmp_path / "finalizing"
    control = TrainingControl(TrainingRuntimeConfig(checkpoint_interval_seconds=600))
    predict = finance_transformer.predict_finance_transformer

    def interrupted(*args, **kwargs):
        control.request_stop("final_prediction")
        return predict(*args, **kwargs)

    monkeypatch.setattr(finance_transformer, "predict_finance_transformer", interrupted)
    with pytest.raises(TrainingPaused):
        finance_transformer.run_finance_transformer(
            config, dataset, repository_root=tmp_path, run_dir=run, control=control
        )
    state = torch.load(run / "checkpoints/last.pt", weights_only=False)
    assert state["progress"]["finished"] is True
    assert json.loads((run / "manifest.json").read_text())["status"] == "paused"
    monkeypatch.setattr(finance_transformer, "predict_finance_transformer", predict)
    monkeypatch.setattr(
        finance_transformer_engine,
        "_step_optimizer",
        lambda *a, **k: pytest.fail("finalization resume must not make optimizer updates"),
    )
    finance_transformer.run_finance_transformer(
        config, dataset, repository_root=tmp_path, run_dir=run
    )
    assert json.loads((run / "manifest.json").read_text())["status"] == "complete"
