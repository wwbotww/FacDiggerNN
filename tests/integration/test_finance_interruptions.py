from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from test_finance_pretraining import _training_fixture
from test_finance_transformer_training import _config, _datasets

from facdigger.training.finance_pretrain_engine import train_finance_pretraining
from facdigger.training.finance_transformer_engine import train_finance_transformer
from facdigger.training.runtime import TrainingControl, TrainingPaused, TrainingRuntimeConfig


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            if key != "epoch_elapsed_seconds":
                _equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _equal(a, b)
    else:
        assert left == right


def _run(kind, path, *, control=None, callback=None, resume=None):
    torch.set_num_threads(1)
    if kind == "supervised":
        config = _config()
        train, selection = _datasets()
        kwargs = {"train_dataset": train, "valid_dataset": selection}
        trainer = train_finance_transformer
    else:
        config, train, fit, selection = _training_fixture()
        kwargs = {
            "pretraining_dataset": train,
            "probe_fit_dataset": fit,
            "probe_selection_dataset": selection,
        }
        trainer = train_finance_pretraining
    config = config.model_copy(
        update={
            "model": config.model.model_copy(update={"dropout": 0.1}),
            "training": config.training.model_copy(
                update={
                    "max_epochs": 2,
                    "minimum_epochs": 2,
                    "patience": 2,
                }
            ),
        }
    )
    return trainer(
        config,
        **kwargs,
        dataset_id="fixture",
        checkpoint_dir=path,
        resume_from=resume,
        control=control,
        progress_callback=callback,
    )


@pytest.mark.parametrize(
    "kind,phase,cursor",
    [
        ("supervised", "train", 0),
        ("supervised", "train", 2),
        ("supervised", "selection", None),
        ("supervised", "epoch_complete", None),
        ("pretrain", "local", 1),
        ("pretrain", "market", 1),
        ("pretrain", "probe", None),
        ("pretrain", "epoch_complete", None),
    ],
)
def test_safe_boundary_resume_matches_continuous(tmp_path, kind, phase, cursor):
    full_dir = tmp_path / "full"
    full, _ = _run(kind, full_dir)
    control = TrainingControl(TrainingRuntimeConfig(checkpoint_interval_seconds=1e-9))

    def pause(event):
        cursor_key = {"local": "local_cursor", "market": "market_cursor"}.get(phase, "cursor")
        if (
            event["event"] == "checkpoint_saved"
            and event["phase"] == phase
            and (cursor is None or event[cursor_key] == cursor)
        ):
            control.request_stop("test_boundary")

    partial = tmp_path / "partial"
    with pytest.raises(TrainingPaused):
        _run(kind, partial, control=control, callback=pause)
    # A crash between last.pt and best export must be repairable from last.pt alone.
    best_file = partial / ("best.pt" if kind == "supervised" else "best_encoder.pt")
    best_file.unlink(missing_ok=True)
    resumed, _ = _run(kind, partial, resume=partial / "last.pt")
    _equal(full.state_dict(), resumed.state_dict())
    full_state = torch.load(full_dir / "last.pt", weights_only=False)
    resumed_state = torch.load(partial / "last.pt", weights_only=False)
    for key in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "rng_state",
        "history",
        "global_step",
        "best_epoch",
        "stale_epochs",
    ):
        _equal(full_state[key], resumed_state[key])


@pytest.mark.parametrize("kind", ["supervised", "pretrain"])
def test_legacy_epoch_checkpoint_remains_readable(tmp_path: Path, kind):
    full, _ = _run(kind, tmp_path / "full")
    control = TrainingControl(TrainingRuntimeConfig(checkpoint_interval_seconds=1e-9))

    def pause(event):
        if event["event"] == "checkpoint_saved" and event["phase"] == "epoch_complete":
            control.request_stop()

    partial = tmp_path / "partial"
    with pytest.raises(TrainingPaused):
        _run(kind, partial, control=control, callback=pause)
    state = torch.load(partial / "last.pt", weights_only=False)
    best = state.pop("best_checkpoint")
    state.pop("progress")
    state["contract"] = (
        "finance_patch_transformer_checkpoint"
        if kind == "supervised"
        else "finance_patch_pretrain_resume"
    )
    torch.save(state, partial / "last.pt")
    torch.save(best, partial / ("best.pt" if kind == "supervised" else "best_encoder.pt"))
    resumed, _ = _run(kind, partial, resume=partial / "last.pt")
    _equal(full.state_dict(), resumed.state_dict())


@pytest.mark.skipif(os.name != "posix", reason="POSIX process signals")
@pytest.mark.parametrize("kind", ["supervised", "pretrain"])
@pytest.mark.parametrize("mechanism", ["kill", "term"])
def test_real_process_termination_recovers_committed_progress(tmp_path, kind, mechanism):
    partial = tmp_path / "partial"
    script = Path(__file__).with_name("finance_interrupt_process.py")
    with subprocess.Popen(
        [sys.executable, str(script), kind, mechanism, str(partial)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            if mechanism == "kill":
                deadline = time.monotonic() + 30
                while not (partial / "ready").exists() and process.poll() is None:
                    if time.monotonic() >= deadline:
                        pytest.fail("child did not commit its first update")
                    time.sleep(0.02)
                assert (partial / "ready").exists()
                process.kill()
            _, stderr = process.communicate(timeout=30)
            assert process.returncode == (-9 if mechanism == "kill" else 75), stderr
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    assert json.loads((partial / "manifest.json").read_text())["status"] == (
        "running" if mechanism == "kill" else "paused"
    )
    full, _ = _run(kind, tmp_path / "full")
    checkpoints = partial / "checkpoints"
    resumed, _ = _run(kind, checkpoints, resume=checkpoints / "last.pt")
    _equal(full.state_dict(), resumed.state_dict())


@pytest.mark.parametrize("kind", ["supervised", "pretrain"])
def test_stop_inside_selection_or_probe_replays_only_that_phase(tmp_path, kind, monkeypatch):
    from facdigger.training import finance_pretrain_engine, finance_transformer_engine

    module = finance_transformer_engine if kind == "supervised" else finance_pretrain_engine
    name = (
        "evaluate_finance_transformer_selection"
        if kind == "supervised"
        else "evaluate_train_only_linear_probe"
    )
    original = getattr(module, name)
    full, _ = _run(kind, tmp_path / "full")
    control = TrainingControl(TrainingRuntimeConfig(checkpoint_interval_seconds=1e-9))

    def interrupt(*args, **kwargs):
        counter = 0
        check = kwargs["check_stop"]

        def stop():
            nonlocal counter
            counter += 1
            if counter == 2:
                control.request_stop("inside_selection")
            check()

        return original(*args, **{**kwargs, "check_stop": stop})

    monkeypatch.setattr(module, name, interrupt)
    path = tmp_path / "partial"
    with pytest.raises(TrainingPaused):
        _run(kind, path, control=control)
    checkpoint = torch.load(path / "last.pt", weights_only=False)
    assert checkpoint["progress"]["phase"] == ("selection" if kind == "supervised" else "probe")
    monkeypatch.setattr(module, name, original)
    resumed, _ = _run(kind, path, resume=path / "last.pt")
    _equal(full.state_dict(), resumed.state_dict())


def test_skipped_amp_update_consumes_cursor_but_not_scheduler_step(tmp_path, monkeypatch):
    from facdigger.training import finance_transformer_engine as engine

    original = engine._step_optimizer
    count = 0

    def skip_first(optimizer, scaler, parameters, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            optimizer.zero_grad(set_to_none=True)
            return None, False
        return original(optimizer, scaler, parameters, **kwargs)

    monkeypatch.setattr(engine, "_step_optimizer", skip_first)
    full, _ = _run("supervised", tmp_path / "full")
    count = 0
    control = TrainingControl(TrainingRuntimeConfig(checkpoint_interval_seconds=1e-9))

    def pause(event):
        if event["event"] == "checkpoint_saved" and event.get("cursor") == 2:
            control.request_stop()

    path = tmp_path / "partial"
    with pytest.raises(TrainingPaused):
        _run("supervised", path, control=control, callback=pause)
    checkpoint = torch.load(path / "last.pt", weights_only=False)
    assert checkpoint["global_step"] == 0
    assert checkpoint["progress"]["amp_skipped_optimizer_steps"] == 1
    resumed, _ = _run("supervised", path, resume=path / "last.pt")
    _equal(full.state_dict(), resumed.state_dict())
