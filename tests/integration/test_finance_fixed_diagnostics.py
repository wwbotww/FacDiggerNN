from __future__ import annotations

import json
import math

import polars as pl
import pytest
import torch
from test_finance_diagnostic_prefix import _cached
from test_finance_interruptions import _equal
from test_finance_transformer_training import _config, _datasets

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.datasets.window import FinanceTransformerInferenceWindowDataset
from facdigger.models.finance_patch_transformer import build_finance_transformer_model
from facdigger.models.finance_scoring import predict_finance_horizons
from facdigger.models.finance_statistics import FinanceStatisticsRanker
from facdigger.research import finance_fixed_diagnostics as runner
from facdigger.research.finance_diagnostics import score_panel, statistics_style_exposures
from facdigger.research.finance_fixed_gradients import loss_gradient_diagnostics
from facdigger.training.e1_engine import _rng_state
from facdigger.training.runtime import save_checkpoint, write_json


def _fixture(tmp_path, monkeypatch, candidate):
    torch.set_num_threads(1)
    train, _ = _datasets()
    config = _config()
    config = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={"statistics_windows": [5, 20], "dropout": 0.1}
            ),
            "training": config.training.model_copy(
                update={
                    "max_epochs": 10,
                    "minimum_epochs": 6,
                    "objective": config.training.objective.model_copy(
                        update={"minimum_cross_section_size": 2}
                    ),
                }
            ),
        }
    )
    # Preserve four computational tokens, but supervise three on every date.
    labels = train.sample_rows.filter(pl.col("security_id") != "sec-3")
    cached = _cached(train, tmp_path / "cache/F", windows=(5, 20))
    dataset = (
        cached
        if candidate != "finance"
        else FinanceTransformerInferenceWindowDataset(
            feature_store=train.feature_store,
            market_store=train.market_store,
            inference_index=cached.sample_rows,
            channels=train.channels,
            market_channels=list(train.market_channels),
            context_length=train.context_length,
            primary_horizon=5,
        )
    )
    protocol = {"dataset_id": "fixture"}
    identity = {
        "candidate": candidate,
        "config": config.model_dump(mode="json"),
        "data_protocol": protocol,
        "prefix_epochs": 2,
    }
    dates = labels["asof_date"].unique().sort().to_list()
    panel = [dates[0], dates[-1]]
    model = (
        build_finance_transformer_model(config, context_length=32)
        if candidate == "finance"
        else (FinanceStatisticsRanker(candidate, input_dim=20 * 11, horizons=(1, 5, 20)))
    )
    source = tmp_path / "source"
    for epoch in (1, 2):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.001)
        save_checkpoint(
            source / f"observations/epoch-{epoch}.pt",
            {
                "epoch": epoch,
                "identity": identity,
                "model_state": model.state_dict(),
            },
        )
        part = runner.subset_dates(dataset, panel)
        fit = labels.filter(pl.col("asof_date").is_in(panel))
        daily, predictions = score_panel(model, part, fit, config)
        daily.write_parquet(source / f"epoch-{epoch}-F-daily.parquet")
        predictions.write_parquet(source / f"epoch-{epoch}-F-predictions.parquet")
        statistics_style_exposures(predictions, cached.subset(panel), fit).write_parquet(
            source / f"epoch-{epoch}-F-styles.parquet"
        )
    save_checkpoint(source / "checkpoints/last.pt", {"fixture": True})
    write_json(
        source / "diagnostic.json",
        {
            "identity": identity,
            "status": "observations_complete",
            "training_status": "paused",
            "holdout_used": False,
            "committed_epoch": 2,
            "committed_phase": "epoch_complete",
            "F_panel": [str(d) for d in panel],
            "global_step": 2 * math.ceil(len(dates) / config.training.dates_per_optimizer_step),
            "checkpoint_sha256": sha256_file(source / "checkpoints/last.pt"),
            "attempts": [{"environment": {}}],
        },
    )
    monkeypatch.setattr(runner, "fit_inputs", lambda *a: ({}, labels, cached.sample_rows))
    monkeypatch.setattr(runner, "window_datasets", lambda *a: {"F": dataset})
    monkeypatch.setattr(runner, "fixed_dates", lambda rows, **kw: dates if kw else panel)
    monkeypatch.setattr(runner, "CHUNK_DATES", 1)
    monkeypatch.setattr(
        runner, "collect_git_state", lambda *a: {"dirty": False, "commit": "fixture"}
    )
    monkeypatch.setattr(runner, "collect_environment", lambda: {})

    def no_optimizer(*args, **kwargs):
        raise AssertionError("fixed diagnostics must never construct an optimizer")

    monkeypatch.setattr(torch.optim, "AdamW", no_optimizer)
    return config, dataset, labels, model, source


@pytest.mark.parametrize("candidate", ["finance", "statistics_linear", "statistics_mlp"])
def test_fixed_diagnostics_reuse_resume_and_match_complete_forward(
    tmp_path, monkeypatch, candidate
):
    config, dataset, labels, model, source = _fixture(tmp_path, monkeypatch, candidate)
    original_hashes = {p: sha256_file(p) for p in source.rglob("*") if p.is_file()}
    scored_dates = []
    original_score = runner.score_panel

    def score(model, data, *args, **kwargs):
        # These older C observations were FP32, despite the training AMP setting.
        assert kwargs["precision"] == "fp32"
        scored_dates.extend(data.sample_rows["asof_date"].unique().to_list())
        return original_score(model, data, *args, **kwargs)

    monkeypatch.setattr(runner, "score_panel", score)
    original_commit = runner._write_chunk

    def pause_after_commit(path, tables):
        original_commit(path, tables)
        raise TimeoutError("injected boundary pause")

    monkeypatch.setattr(runner, "_write_chunk", pause_after_commit)

    def run():
        return runner.run_fixed_diagnostics(
            tmp_path / "snapshot",
            tmp_path / "checksums",
            config,
            tmp_path / "cache",
            source,
            tmp_path / "result",
            budget_seconds=300,
            repository_root=tmp_path,
        )

    assert run()["status"] == "paused_budget_or_signal"
    assert len(scored_dates) == 1
    monkeypatch.setattr(runner, "_write_chunk", original_commit)
    completed = run()
    assert completed["status"] == "fixed_diagnostics_complete"
    assert completed["optimizer_updates"] == 0 and completed["model_state_unchanged"] is True
    assert completed["holdout_used"] is False
    assert completed["identity"]["fixed_forward_precision"] == "fp32"
    dates = labels["asof_date"].unique().sort().to_list()
    assert sorted(scored_dates) == sorted([dates[1], dates[2]] * 2)
    for epoch in (1, 2):
        state = torch.load(source / f"observations/epoch-{epoch}.pt", weights_only=False)
        model.load_state_dict(state["model_state"])
        expected, predictions = original_score(model, dataset, labels, config)
        actual = pl.read_parquet(tmp_path / f"result/epoch-{epoch}-F-daily.parquet")
        assert actual.equals(expected.sort("asof_date", "horizon"))
        assert predictions.equals(
            pl.read_parquet(tmp_path / f"result/epoch-{epoch}-F-predictions.parquet")
        )
        assert actual["computational_rows"].to_list() == [4] * 12
        assert actual["labelled_rows"].to_list() == [3] * 12
    summary = json.loads((tmp_path / "result/full-fit-summary.json").read_text())
    assert len(summary["gradients"]) == (0 if candidate == "statistics_linear" else 8)
    if candidate != "statistics_linear":
        assert len([g for g in summary["gradient_summary"] if g["group"] == "shared"]) == 2
    assert not list((tmp_path / "result").glob("*-V-*"))
    assert not list((tmp_path / "result").glob("*-S-*"))
    assert not (tmp_path / "result/manifest.json").exists()
    assert original_hashes == {p: sha256_file(p) for p in original_hashes}
    assert run()["status"] == "fixed_diagnostics_complete"
    assert len(scored_dates) == 4  # Completed observations are never scored again.
    # Even changed source formatting invalidates the bound continuation identity.
    with (source / "diagnostic.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(DataContractError, match="continuation identity"):
        run()


def test_indexed_cuda_scoring_uses_the_same_precision(monkeypatch):
    """Exercise actual scoring on CPU with a CUDA-device/autocast adapter.

    This checks the device-name dispatch, not CUDA numerical reproducibility.
    Before the fix, cuda:0 silently disabled AMP and produced FP32 scores.
    """
    torch.set_num_threads(1)
    dataset, _ = _datasets()
    config = _config()
    model = build_finance_transformer_model(config, context_length=32)
    original_to = torch.Tensor.to
    original_autocast = torch.autocast
    enabled_flags = []

    def cpu_to(tensor, *args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    def cpu_autocast(*, device_type, dtype, enabled):
        assert device_type == "cuda" and dtype == torch.float16
        enabled_flags.append(enabled)
        return original_autocast(device_type="cpu", dtype=torch.bfloat16, enabled=enabled)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(torch, "autocast", cpu_autocast)

    def predict(device, precision):
        enabled_flags.clear()
        scores = predict_finance_horizons(
            model, dataset, batch_size=3, device=device, precision=precision, num_workers=0
        )
        assert enabled_flags and all(flag == (precision == "fp16") for flag in enabled_flags)
        return scores

    plain = predict("cuda", "fp16")
    indexed = predict("cuda:0", "fp16")
    assert (plain == indexed).all()
    fp32 = predict("cuda:0", "fp32")
    assert (fp32 == predict("cpu", "fp32")).all()
    assert not (plain == fp32).all()


@pytest.mark.parametrize("candidate", ["finance", "statistics_mlp"])
def test_gradient_geometry_preserves_state_modes_rng_and_existing_gradients(
    tmp_path,
    monkeypatch,
    candidate,
):
    config, dataset, labels, model, _ = _fixture(tmp_path, monkeypatch, candidate)
    model.train()
    next(iter(model.children())).eval()  # Mixed submodule modes must survive.
    modes = [module.training for module in model.modules()]
    for p in model.parameters():
        p.grad = torch.full_like(p, 0.7)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    gradients = [p.grad.clone() for p in model.parameters()]
    before_rng = _rng_state()
    day = labels["asof_date"].min()
    part = runner.subset_dates(dataset, [day])
    result = loss_gradient_diagnostics(
        model, part, labels.filter(pl.col("asof_date") == day), config
    )
    assert result["additivity"]["passed"]
    assert result["additivity"]["relative_l2"] < 2e-5
    assert result["geometry"]["shared"]["parameters"] > 0
    assert result["geometry"]["shared"]["parameters"] < result["geometry"]["all"]["parameters"]
    assert result["computational_rows"] == 4 and result["labelled_rows"] == 3
    _equal(before_rng, _rng_state())
    _equal(state, model.state_dict())
    assert modes == [module.training for module in model.modules()]
    for before, p in zip(gradients, model.parameters(), strict=True):
        assert torch.equal(before, p.grad)

    def stop():
        raise TimeoutError("injected mid-diagnostic stop")

    with pytest.raises(TimeoutError):
        loss_gradient_diagnostics(
            model, part, labels.filter(pl.col("asof_date") == day), config, check_stop=stop
        )
    _equal(before_rng, _rng_state())
    _equal(state, model.state_dict())
    assert modes == [module.training for module in model.modules()]


def test_committed_chunks_refuse_corruption_and_missing_prediction_keys(tmp_path, monkeypatch):
    config, dataset, labels, model, _ = _fixture(tmp_path, monkeypatch, "statistics_linear")
    daily, predictions = score_panel(model, dataset, labels, config)
    tables = {
        "daily": daily,
        "predictions": predictions,
        "styles": statistics_style_exposures(predictions, dataset, labels),
    }
    runner.validate_observation(tables, dataset.sample_rows, labels, config)
    with pytest.raises(DataContractError, match="prediction keys"):
        runner.validate_observation(
            {**tables, "predictions": predictions.head(-1)}, dataset.sample_rows, labels, config
        )
    with pytest.raises(DataContractError, match="daily support"):
        runner.validate_observation(
            {**tables, "daily": daily.head(-1)}, dataset.sample_rows, labels, config
        )
    path = tmp_path / "chunk"
    runner._write_chunk(path, tables)
    with (path / "predictions.parquet").open("ab") as stream:
        stream.write(b"broken")
    with pytest.raises(DataContractError, match="checksum mismatch"):
        runner._read_chunk(path)


def test_gradient_component_loss_preserves_original_total():
    from facdigger.training.finance_transformer_engine import _multi_horizon_loss

    scores = torch.tensor(
        [[0.2, -0.3, 0.1], [0.7, 0.8, -0.2], [-0.6, 0.2, 0.4]], requires_grad=True
    )
    targets = torch.tensor([[-1.0, 0.0, 1.0], [0.0, 1.0, -1.0], [1.0, -1.0, 0.0]])
    kwargs = dict(
        horizons=(1, 5, 20),
        horizon_weights={1: 0.2, 5: 0.6, 20: 0.2},
        epsilon=1e-6,
        scale_regularization=0.01,
    )
    loss, _ = _multi_horizon_loss(scores, targets, **kwargs)
    terms = [
        _multi_horizon_loss(scores, targets, **kwargs, diagnostic_component=c)[0]
        for c in (1, 5, 20, "scale")
    ]
    torch.testing.assert_close(sum(terms), loss)
    expected = torch.autograd.grad(loss, scores, retain_graph=True)[0]
    actual = sum(torch.autograd.grad(term, scores, retain_graph=True)[0] for term in terms)
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="unknown diagnostic"):
        _multi_horizon_loss(scores, targets, **kwargs, diagnostic_component="other")


def test_fit_inputs_reconstruct_original_plan_without_collecting_holdout_targets(
    tmp_path,
    monkeypatch,
):
    from facdigger.data.snapshots import build_dataset_snapshot
    from facdigger.training.finance_data import finance_data_protocol, load_finance_selection
    from facdigger.training.finance_transformer_config import FinanceTransformerExperimentConfig
    from facdigger.training.runtime import write_snapshot_checksums
    from tests.integration.test_finance_selection_snapshot import finance_config

    snapshot, manifest = build_dataset_snapshot(finance_config(tmp_path))
    config = FinanceTransformerExperimentConfig.model_validate(
        {
            "model": {"statistics_windows": [5, 20]},
        }
    )
    expected = finance_data_protocol(snapshot, manifest, config)
    original, _, _ = load_finance_selection(snapshot, manifest)
    inventory = tmp_path / "checksums.json"
    write_snapshot_checksums(snapshot, inventory)

    def metadata_only(*args, sample_index):
        assert not any(c.startswith("target") for c in sample_index.columns)
        assert {"train", "valid", "test"}.issubset(set(sample_index["split"].to_list()))
        return load_finance_selection(*args, sample_index=sample_index)

    monkeypatch.setattr(runner, "load_finance_selection", metadata_only)
    _, fit, pool = runner.fit_inputs(snapshot, inventory, config, expected)
    assert fit.equals(original.filter(pl.col("split") == "train_fit"))
    assert fit["target_5"].is_finite().all()
    assert fit["asof_date"].unique().sort().equals(pool["asof_date"].unique().sort())
    write_json(inventory, {"manifest.json": "incorrect"})
    with pytest.raises(DataContractError, match="checksum mismatch"):
        runner.fit_inputs(snapshot, inventory, config, expected)
