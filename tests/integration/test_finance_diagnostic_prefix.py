from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from test_finance_interruptions import _equal
from test_finance_transformer_training import _config, _datasets

from facdigger.data.contracts import DataContractError
from facdigger.datasets.finance_statistics import FinanceStatisticsDataset, cache_statistics
from facdigger.models.finance_patch_transformer import MultiScaleStatisticsEncoder
from facdigger.models.finance_scoring import predict_finance_horizons, predict_finance_transformer
from facdigger.research.finance_diagnostics import computation_rows
from facdigger.training.finance_transformer_engine import train_finance_transformer
from facdigger.training.runtime import TrainingControl, TrainingPaused, TrainingRuntimeConfig


def _cached(source, path, *, windows=(4, 8, 16, 32)):
    from facdigger.datasets.window import FinanceTransformerInferenceWindowDataset

    raw = FinanceTransformerInferenceWindowDataset(
        feature_store=source.feature_store,
        market_store=source.market_store,
        inference_index=computation_rows(source.sample_rows),
        channels=source.channels,
        market_channels=list(source.market_channels),
        context_length=source.context_length,
        primary_horizon=5,
    )
    cache_statistics(raw, path, windows=windows, identity={"dataset_id": "fixture"})
    return FinanceStatisticsDataset(
        path,
        identity={"dataset_id": "fixture"},
        windows=windows,
        expected_rows=raw.sample_rows,
    )


def test_cache_exact_masked_statistics_and_fail_closed_identity(tmp_path):
    source, _ = _datasets()
    # Real missing values plus a fully unobserved channel verify mask/latest semantics.
    block = source.feature_store.block_at(0)
    block.observed[4:14, 0] = False
    block.values[4:14, 0] = 0
    block.observed[:, 1] = False
    block.values[:, 1] = 0
    cached = _cached(source, tmp_path / "cache")
    for index in [0, len(source) - 1]:
        item = source[index]
        exact = MultiScaleStatisticsEncoder.statistics(
            torch.tensor(item["values"])[None],
            torch.tensor(item["observed_mask"])[None],
            num_asset_channels=len(source.channels),
            windows=(4, 8, 16, 32),
        ).numpy()
        # Different reduction batch shapes may round by a few float32 ULPs.
        np.testing.assert_allclose(
            cached[index]["values"],
            exact,
            rtol=8 * np.finfo(np.float32).eps,
            atol=np.finfo(np.float32).eps,
        )
    dates = cached.asof_dates[:1]
    subset = cached.subset(dates)
    assert len(subset) == 4
    np.testing.assert_array_equal(subset[0]["values"], cached[0]["values"])
    with pytest.raises(DataContractError, match="identity"):
        FinanceStatisticsDataset(
            tmp_path / "cache",
            identity={},
            windows=(4, 8, 16, 32),
            expected_rows=cached.sample_rows,
        )
    with pytest.raises(DataContractError, match="universe"):
        FinanceStatisticsDataset(
            tmp_path / "cache",
            identity={"dataset_id": "fixture"},
            windows=(4, 8, 16, 32),
            expected_rows=cached.sample_rows.head(2),
        )


@pytest.mark.parametrize("candidate", [None, "statistics_linear", "statistics_mlp"])
def test_observation_and_prefix_resume_preserve_training_state(tmp_path, candidate):
    torch.set_num_threads(1)
    train, selection = _datasets()
    config = _config()
    config = config.model_copy(
        update={
            "model": config.model.model_copy(update={"dropout": 0.1}),
            "training": config.training.model_copy(update={"max_epochs": 10, "minimum_epochs": 6}),
        }
    )
    train_labels, selection_labels = train.sample_rows, selection.sample_rows
    if candidate:
        train, selection = _cached(train, tmp_path / "F"), _cached(selection, tmp_path / "S")
    observed = []

    def observer(model, epoch):
        # Deliberately consume every RNG and change evaluation mode/gradients.
        observed.append(epoch)
        random.random()
        np.random.rand()
        torch.rand(7)
        model.eval()
        model.zero_grad(set_to_none=True)
        kwargs = dict(batch_size=2, device="cpu", precision="fp32", num_workers=0)
        scores = predict_finance_horizons(model, selection, **kwargs)
        primary = predict_finance_transformer(model, selection, **kwargs)
        np.testing.assert_array_equal(scores[:, 1], primary)

    def run(path, *, observe=False, stop=False, resume=None):
        control = TrainingControl(TrainingRuntimeConfig(checkpoint_interval_seconds=9999))

        def progress(event):
            if stop and event["event"] == "epoch_completed" and event["epoch"] == 2:
                control.request_stop("prefix")

        return train_finance_transformer(
            config,
            train_dataset=train,
            train_labelled_rows=train_labels,
            valid_dataset=selection,
            valid_labelled_rows=selection_labels,
            data_protocol={"dataset_id": "fixture"},
            dataset_id="fixture",
            checkpoint_dir=path,
            stop_after_epoch=3,
            resume_from=resume,
            control=control,
            progress_callback=progress,
            state_observer=observer if observe else None,
            diagnostic_model=candidate,
        )

    run(tmp_path / "continuous")
    run(tmp_path / "observed", observe=True)
    with pytest.raises(TrainingPaused, match="prefix"):
        run(tmp_path / "paused", observe=True, stop=True)
    paused = torch.load(tmp_path / "paused/last.pt", weights_only=False)
    assert paused["epoch"] == 2 and paused["progress"]["phase"] == "epoch_complete"
    assert paused["progress"]["finished"] is False
    assert paused["config"]["training"]["max_epochs"] == 10
    assert paused["global_step"] == 4
    run(tmp_path / "paused", observe=True, resume=tmp_path / "paused/last.pt")
    expected = torch.load(tmp_path / "continuous/last.pt", weights_only=False)
    for name in ("observed", "paused"):
        actual = torch.load(tmp_path / name / "last.pt", weights_only=False)
        _equal(expected, actual)
    assert observed[:4] == [0, 1, 2, 3]
    # Changing the candidate must not reuse its optimizer/checkpoint protocol.
    if candidate:
        assert expected["model_type"] == candidate
        assert {g["group_name"] for g in expected["optimizer_state"]["param_groups"]} == {
            "head_decay",
            "head_no_decay",
        }


def test_prefix_orchestration_exports_diagnostics_without_finishing_training(tmp_path, monkeypatch):
    import json

    import polars as pl

    from facdigger.research import finance_prefix_diagnostics as runner
    from facdigger.training import common

    train, selection = _datasets()
    config = _config()
    config = config.model_copy(
        update={
            "model": config.model.model_copy(update={"statistics_windows": [5, 20]}),
            "training": config.training.model_copy(update={"max_epochs": 10, "minimum_epochs": 6}),
        }
    )
    cache = tmp_path / "cache"
    ds = {
        "F": _cached(train, cache / "F", windows=(5, 20)),
        "S": _cached(selection, cache / "S", windows=(5, 20)),
        "V": _cached(selection, cache / "V", windows=(5, 20)),
    }
    labelled = {
        "F": train.sample_rows,
        "S": selection.sample_rows,
        "V": selection.sample_rows.with_columns(pl.lit("valid").alias("split")),
    }
    pools = {k: v.sample_rows for k, v in ds.items()}
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    labelled["V"].select("sample_id").with_columns(
        pl.lit(True).alias("eligible"),
        pl.lit(None, dtype=pl.String).alias("industry_code"),
        pl.lit(None, dtype=pl.Float64).alias("log_float_market_cap"),
    ).write_parquet(snapshot / "sample_metadata.parquet")
    # The tiny fixture isolates orchestration; real plan/scaler gates have separate tests.
    monkeypatch.setattr(
        runner,
        "diagnostic_inputs",
        lambda *a: ({"dataset_id": "fixture"}, {"dataset_id": "fixture"}, labelled, pools),
    )
    monkeypatch.setattr(
        runner, "collect_git_state", lambda *a: {"dirty": False, "commit": "fixture"}
    )
    monkeypatch.setattr(runner, "collect_environment", lambda: {})
    monkeypatch.setattr(
        runner, "fixed_dates", lambda rows: rows["asof_date"].unique().sort().to_list()
    )
    monkeypatch.setattr(common, "load_source_provenance", lambda *a: {"research_ready": False})
    result = runner.run_prefix_diagnostics(
        snapshot,
        tmp_path / "checksums",
        config,
        cache,
        tmp_path / "run",
        candidate="statistics_mlp",
        repository_root=tmp_path,
        budget_seconds=600,
    )
    assert result["status"] == "observations_complete"
    assert result["training_status"] == "paused" and result["committed_epoch"] == 2
    assert result["best_epoch"] in (1, 2)
    assert (tmp_path / "run/epoch-0-S-daily.parquet").exists()
    assert not (tmp_path / "run/epoch-0-V-daily.parquet").exists()
    evaluation = json.loads((tmp_path / "run/epoch-2-V-evaluation.json").read_text())
    assert evaluation["evaluation_split"] == "valid"
    assert evaluation["metrics"]["cross_section"]["research_ready"] is False
    last = torch.load(tmp_path / "run/checkpoints/last.pt", weights_only=False)
    assert last["progress"]["finished"] is False
    # A fixed-state observation must not masquerade as a release/checkpoint directory.
    assert not (tmp_path / "run/manifest.json").exists()
