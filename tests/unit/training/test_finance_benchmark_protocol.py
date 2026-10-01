from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import build_dataset_snapshot, sha256_file
from facdigger.datasets.index import align_labelled_samples
from facdigger.training.finance_benchmark import run_finance_training_benchmark
from facdigger.training.finance_data import finance_data_protocol, load_finance_selection
from facdigger.training.finance_pretrain_config import FinancePretrainingExperimentConfig
from facdigger.training.finance_transformer_config import FinanceTransformerExperimentConfig
from tests.integration.test_dataset_pipeline import sessions
from tests.integration.test_finance_selection_snapshot import finance_config


@pytest.fixture
def benchmark_inputs(tmp_path):
    config = finance_config(tmp_path)
    # A real missing future price removes a supervised label while preserving
    # the stock's already-observed endpoint and its computational context.
    bars = pl.read_parquet(config.sources.bars)
    bars.filter(~(
        (pl.col("security_id") == "sec-b") & (pl.col("trade_date") == sessions(180)[65])
    )).write_parquet(config.sources.bars)
    snapshot, manifest = build_dataset_snapshot(config)
    supervised = FinanceTransformerExperimentConfig.model_validate({
        "model": {"statistics_windows": [5, 20]},
    })
    pretraining = FinancePretrainingExperimentConfig.model_validate({
        "model": {"statistics_windows": [5, 20]},
        "training": {"probe": {"fit_dates": 5, "selection_dates": 4}},
    })
    return snapshot, manifest, supervised, pretraining


def test_benchmark_loads_full_pool_and_binds_actual_two_stage_protocols(
    benchmark_inputs, monkeypatch,
):
    snapshot, manifest, supervised, pretraining = benchmark_inputs
    before = {path.name: sha256_file(path) for path in snapshot.iterdir() if path.is_file()}
    recorded = {}

    def supervised_kernel(config, **kwargs):
        recorded["supervised"] = kwargs
        assert config == supervised
        return {
            "projected_cell_hours": 0.5, "device": "cpu", "precision": "fp32",
            "cuda_peak_reserved_bytes": None,
        }

    def pretraining_kernel(config, **kwargs):
        recorded["pretraining"] = kwargs
        assert config == pretraining
        return {
            "projected_run_hours": 0.25, "device": "cpu", "precision": "fp32",
            "cuda_peak_reserved_bytes": None,
        }

    monkeypatch.setattr(
        "facdigger.training.finance_benchmark.benchmark_finance_transformer_updates",
        supervised_kernel,
    )
    monkeypatch.setattr(
        "facdigger.training.finance_benchmark.benchmark_finance_pretraining_updates",
        pretraining_kernel,
    )
    monkeypatch.setattr("facdigger.training.finance_benchmark.training_hardware", lambda: {})
    monkeypatch.setattr("facdigger.training.finance_benchmark.process_peak_rss_bytes", lambda: 100)
    report = run_finance_training_benchmark(supervised, pretraining, snapshot, optimizer_updates=2)
    index, _, plan = load_finance_selection(snapshot, manifest)
    expected_labels = index.filter(pl.col("split") == "train_fit")
    expected_pool = pl.read_parquet(snapshot / manifest["artifacts"]["inference_index"]).filter(
        pl.col("asof_date").is_in(expected_labels["asof_date"].unique().to_list())
    ).sort("asof_date", "security_id")
    loaded = recorded["supervised"]["train_dataset"]
    assert loaded.sample_rows.select(expected_pool.columns[:7]).equals(
        expected_pool.select(expected_pool.columns[:7])
    )
    assert not any(name.startswith("target") for name in loaded.sample_rows.columns)
    assert recorded["supervised"]["train_labelled_rows"].equals(expected_labels)
    assert len(loaded) > expected_labels.height
    align_labelled_samples(loaded.sample_rows, expected_labels)
    assert report["supervised_train_rows"] == expected_labels.height
    assert report["supervised_computational_rows"] == len(loaded)
    assert report["supervised_context_only_rows"] == len(loaded) - expected_labels.height
    assert report["data_protocols"] == {
        "supervised": finance_data_protocol(snapshot, manifest, supervised),
        "pretraining": finance_data_protocol(snapshot, manifest, pretraining),
    }
    assert recorded["supervised"]["data_protocol"] == report["data_protocols"]["supervised"]
    ssl = recorded["pretraining"]["pretraining_dataset"]
    assert ssl.sample_rows["future_end"].max() <= date.fromisoformat(plan["a_end"])
    assert ssl.sample_rows.equals(pl.read_parquet(snapshot / "pretraining_index.parquet"))
    assert ssl.feature_store is loaded.feature_store
    assert ssl.market_store is loaded.market_store
    assert report["pretraining_rows"] == len(ssl)
    assert report["admission"]["admitted"] is False  # No GPU measurement was made.
    assert {path.name: sha256_file(path) for path in snapshot.iterdir() if path.is_file()} == before


@pytest.mark.parametrize("stage", ["supervised", "pretraining"])
def test_benchmark_rejects_either_stage_protocol_before_any_update(
    benchmark_inputs, monkeypatch, stage,
):
    snapshot, _, supervised, pretraining = benchmark_inputs
    if stage == "supervised":
        supervised = supervised.model_copy(update={"selection_fraction": 0.2})
    else:
        pretraining = pretraining.model_copy(update={
            "training": pretraining.training.model_copy(update={
                "probe": pretraining.training.probe.model_copy(update={"fit_dates": 6}),
            }),
        })

    def forbidden(*args, **kwargs):
        pytest.fail("a mismatched protocol reached a benchmark training kernel")

    monkeypatch.setattr(
        "facdigger.training.finance_benchmark.benchmark_finance_transformer_updates", forbidden
    )
    monkeypatch.setattr(
        "facdigger.training.finance_benchmark.benchmark_finance_pretraining_updates", forbidden
    )
    with pytest.raises(DataContractError, match="differs"):
        run_finance_training_benchmark(supervised, pretraining, snapshot, optimizer_updates=2)
