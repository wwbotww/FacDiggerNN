from __future__ import annotations

import copy
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
import torch

from facdigger.datasets.window import (
    FinanceTransformerInferenceWindowDataset,
    FinanceTransformerWindowDataset,
)
from facdigger.models.finance_patch_transformer import FinancePatchTransformer
from facdigger.training.finance_transformer_engine import (
    _multi_horizon_loss,
    backward_complete_date_with_embedding_replay,
)


def _dataset(rows: int = 6) -> FinanceTransformerWindowDataset:
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(32)]
    securities = [f"S{index}" for index in range(rows)]
    features = pl.DataFrame(
        {
            "security_id": [security for security in securities for _ in dates],
            "trade_date": dates * rows,
            "x0": [
                float(index) / 10 + security_index * 0.03
                for security_index, _ in enumerate(securities)
                for index in range(32)
            ],
            "x1": [
                float(index) / 20 - security_index * 0.02
                for security_index, _ in enumerate(securities)
                for index in range(32)
            ],
            "x2": [
                float(index) / 30 + security_index * 0.01
                for security_index, _ in enumerate(securities)
                for index in range(32)
            ],
            "x3": [
                float(index) / 40 - security_index * 0.04
                for security_index, _ in enumerate(securities)
                for index in range(32)
            ],
            "observed_x0": [True] * (rows * 32),
            "observed_x1": [True] * (rows * 32),
            "observed_x2": [True] * (rows * 32),
            "observed_x3": [True] * (rows * 32),
        }
    )
    market = pl.DataFrame(
        {
            "trade_date": dates,
            "m0": [float(index) / 50 for index in range(32)],
            "m1": [float(index) / 60 for index in range(32)],
            "observed_m0": [True] * 32,
            "observed_m1": [True] * 32,
        }
    )
    sample_index = pl.DataFrame(
        {
            "sample_id": [f"{security}|latest" for security in securities],
            "security_id": securities,
            "symbol": securities,
            "asof_date": [dates[-1]] * rows,
            "feature_start": [dates[0]] * rows,
            "feature_end": [dates[-1]] * rows,
            "split": ["train_fit"] * rows,
            "target": np.linspace(-0.5, 0.5, rows),
            "target_1": np.linspace(-0.4, 0.6, rows),
            "target_5": np.linspace(-0.5, 0.5, rows),
            "target_20": np.linspace(-0.6, 0.4, rows),
        }
    )
    return FinanceTransformerWindowDataset(
        features=features,
        market_features=market,
        sample_index=sample_index,
        channels=["x0", "x1", "x2", "x3"],
        market_channels=["m0", "m1"],
        context_length=32,
        split="train_fit",
        horizons=[1, 5, 20],
        primary_horizon=5,
    )


def _model() -> FinancePatchTransformer:
    return FinancePatchTransformer(
        context_length=32,
        num_local_channels=4,
        num_market_channels=2,
        num_asset_channels=2,
        horizons=(1, 5, 20),
        patch_length=8,
        patch_stride=4,
        local_d_model=16,
        local_num_attention_heads=4,
        local_num_hidden_layers=1,
        local_ffn_dim=32,
        market_d_model=8,
        market_num_attention_heads=2,
        market_num_hidden_layers=1,
        market_ffn_dim=16,
        embedding_dim=16,
        cross_num_attention_heads=4,
        cross_num_hidden_layers=1,
        cross_ffn_dim=32,
        statistics_output_dim=8,
        statistics_windows=(4, 8, 16, 32),
        dropout=0.0,
    )


def _batch(dataset: FinanceTransformerWindowDataset) -> dict[str, torch.Tensor]:
    samples = [dataset[index] for index in range(len(dataset))]
    return {
        "values": torch.from_numpy(np.stack([sample["values"] for sample in samples])),
        "observed_mask": torch.from_numpy(
            np.stack([sample["observed_mask"] for sample in samples])
        ),
        "sample_index": torch.arange(len(samples)),
    }


@pytest.mark.parametrize("unlabelled", [0, 2])
def test_embedding_replay_matches_single_graph_parameter_gradients(unlabelled: int) -> None:
    torch.manual_seed(42)
    source = _dataset()
    dataset = FinanceTransformerInferenceWindowDataset(
        feature_store=source.feature_store, market_store=source.market_store,
        inference_index=source.sample_rows.drop(
            "split", "target", "target_1", "target_5", "target_20"
        ),
        channels=source.channels, market_channels=list(source.market_channels),
        context_length=source.context_length, primary_horizon=source.primary_horizon,
    )
    labelled_count = len(dataset) - unlabelled
    label_mask = torch.arange(len(dataset)) < labelled_count
    row_to_label = torch.arange(len(dataset))
    row_to_label[~label_mask] = -1
    reference = _model().train()
    replayed = copy.deepcopy(reference).train()
    batch = _batch(dataset)
    target_ranks = torch.stack(
        [
            torch.linspace(-1.0, 1.0, labelled_count),
            torch.linspace(-0.9, 0.9, labelled_count),
            torch.linspace(-0.8, 0.8, labelled_count),
        ],
        dim=1,
    )
    market_window = dataset.market_window_for_sample_indices(list(range(len(dataset))))
    market_values = torch.from_numpy(market_window.values).unsqueeze(0)
    market_observed = torch.from_numpy(market_window.observed_mask).unsqueeze(0)

    # The production path deliberately partitions the temporal encoder by
    # physical microbatch.  Use the same partition for the graph-preserving
    # oracle: a single larger GEMM can accumulate floating-point values in a
    # different order and is not the computation replay promises to reproduce.
    local_embeddings = [
        reference.encode_local(
            batch["values"][start : start + 2],
            batch["observed_mask"][start : start + 2],
        ).embedding
        for start in range(0, len(dataset), 2)
    ]
    market_embedding = reference.encode_market(market_values, market_observed)
    local_all = torch.cat(local_embeddings)
    local_all.retain_grad()
    output = reference.score_date(local_all, market_embedding)
    loss, expected_audit = _multi_horizon_loss(
        output.scores[label_mask],
        target_ranks,
        horizons=(1, 5, 20),
        horizon_weights={1: 0.2, 5: 0.6, 20: 0.2},
        epsilon=1e-6,
        scale_regularization=0.01,
    )
    loss.backward()

    if unlabelled:
        assert local_all.grad[~label_mask].abs().max().item() > 0

    audit = backward_complete_date_with_embedding_replay(
        replayed,
        dataset,
        batch,
        device="cpu",
        target_rank_lookup=target_ranks,
        row_to_label_index=row_to_label,
        horizon_weights={1: 0.2, 5: 0.6, 20: 0.2},
        epsilon=1e-6,
        scale_regularization=0.01,
        amp_enabled=False,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        physical_microbatch_size=2,
        dates_in_optimizer_step=1,
        relative_tolerance=1e-5,
        absolute_tolerance=1e-6,
    )

    assert audit["rows"] == len(dataset)
    assert audit["labelled_rows"] == labelled_count
    assert audit["unlabelled_rows"] == unlabelled
    assert audit["loss"] == pytest.approx(expected_audit["loss"], abs=1e-6)
    assert audit["scale_penalty"] == pytest.approx(expected_audit["scale_penalty"], abs=1e-6)
    replayed_parameters = dict(replayed.named_parameters())
    for name, parameter in reference.named_parameters():
        replayed_gradient = replayed_parameters[name].grad
        if parameter.grad is None or replayed_gradient is None:
            assert parameter.grad is None and replayed_gradient is None, name
            continue
        torch.testing.assert_close(
            replayed_gradient,
            parameter.grad,
            rtol=1e-2,
            atol=1e-4,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def test_missing_label_does_not_change_computational_forward(tmp_path) -> None:
    from facdigger.datasets.index import align_labelled_samples
    from facdigger.models.finance_scoring import predict_finance_transformer
    from facdigger.training.common import load_snapshot_inference_rows

    torch.manual_seed(42)
    source = _dataset()
    inference = source.sample_rows.drop(
        "split", "target", "target_1", "target_5", "target_20"
    ).with_columns(pl.lit(True).alias("eligible"))
    inference.write_parquet(tmp_path / "inference.parquet")
    manifest = {"artifacts": {"inference_index": "inference.parquet"}}
    labels = source.sample_rows
    missing_one_label = labels.filter(pl.col("security_id") != "S3")

    def windows(labelled):
        pool = load_snapshot_inference_rows(
            tmp_path, manifest, asof_dates=labelled["asof_date"].unique().to_list()
        )
        return FinanceTransformerInferenceWindowDataset(
            feature_store=source.feature_store, market_store=source.market_store,
            inference_index=pool, channels=source.channels,
            market_channels=list(source.market_channels), context_length=32, primary_horizon=5,
        )

    full = windows(labels)
    changed = windows(missing_one_label)
    assert full.sample_rows.equals(changed.sample_rows)
    model = _model().eval()
    kwargs = {"batch_size": 2, "device": "cpu", "precision": "fp32", "num_workers": 0}
    expected = predict_finance_transformer(model, full, **kwargs)
    actual = predict_finance_transformer(model, changed, **kwargs)
    np.testing.assert_array_equal(actual, expected)
    projection = align_labelled_samples(changed.sample_rows, missing_one_label)
    np.testing.assert_array_equal(
        actual[projection["_computational_row"].to_numpy()], np.delete(expected, 3)
    )
