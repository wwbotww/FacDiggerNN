from __future__ import annotations

import copy
from datetime import date, timedelta

import numpy as np
import polars as pl
import torch

from facdigger.datasets.window import FinanceTransformerWindowDataset
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


def test_embedding_replay_matches_single_graph_parameter_gradients() -> None:
    dataset = _dataset()
    reference = _model().train()
    replayed = copy.deepcopy(reference).train()
    batch = _batch(dataset)
    target_ranks = torch.stack(
        [
            torch.linspace(-1.0, 1.0, len(dataset)),
            torch.linspace(-0.9, 0.9, len(dataset)),
            torch.linspace(-0.8, 0.8, len(dataset)),
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
    output = reference.score_date(torch.cat(local_embeddings), market_embedding)
    loss, _ = _multi_horizon_loss(
        output.scores,
        target_ranks,
        horizons=(1, 5, 20),
        horizon_weights={1: 0.2, 5: 0.6, 20: 0.2},
        epsilon=1e-6,
        scale_regularization=0.01,
    )
    loss.backward()

    backward_complete_date_with_embedding_replay(
        replayed,
        dataset,
        batch,
        device="cpu",
        target_rank_lookup=target_ranks,
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
