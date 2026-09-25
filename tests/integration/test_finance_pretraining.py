from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import torch

from facdigger.data.config import (
    FINANCE_TRANSFORMER_CHANNELS,
    MARKET_CONTEXT_CHANNELS,
)
from facdigger.datasets.window import (
    FinancePretrainingWindowDataset,
    FinanceTransformerWindowDataset,
    MarketFeatureStore,
    SecurityFeatureStore,
)
from facdigger.training.finance_pretrain_config import (
    FinancePretrainingExperimentConfig,
)
from facdigger.training.finance_pretrain_engine import (
    benchmark_finance_pretraining_updates,
    train_finance_pretraining,
)
from facdigger.training.finance_transformer_config import (
    FinanceTransformerExperimentConfig,
)
from facdigger.training.finance_transformer_engine import (
    FINANCE_PRETRAIN_ENCODER_CHECKPOINT,
    benchmark_finance_transformer_updates,
)


def _frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    dates = [date(2022, 1, 3) + timedelta(days=index) for index in range(18)]
    securities = ["A", "B", "C"]
    feature_rows = []
    for security_index, security in enumerate(securities):
        for date_index, trade_date in enumerate(dates):
            row = {"security_id": security, "trade_date": trade_date}
            for channel_index, channel in enumerate(FINANCE_TRANSFORMER_CHANNELS):
                row[channel] = 0.03 * date_index + 0.2 * security_index + 0.01 * channel_index
                row[f"observed_{channel}"] = True
            feature_rows.append(row)
    market_rows = []
    for date_index, trade_date in enumerate(dates):
        row = {"trade_date": trade_date}
        for channel_index, channel in enumerate(MARKET_CONTEXT_CHANNELS):
            row[channel] = 0.02 * date_index + 0.01 * channel_index
            row[f"observed_{channel}"] = True
        market_rows.append(row)
    pretrain_rows = []
    for asof_index in range(7, 13):
        for security in securities:
            pretrain_rows.append(
                {
                    "sample_id": f"{security}|{dates[asof_index]}",
                    "security_id": security,
                    "symbol": security,
                    "asof_date": dates[asof_index],
                    "feature_start": dates[asof_index - 7],
                    "feature_end": dates[asof_index],
                    "future_start": dates[asof_index + 1],
                    "future_end": dates[asof_index + 5],
                }
            )
    probe_rows = []
    probe_dates = [
        (8, "probe_fit"),
        (9, "probe_fit"),
        (10, "probe_selection"),
        (11, "probe_selection"),
    ]
    for asof_index, split in probe_dates:
        for security_index, security in enumerate(securities):
            target = (security_index - 1) * (1.0 + 0.1 * asof_index)
            probe_rows.append(
                {
                    "sample_id": f"{security}|probe|{dates[asof_index]}",
                    "security_id": security,
                    "symbol": security,
                    "asof_date": dates[asof_index],
                    "feature_start": dates[asof_index - 7],
                    "feature_end": dates[asof_index],
                    "split": split,
                    "target": target,
                    "target_1": target * 0.5,
                    "target_5": target,
                    "target_20": target * 1.5,
                }
            )
    return (
        pl.DataFrame(feature_rows),
        pl.DataFrame(market_rows),
        pl.DataFrame(pretrain_rows),
        pl.DataFrame(probe_rows),
    )


def _training_fixture():
    features, market, pretrain_index, probe_index = _frames()
    feature_store = SecurityFeatureStore(
        features=features, channels=list(FINANCE_TRANSFORMER_CHANNELS)
    )
    market_store = MarketFeatureStore(features=market, channels=list(MARKET_CONTEXT_CHANNELS))
    pretraining_dataset = FinancePretrainingWindowDataset(
        feature_store=feature_store,
        market_store=market_store,
        pretraining_index=pretrain_index,
        channels=list(FINANCE_TRANSFORMER_CHANNELS),
        market_channels=list(MARKET_CONTEXT_CHANNELS),
        context_length=8,
        future_horizon=5,
    )

    def probe(split: str) -> FinanceTransformerWindowDataset:
        return FinanceTransformerWindowDataset(
            feature_store=feature_store,
            market_store=market_store,
            sample_index=probe_index,
            channels=list(FINANCE_TRANSFORMER_CHANNELS),
            market_channels=list(MARKET_CONTEXT_CHANNELS),
            context_length=8,
            split=split,
            horizons=[1, 5, 20],
            primary_horizon=5,
        )

    config = FinancePretrainingExperimentConfig.model_validate(
        {
            "seed": 9,
            "model": {
                "patch_length": 2,
                "patch_stride": 1,
                "local_d_model": 8,
                "local_num_attention_heads": 2,
                "local_num_hidden_layers": 1,
                "local_ffn_dim": 16,
                "market_d_model": 8,
                "market_num_attention_heads": 2,
                "market_num_hidden_layers": 1,
                "market_ffn_dim": 16,
                "embedding_dim": 8,
                "cross_num_attention_heads": 2,
                "cross_num_hidden_layers": 1,
                "cross_ffn_dim": 16,
                "statistics_output_dim": 4,
                "statistics_windows": [2, 4],
                "dropout": 0.0,
            },
            "training": {
                "batch_size": 3,
                "max_epochs": 1,
                "minimum_epochs": 1,
                "patience": 1,
                "device": "cpu",
                "precision": "fp32",
                "objective": {
                    "mask_ratio": 0.3,
                    "minimum_span_patches": 1,
                    "maximum_span_patches": 2,
                    "shared_channel_fraction": 0.5,
                    "reconstruction_weight": 0.4,
                    "future_summary_weight": 0.6,
                },
                "probe": {
                    "fit_dates": 2,
                    "selection_dates": 2,
                    "epochs": 2,
                    "minimum_cross_section_size": 2,
                },
            },
        }
    )
    return config, pretraining_dataset, probe("probe_fit"), probe("probe_selection")


def test_finance_pretraining_selects_and_writes_encoder_checkpoint(tmp_path) -> None:
    config, pretraining_dataset, fit, selection = _training_fixture()

    def probe(split):
        return fit if split == "probe_fit" else selection

    checkpoint_dir = tmp_path / "checkpoints"
    _, audit = train_finance_pretraining(
        config,
        pretraining_dataset=pretraining_dataset,
        probe_fit_dataset=probe("probe_fit"),
        probe_selection_dataset=probe("probe_selection"),
        dataset_id="dataset",
        checkpoint_dir=checkpoint_dir,
    )

    checkpoint = torch.load(
        checkpoint_dir / "best_encoder.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["contract"] == FINANCE_PRETRAIN_ENCODER_CHECKPOINT
    assert checkpoint["dataset_id"] == "dataset"
    assert audit["epochs_completed"] == 1
    assert audit["outer_validation_rows_used"] == 0
    assert (checkpoint_dir / "last.pt").is_file()

    pretrain_benchmark = benchmark_finance_pretraining_updates(
        config,
        pretraining_dataset=pretraining_dataset,
        dataset_id="dataset",
        local_optimizer_updates=2,
        market_optimizer_updates=1,
        warmup_updates=0,
    )
    supervised_config = FinanceTransformerExperimentConfig.model_validate(
        {
            "seed": 9,
            "model": config.model.model_dump(),
            "training": {
                "batch_size": 3,
                "max_epochs": 1,
                "minimum_epochs": 1,
                "patience": 1,
                "dates_per_optimizer_step": 1,
                "device": "cpu",
                "precision": "fp32",
                "objective": {
                    "minimum_cross_section_size": 2,
                    "minimum_selection_dates": 2,
                },
            },
        }
    )
    supervised_benchmark = benchmark_finance_transformer_updates(
        supervised_config,
        train_dataset=probe("probe_fit"),
        dataset_id="dataset",
        optimizer_updates=2,
        warmup_updates=0,
    )

    assert pretrain_benchmark["full_data_training_unchanged"] is True
    assert pretrain_benchmark["local_updates"] == 2
    assert supervised_benchmark["successful_updates"] == 2
    assert supervised_benchmark["projected_cell_hours"] > 0
