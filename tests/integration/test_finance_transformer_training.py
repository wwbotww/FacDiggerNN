from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from facdigger.data.config import (
    FINANCE_TRANSFORMER_CHANNELS,
    MARKET_CONTEXT_CHANNELS,
)
from facdigger.datasets.window import (
    FinanceTransformerWindowDataset,
    MarketFeatureStore,
    SecurityFeatureStore,
)
from facdigger.training.finance_transformer_config import (
    FinanceTransformerExperimentConfig,
)
from facdigger.training.finance_transformer_engine import train_finance_transformer


def _datasets() -> tuple[FinanceTransformerWindowDataset, FinanceTransformerWindowDataset]:
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(40)]
    securities = [f"sec-{index}" for index in range(4)]
    feature_rows: list[dict[str, object]] = []
    for security_index, security in enumerate(securities):
        for date_index, trade_date in enumerate(dates):
            row: dict[str, object] = {
                "security_id": security,
                "trade_date": trade_date,
            }
            for channel_index, channel in enumerate(FINANCE_TRANSFORMER_CHANNELS):
                row[channel] = (
                    date_index * 0.01
                    + security_index * 0.03
                    + channel_index * 0.001
                )
                row[f"observed_{channel}"] = True
            feature_rows.append(row)
    market_rows: list[dict[str, object]] = []
    for date_index, trade_date in enumerate(dates):
        row = {"trade_date": trade_date}
        for channel_index, channel in enumerate(MARKET_CONTEXT_CHANNELS):
            row[channel] = date_index * 0.01 + channel_index * 0.002
            row[f"observed_{channel}"] = True
        market_rows.append(row)
    sample_rows: list[dict[str, object]] = []
    asof_dates = dates[32:39]
    for asof_index, asof_date in enumerate(asof_dates):
        split = "train_fit" if asof_index < 4 else "inner_selection"
        for security_index, security in enumerate(securities):
            target = (security_index - 1.5) * (1 + asof_index * 0.02)
            sample_rows.append(
                {
                    "sample_id": f"{security}|{asof_date.isoformat()}",
                    "security_id": security,
                    "symbol": security,
                    "asof_date": asof_date,
                    "feature_start": dates[asof_index + 1],
                    "feature_end": asof_date,
                    "split": split,
                    "target": target,
                    "target_1": target * 0.8,
                    "target_5": target,
                    "target_20": target * 1.2,
                }
            )
    feature_store = SecurityFeatureStore(
        features=pl.DataFrame(feature_rows),
        channels=FINANCE_TRANSFORMER_CHANNELS,
    )
    market_store = MarketFeatureStore(
        features=pl.DataFrame(market_rows),
        channels=MARKET_CONTEXT_CHANNELS,
    )
    sample_index = pl.DataFrame(sample_rows)

    def dataset(split: str) -> FinanceTransformerWindowDataset:
        return FinanceTransformerWindowDataset(
            feature_store=feature_store,
            market_store=market_store,
            sample_index=sample_index,
            channels=FINANCE_TRANSFORMER_CHANNELS,
            market_channels=MARKET_CONTEXT_CHANNELS,
            context_length=32,
            split=split,
            horizons=[1, 5, 20],
            primary_horizon=5,
        )

    return dataset("train_fit"), dataset("inner_selection")


def test_finance_transformer_trains_one_epoch_and_writes_replay_checkpoint(tmp_path) -> None:
    train, selection = _datasets()
    config = FinanceTransformerExperimentConfig.model_validate(
        {
            "model": {
                "patch_length": 8,
                "patch_stride": 4,
                "local_d_model": 16,
                "local_num_attention_heads": 4,
                "local_num_hidden_layers": 1,
                "local_ffn_dim": 32,
                "market_d_model": 8,
                "market_num_attention_heads": 2,
                "market_num_hidden_layers": 1,
                "market_ffn_dim": 16,
                "embedding_dim": 16,
                "cross_num_attention_heads": 4,
                "cross_num_hidden_layers": 1,
                "cross_ffn_dim": 32,
                "statistics_output_dim": 8,
                "statistics_windows": [4, 8, 16, 32],
                "dropout": 0.0,
            },
            "training": {
                "batch_size": 2,
                "max_epochs": 1,
                "minimum_epochs": 1,
                "patience": 1,
                "encoder_learning_rate": 0.0001,
                "head_learning_rate": 0.0003,
                "dates_per_optimizer_step": 2,
                "device": "cpu",
                "precision": "fp32",
                "objective": {
                    "minimum_cross_section_size": 4,
                    "minimum_selection_dates": 2,
                    "minimum_selection_coverage": 1.0,
                    "selection_subperiods": 2,
                },
            },
        }
    )

    _, audit = train_finance_transformer(
        config,
        train_dataset=train,
        valid_dataset=selection,
        dataset_id="synthetic-finance-transformer",
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert audit["epochs_completed"] == 1
    assert audit["optimization_protocol"]["method"] == "exact_leaf_embedding_replay"
    assert audit["history"][0]["maximum_local_replay_error"] == 0.0
    assert (tmp_path / "checkpoints" / "best.pt").is_file()
    assert (tmp_path / "checkpoints" / "last.pt").is_file()
