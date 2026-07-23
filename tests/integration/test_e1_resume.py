from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from facdigger.datasets.window import SnapshotWindowDataset  # noqa: E402
from facdigger.training.e1_config import E1ExperimentConfig  # noqa: E402
from facdigger.training.e1_engine import train_e1  # noqa: E402


def _datasets() -> tuple[SnapshotWindowDataset, SnapshotWindowDataset]:
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(16)]
    channels = ["x0", "x1"]
    feature_rows: list[dict] = []
    sample_rows: list[dict] = []
    for security_index in range(3):
        security_id = f"S{security_index}"
        for index, trade_date in enumerate(dates):
            feature_rows.append(
                {
                    "security_id": security_id,
                    "trade_date": trade_date,
                    "x0": security_index * 0.2 + index * 0.01,
                    "x1": security_index * -0.1 + index * 0.02,
                }
            )
        for index in range(7, 16):
            split = "train" if index <= 11 else "valid"
            sample_rows.append(
                {
                    "sample_id": f"{security_id}|{dates[index]}",
                    "security_id": security_id,
                    "symbol": security_id,
                    "asof_date": dates[index],
                    "feature_start": dates[index - 7],
                    "split": split,
                    "target": security_index * 0.3 + index * 0.01,
                }
            )
    features = pl.DataFrame(feature_rows)
    samples = pl.DataFrame(sample_rows)
    common = {
        "features": features,
        "sample_index": samples,
        "channels": channels,
        "context_length": 8,
    }
    return (
        SnapshotWindowDataset(**common, split="train"),
        SnapshotWindowDataset(**common, split="valid"),
    )


def _config(tmp_path) -> E1ExperimentConfig:
    return E1ExperimentConfig.model_validate(
        {
            "output_root": tmp_path,
            "seed": 23,
            "channels": ["x0", "x1"],
            "model": {
                "patch_length": 4,
                "patch_stride": 4,
                "d_model": 8,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "ffn_dim": 16,
                "dropout": 0.0,
                "alpha_hidden_dim": 8,
                "alpha_dropout": 0.0,
                "norm_type": "layernorm",
            },
            "training": {
                "batch_size": 6,
                "max_epochs": 3,
                "patience": 10,
                "minimum_epochs": 1,
                "learning_rate": 0.001,
                "gradient_accumulation_steps": 2,
                "device": "cpu",
                "precision": "fp32",
            },
        }
    )


def test_epoch_resume_matches_uninterrupted_training(tmp_path) -> None:
    train_dataset, valid_dataset = _datasets()
    config = _config(tmp_path)
    full_dir = tmp_path / "full"
    resumed_dir = tmp_path / "resumed"

    _, full_audit = train_e1(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        dataset_id="tiny-dataset",
        checkpoint_dir=full_dir,
    )
    train_e1(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        dataset_id="tiny-dataset",
        checkpoint_dir=resumed_dir,
        stop_after_epoch=1,
    )
    _, resumed_audit = train_e1(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        dataset_id="tiny-dataset",
        checkpoint_dir=resumed_dir,
        resume_from=resumed_dir / "last.pt",
    )

    full = torch.load(full_dir / "last.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_dir / "last.pt", map_location="cpu", weights_only=False)
    assert full_audit["global_step"] == resumed_audit["global_step"]
    assert full_audit["history"] == resumed_audit["history"]
    assert resumed_audit["resumed_from_epoch"] == 1
    for name, value in full["model_state"].items():
        torch.testing.assert_close(value, resumed["model_state"][name], rtol=0, atol=0)
