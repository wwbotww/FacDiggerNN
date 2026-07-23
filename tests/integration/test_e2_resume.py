from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from facdigger.datasets.window import SnapshotWindowDataset  # noqa: E402
from facdigger.models.patchtst_alpha import PatchTSTAlphaModel  # noqa: E402
from facdigger.models.patchtst_transfer import module_fingerprint  # noqa: E402
from facdigger.training.e2_config import E2ExperimentConfig  # noqa: E402
from facdigger.training.e2_engine import train_e2  # noqa: E402


def _datasets() -> tuple[SnapshotWindowDataset, SnapshotWindowDataset]:
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(16)]
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
            sample_rows.append(
                {
                    "sample_id": f"{security_id}|{dates[index]}",
                    "security_id": security_id,
                    "symbol": security_id,
                    "asof_date": dates[index],
                    "feature_start": dates[index - 7],
                    "split": "train" if index <= 11 else "valid",
                    "target": security_index * 0.3 + index * 0.01,
                }
            )
    common = {
        "features": pl.DataFrame(feature_rows),
        "sample_index": pl.DataFrame(sample_rows),
        "channels": ["x0", "x1"],
        "context_length": 8,
    }
    return (
        SnapshotWindowDataset(**common, split="train"),
        SnapshotWindowDataset(**common, split="valid"),
    )


def _config(tmp_path) -> E2ExperimentConfig:
    return E2ExperimentConfig.model_validate(
        {
            "output_root": tmp_path,
            "seed": 31,
            "channels": ["x0", "x1"],
            "model": {
                "patch_length": 4,
                "patch_stride": 4,
                "d_model": 8,
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "ffn_dim": 16,
                "dropout": 0.0,
                "norm_type": "layernorm",
                "alpha_hidden_dim": 8,
                "alpha_dropout": 0.0,
            },
            "training": {
                "batch_size": 6,
                "max_epochs": 3,
                "patience": 10,
                "minimum_epochs": 2,
                "head_only_epochs": 1,
                "unfreeze_last_n_blocks": 1,
                "head_learning_rate": 0.001,
                "encoder_learning_rate": 0.0001,
                "device": "cpu",
                "precision": "fp32",
            },
        }
    )


def _initializer() -> tuple[PatchTSTAlphaModel, dict]:
    model = PatchTSTAlphaModel(
        context_length=8,
        num_input_channels=2,
        patch_length=4,
        patch_stride=4,
        d_model=8,
        num_attention_heads=2,
        num_hidden_layers=2,
        ffn_dim=16,
        dropout=0.0,
        attention_dropout=0.0,
        positional_dropout=0.0,
        path_dropout=0.0,
        ff_dropout=0.0,
        norm_type="layernorm",
        pre_norm=False,
        scaling="mean",
        alpha_hidden_dim=8,
        alpha_dropout=0.0,
    )
    with torch.no_grad():
        for parameter in model.backbone.parameters():
            parameter.mul_(0.5)
    fingerprint = module_fingerprint(model.backbone)
    return model, {
        "source_weights_sha256": "synthetic-source-hash",
        "fingerprints": {"target_backbone_after_transfer": fingerprint},
    }


def test_e2_resume_preserves_stage_and_optimizer_continuity(tmp_path) -> None:
    train_dataset, valid_dataset = _datasets()
    config = _config(tmp_path)
    full_dir = tmp_path / "full"
    resumed_dir = tmp_path / "resumed"

    _, full_audit, _ = train_e2(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        dataset_id="tiny-e2",
        checkpoint_dir=full_dir,
        initializer=_initializer,
    )
    train_e2(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        dataset_id="tiny-e2",
        checkpoint_dir=resumed_dir,
        stop_after_epoch=1,
        initializer=_initializer,
    )
    _, resumed_audit, _ = train_e2(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        dataset_id="tiny-e2",
        checkpoint_dir=resumed_dir,
        resume_from=resumed_dir / "last.pt",
        initializer=_initializer,
    )

    full = torch.load(full_dir / "last.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_dir / "last.pt", map_location="cpu", weights_only=False)
    assert full_audit["history"] == resumed_audit["history"]
    assert resumed_audit["resumed_from_epoch"] == 1
    assert resumed_audit["stage_audits"]["ft0_head_only"]["encoder_changed"] is False
    assert resumed_audit["stage_audits"]["ft1_last_blocks"]["encoder_changed"] is True
    for name, value in full["model_state"].items():
        torch.testing.assert_close(value, resumed["model_state"][name], rtol=0, atol=0)
