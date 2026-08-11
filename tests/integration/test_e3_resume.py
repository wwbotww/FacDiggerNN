from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import polars as pl
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from facdigger.datasets.window import SnapshotWindowDataset  # noqa: E402
from facdigger.models.patchtst_pretrain import FinancialPatchTSTPretrainer  # noqa: E402
from facdigger.models.patchtst_transfer import module_fingerprint  # noqa: E402
from facdigger.training.e3_config import E3ExperimentConfig  # noqa: E402
from facdigger.training.e3_engine import (  # noqa: E402
    _batches_in_current_optimizer_step,
    initialize_alpha_from_financial_checkpoint,
    train_financial_pretraining,
)


def _datasets() -> tuple[SnapshotWindowDataset, SnapshotWindowDataset]:
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(20)]
    features = pl.DataFrame(
        [
            {
                "security_id": security,
                "trade_date": day,
                "x0": float(index + security_index),
                "x1": float(index - security_index),
            }
            for security_index, security in enumerate(["a", "b"])
            for index, day in enumerate(dates)
        ]
    )
    rows = []
    for security in ["a", "b"]:
        for index in range(7, 20):
            split = "pretrain_train" if index < 16 else "pretrain_selection"
            rows.append(
                {
                    "sample_id": f"{security}-{index}",
                    "security_id": security,
                    "asof_date": dates[index],
                    "feature_start": dates[index - 7],
                    "feature_end": dates[index],
                    "target": 0.0,
                    "split": split,
                }
            )
    sample_index = pl.DataFrame(rows)
    common = {
        "features": features,
        "channels": ["x0", "x1"],
        "context_length": 8,
    }
    return (
        SnapshotWindowDataset(
            **common, sample_index=sample_index, split="pretrain_train"
        ),
        SnapshotWindowDataset(
            **common, sample_index=sample_index, split="pretrain_selection"
        ),
    )


def _config(tmp_path) -> E3ExperimentConfig:
    return E3ExperimentConfig.model_validate(
        {
            "output_root": tmp_path,
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
            "pretraining": {
                "batch_size": 4,
                "gradient_accumulation_steps": 2,
                "max_epochs": 2,
                "minimum_epochs": 2,
                "patience": 2,
                "mask_ratio": 0.5,
                "device": "cpu",
                "precision": "fp32",
            },
            "finetuning": {
                "max_epochs": 2,
                "minimum_epochs": 2,
                "head_only_epochs": 1,
                "unfreeze_last_n_blocks": 1,
                "objective": {
                    "minimum_cross_section_size": 2,
                    "minimum_selection_dates": 2,
                    "minimum_selection_coverage": 1.0,
                },
            },
        }
    )


def _initializer() -> tuple[FinancialPatchTSTPretrainer, dict]:
    config = _config_model()
    model = FinancialPatchTSTPretrainer(
        context_length=8,
        num_input_channels=2,
        model_config=config,
        mask_ratio=0.5,
        loss="huber",
        huber_delta=1.0,
    )
    return model, {
        "source_model": "synthetic",
        "source_revision": "test",
        "source_weights_sha256": "synthetic-source",
        "fingerprints": {"financial_pretrainer_after_transfer": module_fingerprint(model.backbone)},
    }


def _config_model():
    return SimpleNamespace(
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
    )


def test_pretraining_resume_is_bitwise_exact(tmp_path) -> None:
    train_dataset, selection_dataset = _datasets()
    config = _config(tmp_path)
    leakage = {
        "source_split": "train",
        "formal_validation_rows_used": 0,
        "formal_test_rows_used": 0,
    }
    full_dir = tmp_path / "full"
    resumed_dir = tmp_path / "resumed"

    _, full_audit, _ = train_financial_pretraining(
        config,
        train_dataset=train_dataset,
        selection_dataset=selection_dataset,
        leakage_audit=leakage,
        dataset_id="tiny-e3",
        checkpoint_dir=full_dir,
        initializer=_initializer,
    )
    train_financial_pretraining(
        config,
        train_dataset=train_dataset,
        selection_dataset=selection_dataset,
        leakage_audit=leakage,
        dataset_id="tiny-e3",
        checkpoint_dir=resumed_dir,
        stop_after_epoch=1,
        initializer=_initializer,
    )
    _, resumed_audit, _ = train_financial_pretraining(
        config,
        train_dataset=train_dataset,
        selection_dataset=selection_dataset,
        leakage_audit=leakage,
        dataset_id="tiny-e3",
        checkpoint_dir=resumed_dir,
        resume_from=resumed_dir / "last.pt",
        initializer=_initializer,
    )

    full = torch.load(full_dir / "last.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_dir / "last.pt", map_location="cpu", weights_only=False)
    assert full_audit["history"] == resumed_audit["history"]
    assert resumed_audit["resumed_from_epoch"] == 1
    assert full_audit["global_step"] == 6
    assert full_audit["history"][-1]["amp_skipped_optimizer_steps"] == 0
    for name, value in full["model_state"].items():
        torch.testing.assert_close(value, resumed["model_state"][name], rtol=0, atol=0)


def test_pretraining_residual_accumulation_group_uses_its_actual_size() -> None:
    group_sizes = [
        _batches_in_current_optimizer_step(
            batch_index, total_batches=5, configured_batches=2
        )
        for batch_index in range(1, 6)
    ]

    assert group_sizes == [2, 2, 2, 2, 1]


def test_pretraining_resume_rejects_legacy_patch_alignment_checkpoint(tmp_path) -> None:
    train_dataset, selection_dataset = _datasets()
    config = _config(tmp_path)
    leakage = {
        "source_split": "train",
        "formal_validation_rows_used": 0,
        "formal_test_rows_used": 0,
    }
    checkpoint_dir = tmp_path / "legacy"
    train_financial_pretraining(
        config,
        train_dataset=train_dataset,
        selection_dataset=selection_dataset,
        leakage_audit=leakage,
        dataset_id="tiny-e3",
        checkpoint_dir=checkpoint_dir,
        stop_after_epoch=1,
        initializer=_initializer,
    )
    checkpoint = torch.load(
        checkpoint_dir / "last.pt", map_location="cpu", weights_only=False
    )
    checkpoint["schema_version"] = 1
    checkpoint.pop("patch_alignment_protocol")
    legacy_path = checkpoint_dir / "legacy.pt"
    torch.save(checkpoint, legacy_path)

    with pytest.raises(ValueError, match="patch/mask alignment"):
        train_financial_pretraining(
            config,
            train_dataset=train_dataset,
            selection_dataset=selection_dataset,
            leakage_audit=leakage,
            dataset_id="tiny-e3",
            checkpoint_dir=checkpoint_dir,
            resume_from=legacy_path,
            initializer=_initializer,
        )
    with pytest.raises(ValueError, match="patch/mask alignment"):
        initialize_alpha_from_financial_checkpoint(
            config,
            context_length=train_dataset.context_length,
            checkpoint_path=legacy_path,
        )
