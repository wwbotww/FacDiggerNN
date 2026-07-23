from __future__ import annotations

import json
from datetime import date, timedelta

import polars as pl
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from facdigger.data.config import DatasetBuildConfig  # noqa: E402
from facdigger.data.snapshots import build_dataset_snapshot  # noqa: E402
from facdigger.models.patchtst_alpha import PatchTSTAlphaModel  # noqa: E402
from facdigger.models.patchtst_transfer import module_fingerprint  # noqa: E402
from facdigger.training.e2 import run_e2  # noqa: E402
from facdigger.training.e2_config import E2ExperimentConfig  # noqa: E402


def _sessions(count: int) -> list[date]:
    result: list[date] = []
    current = date(2022, 1, 3)
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current)
        current += timedelta(days=1)
    return result


def _initializer() -> tuple[PatchTSTAlphaModel, dict]:
    model = PatchTSTAlphaModel(
        context_length=20,
        num_input_channels=7,
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
    fingerprint = module_fingerprint(model.backbone)
    return model, {
        "schema_version": 1,
        "source_model": "synthetic-test-source",
        "source_revision": "test",
        "source_weights_sha256": "synthetic-source-hash",
        "checkpoint_to_library": {"loaded_numel_ratio": 1.0},
        "source_backbone_to_target": {"loaded_numel_ratio": 1.0},
        "fingerprints": {"target_backbone_after_transfer": fingerprint},
    }


def test_e2_runner_writes_transfer_training_and_evaluation_artifacts(tmp_path) -> None:
    calendar = _sessions(85)
    bars: list[dict] = []
    universe: list[dict] = []
    for security_index in range(6):
        security_id = f"sec-{security_index}"
        for index, trade_date in enumerate(calendar):
            close = 20.0 + security_index * 3 + index * (0.02 + security_index * 0.002)
            volume = 1_000_000.0 + security_index * 100_000 + index
            bars.append(
                {
                    "security_id": security_id,
                    "symbol": f"S{security_index}",
                    "trade_date": trade_date,
                    "open": close - 0.05,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": volume,
                    "dollar_volume": close * volume,
                    "adj_factor": 1.0,
                    "source_revision": "synthetic-e2",
                }
            )
            universe.append(
                {
                    "security_id": security_id,
                    "symbol": f"S{security_index}",
                    "trade_date": trade_date,
                    "listed_days": 300 + index,
                    "exchange": "XNAS",
                    "security_type": "common_stock",
                    "is_primary_listing": True,
                    "is_listed": True,
                    "is_delisted": False,
                    "is_halted": False,
                    "industry_code": "TECH" if security_index < 3 else "FIN",
                    "float_market_cap": float(1_000_000_000 * (security_index + 1)),
                    "close": close,
                    "adv20_usd": close * volume,
                    "eligible": True,
                }
            )
    bronze = tmp_path / "bronze"
    bronze.mkdir()
    bars_path = bronze / "bars.parquet"
    universe_path = bronze / "universe.parquet"
    pl.DataFrame(bars).write_parquet(bars_path)
    pl.DataFrame(universe).write_parquet(universe_path)
    dataset_config = DatasetBuildConfig.model_validate(
        {
            "sources": {"bars": bars_path, "universe": universe_path},
            "output_root": tmp_path / "snapshots",
            "features": {"context_length": 20},
            "split": {
                "train_end": calendar[42],
                "valid_end": calendar[62],
                "test_end": calendar[80],
                "embargo_sessions": 2,
            },
        }
    )
    snapshot_dir, _ = build_dataset_snapshot(dataset_config)
    experiment = E2ExperimentConfig.model_validate(
        {
            "experiment_id": "e2-test",
            "output_root": tmp_path / "runs",
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
                "batch_size": 64,
                "max_epochs": 2,
                "patience": 2,
                "minimum_epochs": 2,
                "head_only_epochs": 1,
                "unfreeze_last_n_blocks": 1,
                "device": "cpu",
                "precision": "fp32",
            },
        }
    )

    run_dir, metrics = run_e2(
        experiment,
        snapshot_dir,
        repository_root=tmp_path,
        initializer=_initializer,
    )

    assert metrics["coverage"]["coverage"] == 1.0
    for filename in [
        "manifest.json",
        "resolved_config.yaml",
        "weight_load_report.json",
        "predictions.parquet",
        "metrics.json",
        "report.html",
        "checkpoints/best.pt",
        "checkpoints/last.pt",
    ]:
        assert (run_dir / filename).is_file()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["training"]["stage_audits"]["ft0_head_only"]["encoder_changed"] is False
    assert manifest["training"]["stage_audits"]["ft1_last_blocks"]["encoder_changed"] is True
