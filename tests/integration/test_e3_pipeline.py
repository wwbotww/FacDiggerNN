from __future__ import annotations

import json
from datetime import date, timedelta
from types import SimpleNamespace

import polars as pl
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from facdigger.data.config import (  # noqa: E402
    DatasetBuildConfig,
    InferenceSnapshotConfig,
)
from facdigger.data.inference_snapshots import build_inference_snapshot  # noqa: E402
from facdigger.data.snapshots import build_dataset_snapshot, sha256_file  # noqa: E402
from facdigger.inference.factor_batch import load_factor_batch  # noqa: E402
from facdigger.inference.releases import create_model_release  # noqa: E402
from facdigger.inference.runner import run_inference, run_signal_inference  # noqa: E402
from facdigger.models.patchtst_pretrain import FinancialPatchTSTPretrainer  # noqa: E402
from facdigger.models.patchtst_transfer import module_fingerprint  # noqa: E402
from facdigger.training.e3 import run_e3  # noqa: E402
from facdigger.training.e3_config import E3ExperimentConfig  # noqa: E402


def _sessions(count: int) -> list[date]:
    sessions = []
    current = date(2022, 1, 3)
    while len(sessions) < count:
        if current.weekday() < 5:
            sessions.append(current)
        current += timedelta(days=1)
    return sessions


def _snapshot(tmp_path):
    calendar = _sessions(85)
    bars = []
    universe = []
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
                    "source_revision": "synthetic-e3",
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
    config = DatasetBuildConfig.model_validate(
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
    return build_dataset_snapshot(config)[0]


def _experiment(tmp_path) -> E3ExperimentConfig:
    return E3ExperimentConfig.model_validate(
        {
            "experiment_id": "e3-test",
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
            "pretraining": {
                "batch_size": 64,
                "max_epochs": 1,
                "minimum_epochs": 1,
                "patience": 1,
                "mask_ratio": 0.5,
                "device": "cpu",
                "precision": "fp32",
                "validation_fraction": 0.2,
            },
            "finetuning": {
                "batch_size": 64,
                "max_epochs": 2,
                "patience": 2,
                "minimum_epochs": 2,
                "head_only_epochs": 1,
                "unfreeze_last_n_blocks": 1,
                "device": "cpu",
                "precision": "fp32",
                "objective": {
                    "minimum_cross_section_size": 2,
                    "minimum_selection_dates": 2,
                    "minimum_selection_coverage": 1.0,
                },
            },
        }
    )


def _initializer() -> tuple[FinancialPatchTSTPretrainer, dict]:
    model_config = SimpleNamespace(
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
    model = FinancialPatchTSTPretrainer(
        context_length=20,
        num_input_channels=7,
        model_config=model_config,
        mask_ratio=0.5,
        loss="huber",
        huber_delta=1.0,
    )
    return model, {
        "schema_version": 2,
        "source_model": "synthetic-test-source",
        "source_revision": "test",
        "source_weights_sha256": "synthetic-source-hash",
        "fingerprints": {
            "financial_pretrainer_after_transfer": module_fingerprint(model.backbone)
        },
    }


def test_e3_runner_writes_pretraining_chain_and_evaluation_artifacts(tmp_path) -> None:
    snapshot = _snapshot(tmp_path)
    run_dir, metrics = run_e3(
        _experiment(tmp_path),
        snapshot,
        repository_root=tmp_path,
        pretraining_initializer=_initializer,
    )

    assert metrics["coverage"]["coverage"] == 1.0
    for filename in [
        "manifest.json",
        "resolved_config.yaml",
        "pretraining/checkpoints/best.pt",
        "pretraining/checkpoints/last.pt",
        "pretraining/training_audit.json",
        "pretraining/weight_load_report.json",
        "checkpoints/best.pt",
        "checkpoints/last.pt",
        "weight_load_report.json",
        "predictions.parquet",
        "metrics.json",
        "report.html",
    ]:
        assert (run_dir / filename).is_file()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["pretraining_leakage_audit"]["formal_validation_rows_used"] == 0
    assert manifest["pretraining_leakage_audit"]["formal_test_rows_used"] == 0
    assert manifest["weight_loading"]["financial_backbone_to_alpha"][
        "loaded_numel_ratio"
    ] == 1.0
    assert manifest["finetuning"]["stage_audits"]["ft0_head_only"][
        "encoder_changed"
    ] is False
    assert manifest["finetuning"]["stage_audits"]["ft1_last_blocks"][
        "encoder_changed"
    ] is True
    assert manifest["input"]["feature_scaler_sha256"] == sha256_file(
        snapshot / "scaler.json"
    )
    assert manifest["predictions_sha256"] == sha256_file(
        run_dir / "predictions.parquet"
    )

    replay_dir, replay_manifest = run_inference(
        run_dir, output_dir=tmp_path / "replay", device="cpu"
    )
    assert replay_manifest["replay_verification"]["matched"] is True
    assert (replay_dir / "predictions.parquet").is_file()
    assert not (replay_dir / "factors.parquet").exists()


def test_e3_release_inference_snapshot_to_factor_batch(tmp_path, monkeypatch) -> None:
    snapshot = _snapshot(tmp_path)
    run_dir, _ = run_e3(
        _experiment(tmp_path),
        snapshot,
        repository_root=tmp_path,
        pretraining_initializer=_initializer,
    )
    commit = "1" * 40
    run_manifest_path = run_dir / "manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["git"] = {"commit": commit, "branch": "test", "dirty": False}
    run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
    monkeypatch.setattr(
        "facdigger.inference.releases.collect_git_state",
        lambda _: {
            "commit": commit,
            "branch": "test",
            "dirty": False,
            "status_porcelain": "",
        },
    )
    release_dir, release = create_model_release(
        run_dir, tmp_path / "releases", repository_root=tmp_path
    )
    training_manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    source_paths = training_manifest["source_paths"]
    inference_dir, inference_manifest = build_inference_snapshot(
        InferenceSnapshotConfig.model_validate(
            {
                "sources": {
                    "bars": source_paths["bars"],
                    "universe": source_paths["universe"],
                },
                "output_root": tmp_path / "inference-snapshots",
            }
        ),
        release_dir,
    )
    batch_dir, batch_manifest = run_signal_inference(
        release_dir,
        dataset_dir=inference_dir,
        output_root=tmp_path / "factor-batches",
        device="cpu",
    )

    factors = pl.read_parquet(batch_dir / "factors.parquet")
    assert release.model_type == "financial_pretrained_patchtst"
    assert inference_manifest["feature_contract"]["scaler_sha256"] == (
        release.feature_contract.scaler_sha256
    )
    assert "labels" not in inference_manifest["artifacts"]
    assert "sample_index" not in inference_manifest["artifacts"]
    assert factors.columns == ["security_id", "symbol", "asof_date", "score", "eligible"]
    assert factors["asof_date"].unique().to_list() == [_sessions(85)[-1]]
    assert batch_manifest["source"]["kind"] == "signal_inference"
    assert batch_manifest["model"]["release_id"] == release.release_id
    assert batch_manifest["input"]["snapshot_id"] == inference_manifest["snapshot_id"]
    assert load_factor_batch(batch_dir).delivery_id == batch_manifest["delivery_id"]
