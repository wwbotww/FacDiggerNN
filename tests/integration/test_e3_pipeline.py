from __future__ import annotations

import json
from types import SimpleNamespace

import polars as pl
import pytest
from factor_fixtures import build_price_volume_snapshot, sessions

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from facdigger.data.config import (  # noqa: E402
    InferenceSnapshotConfig,
)
from facdigger.data.contracts import DataContractError  # noqa: E402
from facdigger.data.inference_snapshots import build_inference_snapshot  # noqa: E402
from facdigger.data.snapshots import sha256_file  # noqa: E402
from facdigger.inference.factor_batch import load_factor_batch  # noqa: E402
from facdigger.inference.history import (  # noqa: E402
    HistoricalReplayConfig,
    run_historical_replay,
    verify_historical_replay,
)
from facdigger.inference.releases import create_model_release  # noqa: E402
from facdigger.inference.runner import run_inference, run_signal_inference  # noqa: E402
from facdigger.models.patchtst_pretrain import FinancialPatchTSTPretrainer  # noqa: E402
from facdigger.models.patchtst_transfer import module_fingerprint  # noqa: E402
from facdigger.training.e3 import run_e3  # noqa: E402
from facdigger.training.e3_config import E3ExperimentConfig  # noqa: E402


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
        "fingerprints": {"financial_pretrainer_after_transfer": module_fingerprint(model.backbone)},
    }


def test_e3_runner_writes_pretraining_chain_and_evaluation_artifacts(tmp_path) -> None:
    snapshot = build_price_volume_snapshot(tmp_path)
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
    assert manifest["weight_loading"]["financial_backbone_to_alpha"]["loaded_numel_ratio"] == 1.0
    assert manifest["finetuning"]["stage_audits"]["ft0_head_only"]["encoder_changed"] is False
    assert manifest["finetuning"]["stage_audits"]["ft1_last_blocks"]["encoder_changed"] is True
    assert manifest["input"]["feature_scaler_sha256"] == sha256_file(snapshot / "scaler.json")
    assert manifest["predictions_sha256"] == sha256_file(run_dir / "predictions.parquet")

    replay_dir, replay_manifest = run_inference(
        run_dir, output_dir=tmp_path / "replay", device="cpu"
    )
    assert replay_manifest["replay_verification"]["matched"] is True
    assert (replay_dir / "predictions.parquet").is_file()
    assert not (replay_dir / "factors.parquet").exists()


def test_e3_release_inference_snapshot_to_factor_batch(tmp_path, monkeypatch) -> None:
    snapshot = build_price_volume_snapshot(tmp_path)
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
    assert factors["asof_date"].unique().to_list() == [sessions(85)[-1]]
    assert batch_manifest["source"]["kind"] == "signal_inference"
    assert batch_manifest["model"]["release_id"] == release.release_id
    assert batch_manifest["input"]["snapshot_id"] == inference_manifest["snapshot_id"]
    assert load_factor_batch(batch_dir).delivery_id == batch_manifest["delivery_id"]

    history_config = HistoricalReplayConfig.model_validate(
        {
            "history_id": "e3-integration-history",
            "release_dir": release_dir,
            "inference_snapshot_dir": inference_dir,
            "output_root": tmp_path / "factor-history",
            "device": "cpu",
            "batch_size": 64,
            "num_workers": 0,
            "acknowledge_non_oos": True,
        }
    )
    overlapping_config = history_config.model_copy(update={"output_root": inference_dir})
    with pytest.raises(DataContractError, match="must not overlap"):
        run_historical_replay(overlapping_config)
    assert not (inference_dir / history_config.history_id).exists()

    first_progress = []
    history_dir, history_manifest = run_historical_replay(
        history_config,
        on_partition=lambda status, year, partition: first_progress.append(
            (status, year, partition)
        ),
    )

    assert [item[0] for item in first_progress] == ["scoring", "published"]
    assert history_manifest.purpose == "backtest_only"
    assert history_manifest.strict_out_of_sample is False
    assert history_manifest.model_parameters_may_use_future_data is True
    assert history_manifest.scaler_may_use_future_data is True
    assert history_manifest.source_data_may_include_later_revisions is True
    assert history_manifest.paper_allowed is False
    assert history_manifest.release_id == release.release_id
    assert len(history_manifest.partitions) == 1
    history_batch_dir = history_dir / history_manifest.partitions[0].path
    history_batch = load_factor_batch(history_batch_dir)
    history_factors = pl.read_parquet(history_batch_dir / "factors.parquet")
    assert history_batch.source.kind == "evaluation_predictions"
    assert history_factors["asof_date"].n_unique() > 1
    assert history_factors["eligible"].all()

    latest_history = history_factors.filter(pl.col("asof_date") == factors["asof_date"].max())
    assert latest_history.select("security_id", "symbol", "asof_date").equals(
        factors.select("security_id", "symbol", "asof_date")
    )
    maximum_score_delta = (
        latest_history.join(
            factors.select("security_id", "asof_date", pl.col("score").alias("daily")),
            on=["security_id", "asof_date"],
            validate="1:1",
        )
        .select((pl.col("score") - pl.col("daily")).abs().max())
        .item()
    )
    assert maximum_score_delta < 1e-6
    assert verify_historical_replay(history_dir) == history_manifest

    # Simulate a crash after the annual child and state were committed but before
    # the aggregate manifest became visible. Resume must verify and reuse the child.
    manifest_path = history_dir / "manifest.json"
    manifest_path.unlink()
    resume_progress = []
    resumed_dir, resumed_manifest = run_historical_replay(
        history_config,
        on_partition=lambda status, year, partition: resume_progress.append(
            (status, year, partition)
        ),
    )
    assert resumed_dir == history_dir
    assert [item[0] for item in resume_progress] == ["verified"]
    assert resumed_manifest.partitions == history_manifest.partitions
    assert {path.name for path in (history_dir / "factor_batches").iterdir()} == {
        history_manifest.partitions[0].delivery_id
    }

    completed_dir, completed_manifest = run_historical_replay(history_config)
    assert completed_dir == history_dir
    assert completed_manifest == resumed_manifest

    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["partitions"][0]["path"] = "../outside"
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(DataContractError, match="year or path"):
        verify_historical_replay(history_dir)
