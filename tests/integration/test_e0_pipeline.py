from __future__ import annotations

import json
import shutil
from datetime import date, timedelta

import polars as pl
import pytest

from facdigger.data.config import DatasetBuildConfig
from facdigger.data.snapshots import build_dataset_snapshot
from facdigger.evaluation.runner import evaluate_prediction_file
from facdigger.inference.runner import run_inference, run_signal_inference
from facdigger.training.e0 import run_e0
from facdigger.training.e0_config import E0ExperimentConfig

pytest.importorskip("torch")


def sessions(count: int) -> list[date]:
    result: list[date] = []
    current = date(2022, 1, 3)
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current)
        current += timedelta(days=1)
    return result


def test_e0_mlp_train_predict_report_pipeline(tmp_path) -> None:
    calendar = sessions(85)
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
                    "source_revision": "synthetic-e0",
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
    experiment = E0ExperimentConfig.model_validate(
        {
            "experiment_id": "e0-test",
            "output_root": tmp_path / "runs",
            "windows": [5, 20],
            "mlp": {
                "hidden_dims": [16],
                "dropout": 0.0,
                "max_epochs": 4,
                "patience": 2,
                "batch_size": 64,
                "device": "cpu",
            },
        }
    )
    run_dir, metrics = run_e0(experiment, snapshot_dir, repository_root=tmp_path)

    assert metrics["coverage"]["coverage"] == 1.0
    assert metrics["evaluation_split"] == "valid"
    for filename in [
        "manifest.json",
        "resolved_config.yaml",
        "predictions.parquet",
        "metrics.json",
        "report.html",
        "checkpoints/best.pt",
    ]:
        assert (run_dir / filename).is_file()
    predictions = pl.read_parquet(run_dir / "predictions.parquet")
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert predictions["score_raw"].is_finite().all()
    assert predictions["score_neutralized"].is_not_null().all()
    assert run_manifest["supervised_selection_audit"][
        "outer_validation_rows_used_for_checkpoint_selection"
    ] == 0
    assert run_manifest["row_counts"]["inner_selection"] > 0

    replay_dir, replay_manifest = run_inference(
        run_dir, output_dir=tmp_path / "replay", device="cpu"
    )
    assert replay_manifest["replay_verification"]["matched"] is True
    factors = pl.read_parquet(replay_dir / "factors.parquet")
    assert "target" not in factors.columns
    assert factors["signal_available"].unique().to_list() == ["after_close"]
    assert factors["earliest_execution"].unique().to_list() == ["next_session_open"]

    signal_dir, signal_manifest = run_signal_inference(
        run_dir, output_dir=tmp_path / "latest-signal", device="cpu"
    )
    latest_factors = pl.read_parquet(signal_dir / "factors.parquet")
    assert signal_manifest["factor_contract"]["reads_labels"] is False
    assert signal_manifest["factor_contract"]["reads_test_membership"] is False
    assert latest_factors["asof_date"].unique().to_list() == [calendar[-1]]
    assert "target" not in latest_factors.columns
    assert latest_factors.height == 6

    target_free_snapshot = tmp_path / "target-free-snapshot"
    target_free_snapshot.mkdir()
    for filename in ["manifest.json", "features.parquet", "inference_index.parquet"]:
        shutil.copyfile(snapshot_dir / filename, target_free_snapshot / filename)
    _, isolated_signal_manifest = run_signal_inference(
        run_dir,
        dataset_dir=target_free_snapshot,
        output_dir=tmp_path / "isolated-latest-signal",
        device="cpu",
    )
    assert isolated_signal_manifest["row_count"] == 6

    evaluation_dir, evaluation_manifest = evaluate_prediction_file(
        run_dir / "predictions.parquet",
        snapshot_dir,
        tmp_path / "independent-evaluation",
    )
    assert evaluation_manifest["coverage"]["coverage"] == 1.0
    assert (evaluation_dir / "metrics.json").is_file()
    assert (evaluation_dir / "report.html").is_file()

    with pytest.raises(ValueError, match="unlock_test=true"):
        run_inference(run_dir, split="test", output_dir=tmp_path / "locked-test")
    with pytest.raises(FileExistsError, match="already exists"):
        run_inference(run_dir, output_dir=replay_dir)
