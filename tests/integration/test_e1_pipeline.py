from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("torch")
pytest.importorskip("transformers")

from facdigger.data.config import DatasetBuildConfig  # noqa: E402
from facdigger.data.snapshots import build_dataset_snapshot  # noqa: E402
from facdigger.inference.runner import run_inference, run_signal_inference  # noqa: E402
from facdigger.training.e1 import run_e1  # noqa: E402
from facdigger.training.e1_config import E1ExperimentConfig  # noqa: E402


def _sessions(count: int) -> list[date]:
    result = []
    current = date(2022, 1, 3)
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current)
        current += timedelta(days=1)
    return result


def test_e1_run_can_be_reloaded_for_bitwise_replay(tmp_path) -> None:
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
                    "source_revision": "synthetic-e1",
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
    bars_path = tmp_path / "bars.parquet"
    universe_path = tmp_path / "universe.parquet"
    pl.DataFrame(bars).write_parquet(bars_path)
    pl.DataFrame(universe).write_parquet(universe_path)
    snapshot, _ = build_dataset_snapshot(
        DatasetBuildConfig.model_validate(
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
    )
    config = E1ExperimentConfig.model_validate(
        {
            "experiment_id": "e1-replay-test",
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
                "max_epochs": 1,
                "minimum_epochs": 1,
                "patience": 1,
                "device": "cpu",
                "precision": "fp32",
            },
        }
    )
    run_dir, _ = run_e1(config, snapshot, repository_root=tmp_path)
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["supervised_selection_audit"][
        "outer_validation_rows_used_for_checkpoint_selection"
    ] == 0

    replay_dir, replay_manifest = run_inference(
        run_dir, output_dir=tmp_path / "replay", device="cpu"
    )

    assert replay_manifest["source_model_type"] == "random_patchtst"
    assert replay_manifest["replay_verification"]["matched"] is True
    assert (replay_dir / "factors.parquet").is_file()

    signal_dir, signal_manifest = run_signal_inference(
        run_dir, output_dir=tmp_path / "latest-signal", device="cpu"
    )
    factors = pl.read_parquet(signal_dir / "factors.parquet")
    assert signal_manifest["factor_contract"]["reads_labels"] is False
    assert factors["asof_date"].unique().to_list() == [calendar[-1]]
    assert "target" not in factors.columns
