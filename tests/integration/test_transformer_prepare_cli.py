from __future__ import annotations

import json

import yaml
from test_dataset_pipeline import sessions, synthetic_frames
from typer.testing import CliRunner

from facdigger.cli import app
from facdigger.data.config import FINANCE_TRANSFORMER_CHANNELS, MARKET_CONTEXT_CHANNELS
from facdigger.research.transformer_config import load_transformer_comparison_config
from facdigger.research.transformer_snapshots import build_transformer_snapshots
from facdigger.training.runtime import load_training_runtime


def test_real_cpu_fold_preparation_can_train_without_bronze(tmp_path):
    bars, universe = synthetic_frames(150)
    bars_path, universe_path = tmp_path / "bars.parquet", tmp_path / "universe.parquet"
    bars.write_parquet(bars_path)
    universe.write_parquet(universe_path)
    calendar = sessions(150)
    folds = [
        {
            "fold_id": f"wf{index}",
            "train_end": calendar[train].isoformat(),
            "valid_end": calendar[valid].isoformat(),
            "test_end": calendar[test].isoformat(),
            "embargo_sessions": 2,
        }
        for index, (train, valid, test) in enumerate(
            [(40, 60, 80), (60, 80, 100), (80, 105, 145)], 1
        )
    ]
    base_path = tmp_path / "dataset.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "sources": {"bars": str(bars_path), "universe": str(universe_path)},
                "features": {
                    "name": "finance_transformer",
                    "context_length": 20,
                    "channels": FINANCE_TRANSFORMER_CHANNELS,
                    "market_channels": MARKET_CONTEXT_CHANNELS,
                },
                "label": {"horizon": 5, "auxiliary_horizons": [1, 20]},
                "split": {key: value for key, value in folds[-1].items() if key != "fold_id"},
            }
        )
    )
    research_path = tmp_path / "research.yaml"
    research_path.write_text(
        yaml.safe_dump(
            {
                "base_dataset_config": str(base_path),
                "snapshot_output_root": str(tmp_path / "snapshots"),
                "folds": folds,
                "experiments": {
                    "scratch": "absent.yaml",
                    "pretrained": "absent.yaml",
                    "pretraining": "absent.yaml",
                },
            }
        )
    )
    output = tmp_path / "prepared"
    runner = CliRunner()
    args = [
        "research",
        "transformer-prepare",
        "--config",
        str(research_path),
        "--output",
        str(output),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert len({plan["dataset_id"] for plan in report["folds"]}) == 3
    bars_path.unlink()
    universe_path.unlink()
    # Reentry and the training runner's shared reader both use the frozen assets.
    repeated = runner.invoke(app, args)
    assert repeated.exit_code == 0, repeated.output
    plans = build_transformer_snapshots(
        load_transformer_comparison_config(research_path),
        load_training_runtime(output / "runtime.yaml"),
    )
    assert plans == report["folds"]
