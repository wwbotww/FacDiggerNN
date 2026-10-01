from __future__ import annotations

import json
import subprocess
import sys

import yaml
from test_dataset_pipeline import sessions, synthetic_frames
from typer.testing import CliRunner

from facdigger.cli import app
from facdigger.data.config import FINANCE_TRANSFORMER_CHANNELS, MARKET_CONTEXT_CHANNELS
from facdigger.research.transformer_config import load_transformer_comparison_config
from facdigger.research.transformer_snapshots import build_transformer_snapshots
from facdigger.training.runtime import load_training_runtime


def test_real_cpu_fold_preparation_can_train_without_bronze(tmp_path):
    bars, universe = synthetic_frames(260)
    bars_path, universe_path = tmp_path / "bars.parquet", tmp_path / "universe.parquet"
    bars.write_parquet(bars_path)
    universe.write_parquet(universe_path)
    calendar = sessions(260)
    folds = [
        {
            "fold_id": f"wf{index}",
            "train_end": calendar[train].isoformat(),
            "valid_end": calendar[valid].isoformat(),
            "test_end": calendar[test].isoformat(),
            "embargo_sessions": 2,
        }
        for index, (train, valid, test) in enumerate(
            [(105, 140, 175), (140, 175, 210), (175, 210, 250)], 1
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
                "finance_selection": {
                    "supervised_selection_fraction": 0.15,
                    "probe_fit_dates": 2,
                    "probe_selection_dates": 2,
                    "future_horizon": 5,
                },
                "split": {key: value for key, value in folds[-1].items() if key != "fold_id"},
            }
        )
    )
    experiment_paths = {}
    for stage in ("scratch", "pretrained", "pretraining"):
        experiment = {"model": {"statistics_windows": [5, 20]}}
        if stage == "pretrained":
            experiment.update(
                {
                    "initialization": "finance_pretrained",
                    "pretrained_checkpoint": str(tmp_path / "not-trained-encoder.pt"),
                }
            )
        elif stage == "pretraining":
            experiment["training"] = {"probe": {"fit_dates": 2, "selection_dates": 2}}
        path = tmp_path / f"{stage}.yaml"
        path.write_text(yaml.safe_dump(experiment))
        experiment_paths[stage] = str(path)
    research_path = tmp_path / "research.yaml"
    research_path.write_text(
        yaml.safe_dump(
            {
                "base_dataset_config": str(base_path),
                "snapshot_output_root": str(tmp_path / "snapshots"),
                "folds": folds,
                "experiments": experiment_paths,
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
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from typer.testing import CliRunner
from facdigger.cli import app
result = CliRunner().invoke(app, sys.argv[1:])
assert result.exit_code == 0, result.output
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
print(result.output)
""",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(process.stdout)
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
