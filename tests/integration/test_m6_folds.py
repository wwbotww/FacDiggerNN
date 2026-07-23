from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import yaml

from facdigger.research.config import M6ResearchConfig
from facdigger.research.folds import build_walk_forward_snapshots


def _sessions(count: int) -> list[date]:
    result = []
    current = date(2020, 1, 1)
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current)
        current += timedelta(days=1)
    return result


def test_walk_forward_rebuilds_fold_specific_content_addressed_snapshots(tmp_path) -> None:
    calendar = _sessions(140)
    bars = []
    universe = []
    for security_index in range(3):
        security_id = f"sec-{security_index}"
        for index, day in enumerate(calendar):
            close = 20.0 + security_index + index * (0.02 + security_index * 0.001)
            volume = 1_000_000.0 + index
            bars.append(
                {
                    "security_id": security_id,
                    "symbol": f"S{security_index}",
                    "trade_date": day,
                    "open": close - 0.05,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": volume,
                    "dollar_volume": close * volume,
                    "adj_factor": 1.0,
                    "source_revision": "m6-test",
                }
            )
            universe.append(
                {
                    "security_id": security_id,
                    "symbol": f"S{security_index}",
                    "trade_date": day,
                    "listed_days": 500 + index,
                    "exchange": "XNAS",
                    "security_type": "common_stock",
                    "is_primary_listing": True,
                    "is_listed": True,
                    "is_delisted": False,
                    "is_halted": False,
                    "industry_code": "TECH",
                    "float_market_cap": 1_000_000_000.0,
                    "close": close,
                    "adv20_usd": close * volume,
                    "eligible": True,
                }
            )
    bars_path = tmp_path / "bars.parquet"
    universe_path = tmp_path / "universe.parquet"
    pl.DataFrame(bars).write_parquet(bars_path)
    pl.DataFrame(universe).write_parquet(universe_path)
    dataset_config_path = tmp_path / "dataset.yaml"
    dataset_config_path.write_text(
        yaml.safe_dump(
            {
                "sources": {"bars": str(bars_path), "universe": str(universe_path)},
                "output_root": str(tmp_path / "unused"),
                "features": {"context_length": 20},
                "split": {
                    "train_end": calendar[50],
                    "valid_end": calendar[70],
                    "test_end": calendar[90],
                    "embargo_sessions": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    model_paths = {}
    for model in ["e0", "e1", "e2", "e3"]:
        path = tmp_path / f"{model}.yaml"
        path.write_text("{}\n", encoding="utf-8")
        model_paths[model] = path
    config = M6ResearchConfig.model_validate(
        {
            "base_dataset_config": dataset_config_path,
            "snapshot_output_root": tmp_path / "fold-snapshots",
            "models": model_paths,
            "seeds": [1, 2, 3],
            "folds": [
                {
                    "fold_id": f"fold-{index}",
                    "train_end": calendar[50 + index * 20],
                    "valid_end": calendar[70 + index * 20],
                    "test_end": calendar[90 + index * 20],
                    "embargo_sessions": 2,
                }
                for index in range(3)
            ],
        }
    )

    first = build_walk_forward_snapshots(config)
    second = build_walk_forward_snapshots(config)

    assert first == second
    assert len({fold["dataset_id"] for fold in first}) == 3
    for fold, configured in zip(first, config.folds, strict=True):
        root = Path(fold["dataset_path"])
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["config"]["split"]["train_end"] == configured.train_end.isoformat()
        assert (root / "scaler.json").is_file()
