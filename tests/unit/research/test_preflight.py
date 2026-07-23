from __future__ import annotations

import json
from datetime import date

import polars as pl

from facdigger.research.config import M6ResearchConfig
from facdigger.research.preflight import research_preflight


def test_preflight_blocks_source_not_marked_research_ready(tmp_path) -> None:
    calendar = [date(2020, 1, 1), date(2021, 1, 1), date(2022, 1, 1), date(2023, 1, 1)]
    bars = tmp_path / "bars.parquet"
    universe = tmp_path / "universe.parquet"
    source_manifest = tmp_path / "source.json"
    pl.DataFrame({"trade_date": calendar}).write_parquet(bars)
    pl.DataFrame({"trade_date": calendar}).write_parquet(universe)
    source_manifest.write_text(
        json.dumps({"provider": "test", "selection": {"research_ready": False}}),
        encoding="utf-8",
    )
    dataset_config = tmp_path / "dataset.yaml"
    dataset_config.write_text(
        f"""sources:
  bars: {bars}
  universe: {universe}
  source_manifest: {source_manifest}
features:
  context_length: 20
split:
  train_end: 2020-01-01
  valid_end: 2021-01-01
  test_end: 2022-01-01
""",
        encoding="utf-8",
    )
    model_paths = {}
    for model in ["e0", "e1", "e2", "e3"]:
        path = tmp_path / f"{model}.yaml"
        path.write_text("experiment_id: test\n", encoding="utf-8")
        model_paths[model] = path
    config = M6ResearchConfig.model_validate(
        {
            "base_dataset_config": dataset_config,
            "seeds": [1, 2, 3],
            "folds": [
                {
                    "fold_id": "f1",
                    "train_end": "2020-01-01",
                    "valid_end": "2020-06-01",
                    "test_end": "2021-01-01",
                },
                {
                    "fold_id": "f2",
                    "train_end": "2020-06-01",
                    "valid_end": "2021-01-01",
                    "test_end": "2022-01-01",
                },
                {
                    "fold_id": "f3",
                    "train_end": "2021-01-01",
                    "valid_end": "2022-01-01",
                    "test_end": "2023-01-01",
                },
            ],
            "models": model_paths,
        }
    )

    report = research_preflight(config)

    assert report["ready"] is False
    assert report["checks"]["source_research_ready"] is False
    assert "research_ready=true" in report["blockers"][0]
