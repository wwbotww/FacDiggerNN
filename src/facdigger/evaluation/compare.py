"""Fair comparison of runs evaluated on exactly the same dataset keys."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import polars as pl

from facdigger.data.contracts import DataContractError


def _summary(metrics: dict[str, Any]) -> dict[str, Any]:
    raw = metrics["metrics"]["raw"]
    neutralized = metrics["metrics"].get("neutralized")
    return {
        "run_id": metrics["run_id"],
        "model_id": metrics["model_id"],
        "coverage": metrics["coverage"]["coverage"],
        "mean_ic": raw["ic"]["mean"],
        "mean_rank_ic": raw["rank_ic"]["mean"],
        "rank_icir": raw["rank_ic"]["ir"],
        "gross_q_high_minus_low": raw["portfolio"].get("gross_q_high_minus_low"),
        "mean_turnover": raw["portfolio"].get("mean_turnover"),
        "neutralized_mean_rank_ic": (
            neutralized["rank_ic"]["mean"] if neutralized is not None else None
        ),
        "research_ready": metrics["metrics"].get("cross_section", {}).get("research_ready"),
    }


def compare_runs(run_dirs: list[str | Path], output_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    if len(run_dirs) < 2:
        raise ValueError("compare requires at least two run directories")
    loaded: list[tuple[Path, dict[str, Any], pl.DataFrame]] = []
    for value in run_dirs:
        path = Path(value).resolve()
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        keys = pl.read_parquet(path / "predictions.parquet").select(
            "security_id", "asof_date", "target"
        )
        loaded.append((path, metrics, keys))
    dataset_ids = {item[1]["dataset_id"] for item in loaded}
    splits = {item[1]["evaluation_split"] for item in loaded}
    if len(dataset_ids) != 1 or len(splits) != 1:
        raise DataContractError("runs must use the same dataset_id and evaluation split")
    reference = loaded[0][2].sort(["asof_date", "security_id"])
    for path, _, keys in loaded[1:]:
        candidate = keys.sort(["asof_date", "security_id"])
        if not reference.equals(candidate, null_equal=True):
            raise DataContractError(f"prediction sample keys/targets differ for run: {path}")
    comparison = {
        "schema_version": 1,
        "dataset_id": next(iter(dataset_ids)),
        "evaluation_split": next(iter(splits)),
        "sample_rows": reference.height,
        "runs": [_summary(item[1]) for item in loaded],
    }
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "comparison.json"
    json_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    columns = [
        "model_id",
        "coverage",
        "mean_rank_ic",
        "rank_icir",
        "gross_q_high_minus_low",
        "mean_turnover",
        "neutralized_mean_rank_ic",
        "research_ready",
    ]
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(run.get(column)))}</td>" for column in columns)
        + "</tr>"
        for run in comparison["runs"]
    )
    (destination / "comparison.html").write_text(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<title>FacDigger Run Comparison</title><style>body{font-family:system-ui;"
        "max-width:1100px;margin:2rem auto}table{border-collapse:collapse}"
        "th,td{border:1px solid #ccc;padding:.45rem}</style></head><body>"
        f"<h1>Run Comparison</h1><p>dataset: {comparison['dataset_id']}</p>"
        f"<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"
        "</body></html>",
        encoding="utf-8",
    )
    return destination, comparison
