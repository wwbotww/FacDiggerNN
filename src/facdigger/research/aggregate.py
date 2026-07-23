"""Aggregate a frozen E0-E3 fold/seed matrix into auditable research conclusions."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl

from facdigger.data.contracts import DataContractError
from facdigger.research.config import M6ResearchConfig
from facdigger.research.statistics import panel_mean_inference

MODEL_KEYS = ("e0", "e1", "e2", "e3")
PAIRED_COMPARISONS = {
    "architecture_e1_vs_e0": ("e1", "e0"),
    "external_transfer_e2_vs_e1": ("e2", "e1"),
    "financial_pretraining_e3_vs_e2": ("e3", "e2"),
    "overall_e3_vs_e1": ("e3", "e1"),
}


def _load_cell(cell: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(cell["run_dir"]).resolve()
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.parquet"
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(f"research cell artifacts missing: {run_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics["dataset_id"] != cell["dataset_id"]:
        raise DataContractError(f"research cell dataset_id mismatch: {run_dir}")
    if metrics["evaluation_split"] != cell["evaluation_split"]:
        raise DataContractError(f"research cell evaluation split mismatch: {run_dir}")
    predictions = pl.read_parquet(predictions_path).select(
        "security_id", "asof_date", "target"
    ).sort(["asof_date", "security_id"])
    return {**cell, "run_dir": str(run_dir), "metrics": metrics, "keys": predictions}


def _validate_matrix(
    loaded: list[dict[str, Any]],
    config: M6ResearchConfig,
    evaluation_split: str,
    active_folds: list[str],
) -> None:
    expected = {
        (fold_id, seed, model)
        for fold_id in active_folds
        for seed in config.seeds
        for model in MODEL_KEYS
    }
    actual = {(cell["fold_id"], cell["seed"], cell["model_key"]) for cell in loaded}
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise DataContractError(
            f"research matrix is incomplete; missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    if any(cell["evaluation_split"] != evaluation_split for cell in loaded):
        raise DataContractError("research matrix mixes evaluation splits")
    for fold_id in active_folds:
        fold_cells = [cell for cell in loaded if cell["fold_id"] == fold_id]
        dataset_ids = {cell["dataset_id"] for cell in fold_cells}
        if len(dataset_ids) != 1:
            raise DataContractError(f"fold {fold_id} mixes dataset snapshots")
        reference = fold_cells[0]["keys"]
        for cell in fold_cells[1:]:
            if not reference.equals(cell["keys"], null_equal=True):
                raise DataContractError(
                    f"fold {fold_id} prediction keys differ: {cell['run_dir']}"
                )


def _daily_map(metrics: dict[str, Any], section: str, field: str) -> dict[str, float]:
    score = metrics["metrics"].get(section)
    if score is None:
        return {}
    result = {}
    for row in score[field]:
        value_key = "rank_ic" if field == "daily_ic" else field
        value = row.get(value_key)
        if value is not None:
            result[str(row["asof_date"])] = float(value)
    return result


def _portfolio_map(metrics: dict[str, Any], cost_bps: float) -> dict[str, float]:
    key = f"net_{cost_bps:g}bps"
    result = {}
    for row in metrics["metrics"]["raw"]["daily_portfolio"]:
        value = row.get(key)
        if value is not None:
            result[str(row["asof_date"])] = float(value)
    return result


def _average_seed_series(
    loaded: list[dict[str, Any]], extractor: Any
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for cell in loaded:
        for asof_date, value in extractor(cell["metrics"]).items():
            grouped[(cell["fold_id"], asof_date)].append(value)
    ordered = sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))
    by_fold: dict[str, list[float]] = defaultdict(list)
    for (fold_id, _), values in ordered:
        by_fold[fold_id].append(sum(values) / len(values))
    return {
        "values": [sum(values) / len(values) for _, values in ordered],
        "dates": [f"{fold_id}:{asof_date}" for (fold_id, asof_date), _ in ordered],
        "fold_values": [by_fold[fold_id] for fold_id in sorted(by_fold)],
    }


def _inference(config: M6ResearchConfig, groups: list[list[float]]) -> dict[str, Any]:
    return panel_mean_inference(
        groups,
        hac_lags=config.hac_lags,
        stride=config.non_overlapping_stride,
        offset=config.non_overlapping_offset,
    )


def _paired_delta(
    loaded: list[dict[str, Any]],
    config: M6ResearchConfig,
    left: str,
    right: str,
    active_folds: list[str],
) -> dict[str, Any]:
    indexed = {
        (cell["fold_id"], cell["seed"], cell["model_key"]): cell for cell in loaded
    }
    daily_by_fold_date: dict[tuple[str, str], list[float]] = defaultdict(list)
    cell_means = []
    for fold_id in active_folds:
        for seed in config.seeds:
            left_daily = _daily_map(
                indexed[(fold_id, seed, left)]["metrics"], "raw", "daily_ic"
            )
            right_daily = _daily_map(
                indexed[(fold_id, seed, right)]["metrics"], "raw", "daily_ic"
            )
            dates = sorted(set(left_daily) & set(right_daily))
            if not dates:
                raise DataContractError(
                    f"paired comparison has no dates: {fold_id}/{seed}/{left}-{right}"
                )
            deltas = [left_daily[date] - right_daily[date] for date in dates]
            cell_means.append(sum(deltas) / len(deltas))
            for date, delta in zip(dates, deltas, strict=True):
                daily_by_fold_date[(fold_id, date)].append(delta)
    fold_values: dict[str, list[float]] = defaultdict(list)
    for (fold_id, _), seed_values in sorted(daily_by_fold_date.items()):
        fold_values[fold_id].append(sum(seed_values) / len(seed_values))
    return {
        "left": left,
        "right": right,
        "cell_count": len(cell_means),
        "positive_cell_ratio": sum(value > 0 for value in cell_means) / len(cell_means),
        "cell_mean_deltas": cell_means,
        "daily_seed_averaged_inference": _inference(
            config, [fold_values[fold_id] for fold_id in sorted(fold_values)]
        ),
    }


def _decision(
    *,
    delta: dict[str, Any],
    minimum_ratio: float,
    source_gate_failed: bool,
    require_source_ready: bool,
) -> dict[str, Any]:
    mean = delta["daily_seed_averaged_inference"]["hac"]["mean"]
    reasons = []
    if mean is None or mean <= 0:
        reasons.append("paired mean Rank IC delta is not positive")
    if delta["positive_cell_ratio"] < minimum_ratio:
        reasons.append("positive fold/seed cell ratio is below the configured threshold")
    if source_gate_failed and require_source_ready:
        reasons.append("source provenance is missing or not explicitly research-ready")
    return {
        "passed": not reasons,
        "status": "go" if not reasons else "no_go",
        "reasons": reasons,
        "mean_delta": mean,
        "positive_cell_ratio": delta["positive_cell_ratio"],
        "minimum_positive_cell_ratio": minimum_ratio,
    }


def aggregate_research_runs(
    cells: list[dict[str, Any]],
    config: M6ResearchConfig,
    *,
    evaluation_split: str,
    fold_ids: list[str] | None = None,
) -> dict[str, Any]:
    active_folds = fold_ids or [fold.fold_id for fold in config.folds]
    known_folds = {fold.fold_id for fold in config.folds}
    if not active_folds or not set(active_folds).issubset(known_folds):
        raise ValueError("fold_ids must be a non-empty subset of configured folds")
    loaded = [_load_cell(cell) for cell in cells]
    _validate_matrix(loaded, config, evaluation_split, active_folds)
    cost_bps = config.decisions.cost_bps
    model_results = {}
    for model in MODEL_KEYS:
        model_cells = [cell for cell in loaded if cell["model_key"] == model]
        raw = _average_seed_series(
            model_cells, lambda metrics: _daily_map(metrics, "raw", "daily_ic")
        )
        neutralized = _average_seed_series(
            model_cells, lambda metrics: _daily_map(metrics, "neutralized", "daily_ic")
        )
        net = _average_seed_series(
            model_cells, lambda metrics: _portfolio_map(metrics, cost_bps)
        )
        model_results[model] = {
            "raw_rank_ic": _inference(config, raw["fold_values"]),
            "neutralized_rank_ic": _inference(config, neutralized["fold_values"]),
            f"net_{cost_bps:g}bps": _inference(config, net["fold_values"]),
        }
    paired = {
        name: _paired_delta(loaded, config, left, right, active_folds)
        for name, (left, right) in PAIRED_COMPARISONS.items()
    }
    source_values = [
        cell["metrics"]["metrics"]["cross_section"].get("source_research_ready")
        for cell in loaded
    ]
    source_explicitly_blocked = any(value is False for value in source_values)
    source_all_ready = all(value is True for value in source_values)
    source_gate_failed = not source_all_ready
    minimum_ratio = config.decisions.minimum_positive_cell_ratio
    decisions = {
        name: _decision(
            delta=paired[name],
            minimum_ratio=minimum_ratio,
            source_gate_failed=source_gate_failed,
            require_source_ready=config.decisions.require_source_research_ready,
        )
        for name in [
            "architecture_e1_vs_e0",
            "external_transfer_e2_vs_e1",
            "financial_pretraining_e3_vs_e2",
        ]
    }
    e3 = model_results["e3"]
    overall_reasons = []
    raw_mean = e3["raw_rank_ic"]["hac"]["mean"]
    neutralized_mean = e3["neutralized_rank_ic"]["hac"]["mean"]
    net_mean = e3[f"net_{cost_bps:g}bps"]["hac"]["mean"]
    if raw_mean is None or raw_mean <= 0:
        overall_reasons.append("E3 raw Rank IC is not positive")
    if config.decisions.require_neutralized_positive and (
        neutralized_mean is None or neutralized_mean <= 0
    ):
        overall_reasons.append("E3 neutralized Rank IC is unavailable or not positive")
    if net_mean is None or net_mean <= 0:
        overall_reasons.append(f"E3 net Q5-Q1 at {cost_bps:g} bps is not positive")
    if paired["overall_e3_vs_e1"]["positive_cell_ratio"] < minimum_ratio:
        overall_reasons.append("E3 does not beat E1 in enough fold/seed cells")
    if source_gate_failed and config.decisions.require_source_research_ready:
        overall_reasons.append("source provenance is missing or not explicitly research-ready")
    decisions["overall_e3"] = {
        "passed": not overall_reasons,
        "status": "go" if not overall_reasons else "no_go",
        "reasons": overall_reasons,
        "raw_rank_ic_mean": raw_mean,
        "neutralized_rank_ic_mean": neutralized_mean,
        f"net_{cost_bps:g}bps_mean": net_mean,
        "e3_vs_e1_positive_cell_ratio": paired["overall_e3_vs_e1"][
            "positive_cell_ratio"
        ],
    }
    return {
        "schema_version": 1,
        "research_id": config.research_id,
        "evaluation_split": evaluation_split,
        "folds": active_folds,
        "seeds": config.seeds,
        "cell_count": len(loaded),
        "source_readiness": {
            "all_cells_explicitly_ready": source_all_ready,
            "explicitly_blocked": source_explicitly_blocked,
            "gate_failed": source_gate_failed,
            "values": source_values,
        },
        "inference_protocol": {
            "hac_lags": config.hac_lags,
            "non_overlapping_stride": config.non_overlapping_stride,
            "non_overlapping_offset": config.non_overlapping_offset,
            "seed_handling": "average daily metrics across seeds before time-series inference",
        },
        "models": model_results,
        "paired_deltas": paired,
        "decisions": decisions,
    }


def write_research_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "research.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = "".join(
        "<tr>"
        f"<th>{html.escape(name)}</th>"
        f"<td>{html.escape(str(decision['status']))}</td>"
        f"<td>{html.escape('; '.join(decision['reasons']) or 'criteria satisfied')}</td>"
        "</tr>"
        for name, decision in result["decisions"].items()
    )
    payload = html.escape(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    (output_dir / "research.html").write_text(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<title>FacDigger Research Freeze</title><style>body{font-family:system-ui;"
        "max-width:1100px;margin:2rem auto}table{border-collapse:collapse}"
        "th,td{border:1px solid #ccc;padding:.45rem;text-align:left}"
        "pre{background:#f5f5f5;padding:1rem;overflow:auto}</style></head><body>"
        f"<h1>{html.escape(result['research_id'])}</h1>"
        f"<p>split: {html.escape(result['evaluation_split'])}; cells: "
        f"{result['cell_count']}</p><table><thead><tr><th>question</th>"
        f"<th>decision</th><th>reasons</th></tr></thead><tbody>{rows}</tbody></table>"
        f"<h2>完整结果</h2><pre>{payload}</pre></body></html>",
        encoding="utf-8",
    )
