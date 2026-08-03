"""Aggregate a frozen E0-E3 fold/seed matrix into auditable research conclusions."""

from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

from facdigger.data.contracts import DataContractError
from facdigger.research.config import M6ResearchConfig
from facdigger.research.statistics import holm_step_down, panel_mean_inference

MODEL_KEYS = ("e0", "e1", "e2", "e3")
ATTRIBUTION_COMPARISONS = {
    "architecture_e1_vs_e0": ("e1", "e0"),
    "external_transfer_e2_vs_e1": ("e2", "e1"),
    "financial_pretraining_e3_vs_e2": ("e3", "e2"),
}
PAIRED_COMPARISONS = {**ATTRIBUTION_COMPARISONS, "overall_e3_vs_e1": ("e3", "e1")}


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
    predictions = (
        pl.read_parquet(predictions_path)
        .select("security_id", "asof_date", "target")
        .sort(["asof_date", "security_id"])
    )
    if predictions.is_empty():
        raise DataContractError(f"research cell predictions are empty: {run_dir}")
    date_counts = {
        str(row["asof_date"]): int(row["len"])
        for row in predictions.group_by("asof_date").len().sort("asof_date").to_dicts()
    }
    return {
        **cell,
        "run_dir": str(run_dir),
        "metrics": metrics,
        "keys": predictions,
        "prediction_date_counts": date_counts,
    }


def _daily_map(
    metrics: dict[str, Any], section: str, field: str, *, required: bool = False
) -> dict[str, float]:
    score = metrics["metrics"].get(section)
    if score is None or field not in score:
        if required:
            raise DataContractError(f"required metric series is missing: {section}.{field}")
        return {}
    result: dict[str, float] = {}
    for row in score[field]:
        asof_date = str(row["asof_date"])
        if asof_date in result:
            raise DataContractError(f"duplicate metric date: {section}.{field}/{asof_date}")
        value_key = "rank_ic" if field == "daily_ic" else field
        value = row.get(value_key)
        if value is None or not math.isfinite(float(value)):
            raise DataContractError(f"non-finite metric value: {section}.{field}/{asof_date}")
        result[asof_date] = float(value)
    if required and not result:
        raise DataContractError(f"required metric series is empty: {section}.{field}")
    return result


def _portfolio_map(metrics: dict[str, Any], cost_bps: float) -> dict[str, float]:
    key = f"net_{cost_bps:g}bps"
    score = metrics["metrics"].get("raw")
    if score is None or "daily_portfolio" not in score:
        return {}
    result: dict[str, float] = {}
    for row in score["daily_portfolio"]:
        asof_date = str(row["asof_date"])
        value = row.get(key)
        if asof_date in result:
            raise DataContractError(f"duplicate portfolio metric date: {asof_date}")
        if value is None or not math.isfinite(float(value)):
            raise DataContractError(f"non-finite portfolio metric value: {asof_date}")
        result[asof_date] = float(value)
    return result


def _validate_matrix(
    loaded: list[dict[str, Any]],
    config: M6ResearchConfig,
    evaluation_split: str,
    active_folds: list[str],
) -> dict[str, Any]:
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

    fold_dates: dict[str, set[str]] = {}
    for fold_id in active_folds:
        fold_cells = [cell for cell in loaded if cell["fold_id"] == fold_id]
        dataset_ids = {cell["dataset_id"] for cell in fold_cells}
        if len(dataset_ids) != 1:
            raise DataContractError(f"fold {fold_id} mixes dataset snapshots")
        reference = fold_cells[0]["keys"]
        reference_counts = fold_cells[0]["prediction_date_counts"]
        for cell in fold_cells:
            if not reference.equals(cell["keys"], null_equal=True):
                raise DataContractError(f"fold {fold_id} prediction keys differ: {cell['run_dir']}")
            daily = _daily_map(cell["metrics"], "raw", "daily_ic", required=True)
            if set(daily) != set(reference_counts):
                raise DataContractError(
                    f"raw daily Rank IC dates do not match predictions: {cell['run_dir']}"
                )
            rows = cell["metrics"]["metrics"]["raw"]["daily_ic"]
            reported_counts = {str(row["asof_date"]): int(row["n"]) for row in rows}
            if reported_counts != reference_counts:
                raise DataContractError(
                    f"raw daily Rank IC counts do not match predictions: {cell['run_dir']}"
                )
        fold_dates[fold_id] = set(reference_counts)
        if len(reference_counts) < config.decisions.minimum_daily_observations_per_fold:
            raise DataContractError(
                f"fold {fold_id} has only {len(reference_counts)} daily observations; "
                f"requires {config.decisions.minimum_daily_observations_per_fold}"
            )
    for index, left in enumerate(active_folds):
        for right in active_folds[index + 1 :]:
            overlap = fold_dates[left] & fold_dates[right]
            if overlap:
                raise DataContractError(
                    f"fold date ranges overlap: {left}/{right}, first={min(overlap)}"
                )
    return {
        "complete": True,
        "daily_date_coverage": 1.0,
        "fold_daily_observations": {fold_id: len(fold_dates[fold_id]) for fold_id in active_folds},
        "expected_seed_count": len(config.seeds),
    }


def _model_series(
    loaded: list[dict[str, Any]],
    config: M6ResearchConfig,
    active_folds: list[str],
    model: str,
    extractor: Callable[[dict[str, Any]], dict[str, float]],
) -> dict[str, Any]:
    indexed = {
        (cell["fold_id"], cell["seed"]): cell for cell in loaded if cell["model_key"] == model
    }
    fold_values: list[list[float]] = []
    fold_dates: dict[str, list[str]] = {}
    unavailable = False
    for fold_id in active_folds:
        seed_maps = [extractor(indexed[(fold_id, seed)]["metrics"]) for seed in config.seeds]
        if all(not values for values in seed_maps):
            unavailable = True
            continue
        reference_dates = set(seed_maps[0])
        if not reference_dates or any(set(values) != reference_dates for values in seed_maps[1:]):
            raise DataContractError(
                f"model metric dates are incomplete across seeds: {fold_id}/{model}"
            )
        dates = sorted(reference_dates)
        fold_dates[fold_id] = dates
        fold_values.append(
            [sum(values[day] for values in seed_maps) / len(seed_maps) for day in dates]
        )
    if unavailable:
        if fold_values:
            raise DataContractError(f"model metric is only partially available: {model}")
        return {"fold_values": [], "fold_dates": {}, "available": False}
    return {"fold_values": fold_values, "fold_dates": fold_dates, "available": True}


def _inference(
    config: M6ResearchConfig, groups: list[list[float]], *, null_mean: float = 0.0
) -> dict[str, Any]:
    return panel_mean_inference(
        groups,
        hac_lags=config.hac_lags,
        stride=config.non_overlapping_stride,
        offset=config.non_overlapping_offset,
        null_mean=null_mean,
        alpha=config.decisions.significance_alpha,
    )


def _paired_delta(
    loaded: list[dict[str, Any]],
    config: M6ResearchConfig,
    left: str,
    right: str,
    active_folds: list[str],
) -> dict[str, Any]:
    indexed = {(cell["fold_id"], cell["seed"], cell["model_key"]): cell for cell in loaded}
    fold_values: list[list[float]] = []
    cell_means: dict[str, float] = {}
    fold_positive: dict[str, bool] = {}
    minimum_effect = config.decisions.minimum_mean_rank_ic_delta
    for fold_id in active_folds:
        daily_by_date: dict[str, list[float]] = defaultdict(list)
        fold_cell_means: list[float] = []
        reference_dates: set[str] | None = None
        for seed in config.seeds:
            left_daily = _daily_map(
                indexed[(fold_id, seed, left)]["metrics"], "raw", "daily_ic", required=True
            )
            right_daily = _daily_map(
                indexed[(fold_id, seed, right)]["metrics"], "raw", "daily_ic", required=True
            )
            if set(left_daily) != set(right_daily):
                raise DataContractError(f"paired dates differ: {fold_id}/{seed}/{left}-{right}")
            if reference_dates is None:
                reference_dates = set(left_daily)
            elif set(left_daily) != reference_dates:
                raise DataContractError(
                    f"paired dates differ across seeds: {fold_id}/{left}-{right}"
                )
            dates = sorted(left_daily)
            deltas = [left_daily[day] - right_daily[day] for day in dates]
            cell_key = f"{fold_id}:{seed}"
            cell_means[cell_key] = sum(deltas) / len(deltas)
            fold_cell_means.append(cell_means[cell_key])
            for day, delta in zip(dates, deltas, strict=True):
                daily_by_date[day].append(delta)
        if any(len(values) != len(config.seeds) for values in daily_by_date.values()):
            raise DataContractError(f"paired seed coverage is incomplete: {fold_id}/{left}-{right}")
        fold_values.append(
            [sum(daily_by_date[day]) / len(config.seeds) for day in sorted(daily_by_date)]
        )
        fold_positive[fold_id] = sum(fold_cell_means) / len(fold_cell_means) > minimum_effect
    inference = _inference(config, fold_values, null_mean=minimum_effect)
    return {
        "left": left,
        "right": right,
        "cell_count": len(cell_means),
        "positive_cell_ratio": sum(value > minimum_effect for value in cell_means.values())
        / len(cell_means),
        "positive_fold_ratio": sum(fold_positive.values()) / len(fold_positive),
        "cell_mean_deltas": cell_means,
        "fold_positive": fold_positive,
        "fold_daily_observations": {
            fold: len(values) for fold, values in zip(active_folds, fold_values, strict=True)
        },
        "daily_seed_averaged_inference": inference,
    }


def _non_overlapping_passes(
    config: M6ResearchConfig,
    inference: dict[str, Any],
    *,
    minimum_effect: float,
) -> bool:
    non_overlapping = inference["non_overlapping"]
    counts_ok = all(
        count >= config.decisions.minimum_non_overlapping_observations_per_fold
        for count in non_overlapping.get("fold_counts", [])
    ) and bool(non_overlapping.get("fold_counts"))
    direction_ok = (
        non_overlapping["mean"] is not None
        and non_overlapping["mean"] > minimum_effect
    )
    return counts_ok and (
        direction_ok if config.decisions.require_non_overlapping_positive else True
    )


def _paired_decision(
    config: M6ResearchConfig,
    delta: dict[str, Any],
    *,
    significance: dict[str, Any],
    source_gate_failed: bool,
) -> dict[str, Any]:
    inference = delta["daily_seed_averaged_inference"]
    hac = inference["hac"]
    reasons: list[str] = []
    if hac["mean"] is None or hac["mean"] <= config.decisions.minimum_mean_rank_ic_delta:
        reasons.append("paired mean Rank IC delta does not exceed the configured effect floor")
    if not hac["estimable"]:
        reasons.append("paired HAC inference is not estimable")
    elif not significance["rejected"]:
        reasons.append("paired one-sided HAC test does not pass Holm correction")
    if delta["positive_cell_ratio"] < config.decisions.minimum_positive_cell_ratio:
        reasons.append("positive fold/seed cell ratio is below the configured threshold")
    non_overlapping_passed = _non_overlapping_passes(
        config,
        inference,
        minimum_effect=config.decisions.minimum_mean_rank_ic_delta,
    )
    if not non_overlapping_passed:
        reasons.append("non-overlapping robustness check is incomplete or non-positive")
    if source_gate_failed and config.decisions.require_source_research_ready:
        reasons.append("source provenance is missing or not explicitly research-ready")
    return {
        "passed": not reasons,
        "status": "go" if not reasons else "no_go",
        "reasons": reasons,
        "mean_delta": hac["mean"],
        "minimum_mean_rank_ic_delta": config.decisions.minimum_mean_rank_ic_delta,
        "positive_cell_ratio": delta["positive_cell_ratio"],
        "minimum_positive_cell_ratio": config.decisions.minimum_positive_cell_ratio,
        "significance": significance,
        "non_overlapping_passed": non_overlapping_passed,
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
    completeness = _validate_matrix(loaded, config, evaluation_split, active_folds)
    cost_bps = config.decisions.cost_bps
    model_results: dict[str, Any] = {}
    for model in MODEL_KEYS:
        model_cells = [cell for cell in loaded if cell["model_key"] == model]
        raw = _model_series(
            model_cells,
            config,
            active_folds,
            model,
            lambda metrics: _daily_map(metrics, "raw", "daily_ic", required=True),
        )
        neutralized = _model_series(
            model_cells,
            config,
            active_folds,
            model,
            lambda metrics: _daily_map(metrics, "neutralized", "daily_ic"),
        )
        net = _model_series(
            model_cells,
            config,
            active_folds,
            model,
            lambda metrics: _portfolio_map(metrics, cost_bps),
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
        cell["metrics"]["metrics"]["cross_section"].get("source_research_ready") for cell in loaded
    ]
    source_explicitly_blocked = any(value is False for value in source_values)
    source_all_ready = all(value is True for value in source_values)
    source_gate_failed = not source_all_ready

    raw_p_values = {
        name: paired[name]["daily_seed_averaged_inference"]["hac"]["p_value_one_sided"]
        for name in ATTRIBUTION_COMPARISONS
    }
    holm = holm_step_down(raw_p_values, alpha=config.decisions.significance_alpha)
    decisions = {
        name: _paired_decision(
            config,
            paired[name],
            significance=holm[name],
            source_gate_failed=source_gate_failed,
        )
        for name in ATTRIBUTION_COMPARISONS
    }

    e3 = model_results["e3"]
    raw = e3["raw_rank_ic"]
    neutralized = e3["neutralized_rank_ic"]
    net = e3[f"net_{cost_bps:g}bps"]
    overall_delta = paired["overall_e3_vs_e1"]
    overall_inference = overall_delta["daily_seed_averaged_inference"]
    alpha = config.decisions.significance_alpha
    overall_reasons: list[str] = []
    raw_hac = raw["hac"]
    if raw_hac["mean"] is None or raw_hac["mean"] <= 0:
        overall_reasons.append("E3 raw Rank IC is not positive")
    if (
        not raw_hac["estimable"]
        or raw_hac["p_value_one_sided"] is None
        or raw_hac["p_value_one_sided"] > alpha
    ):
        overall_reasons.append("E3 raw Rank IC is not significant in the one-sided HAC test")
    if not _non_overlapping_passes(config, raw, minimum_effect=0.0):
        overall_reasons.append("E3 raw non-overlapping robustness check failed")
    delta_hac = overall_inference["hac"]
    if (
        delta_hac["mean"] is None
        or delta_hac["mean"] <= config.decisions.minimum_mean_rank_ic_delta
    ):
        overall_reasons.append("E3 paired Rank IC improvement over E1 is not positive")
    if (
        not delta_hac["estimable"]
        or delta_hac["p_value_one_sided"] is None
        or delta_hac["p_value_one_sided"] > alpha
    ):
        overall_reasons.append(
            "E3 improvement over E1 is not significant in the one-sided HAC test"
        )
    if overall_delta["positive_cell_ratio"] < config.decisions.minimum_positive_cell_ratio:
        overall_reasons.append("E3 does not beat E1 in enough fold/seed cells")
    if not _non_overlapping_passes(
        config,
        overall_inference,
        minimum_effect=config.decisions.minimum_mean_rank_ic_delta,
    ):
        overall_reasons.append("E3 versus E1 non-overlapping robustness check failed")
    neutralized_mean = neutralized["hac"]["mean"]
    if config.decisions.require_neutralized_positive and (
        neutralized_mean is None or neutralized_mean <= 0
    ):
        overall_reasons.append("E3 neutralized Rank IC is unavailable or not positive")
    if source_gate_failed and config.decisions.require_source_research_ready:
        overall_reasons.append("source provenance is missing or not explicitly research-ready")
    decisions["overall_e3"] = {
        "passed": not overall_reasons,
        "status": "go" if not overall_reasons else "no_go",
        "reasons": overall_reasons,
        "raw_rank_ic_mean": raw_hac["mean"],
        "raw_rank_ic_p_value_one_sided": raw_hac["p_value_one_sided"],
        "neutralized_rank_ic_mean": neutralized_mean,
        f"net_{cost_bps:g}bps_mean": net["hac"]["mean"],
        "portfolio_is_hard_gate": False,
        "e3_vs_e1_mean_delta": delta_hac["mean"],
        "e3_vs_e1_p_value_one_sided": delta_hac["p_value_one_sided"],
        "e3_vs_e1_positive_cell_ratio": overall_delta["positive_cell_ratio"],
    }
    eligible = decisions["overall_e3"]["passed"]
    holdout_eligibility = {
        "eligible": eligible,
        "status": "eligible" if eligible else "blocked",
        "required_decision": "overall_e3",
        "reasons": [] if eligible else list(decisions["overall_e3"]["reasons"]),
    }
    return {
        "schema_version": 2,
        "research_id": config.research_id,
        "evaluation_split": evaluation_split,
        "decision_context": "validation_selection"
        if evaluation_split == "valid"
        else "holdout_confirmation",
        "folds": active_folds,
        "seeds": config.seeds,
        "cell_count": len(loaded),
        "completeness": completeness,
        "source_readiness": {
            "all_cells_explicitly_ready": source_all_ready,
            "explicitly_blocked": source_explicitly_blocked,
            "gate_failed": source_gate_failed,
            "values": source_values,
        },
        "inference_protocol": {
            "hac_lags": config.hac_lags,
            "alternative": "greater",
            "significance_alpha": alpha,
            "multiple_comparison_method": config.decisions.multiple_comparison_method,
            "holm_family": list(ATTRIBUTION_COMPARISONS),
            "non_overlapping_stride": config.non_overlapping_stride,
            "non_overlapping_offset": config.non_overlapping_offset,
            "minimum_daily_observations_per_fold": (
                config.decisions.minimum_daily_observations_per_fold
            ),
            "minimum_non_overlapping_observations_per_fold": (
                config.decisions.minimum_non_overlapping_observations_per_fold
            ),
            "seed_handling": (
                "average exactly the configured seeds per fold/date before time-series inference"
            ),
            "fold_handling": "pool means but never create autocovariance across fold boundaries",
        },
        "models": model_results,
        "paired_deltas": paired,
        "multiple_comparisons": {"holm": holm},
        "decisions": decisions,
        "holdout_eligibility": holdout_eligibility,
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
