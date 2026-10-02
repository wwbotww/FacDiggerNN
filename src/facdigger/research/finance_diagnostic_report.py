"""Summarize C without choosing a new budget, model or favourable support subset."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

from facdigger.data.contracts import DataContractError
from facdigger.research.statistics import newey_west_mean_inference, non_overlapping_mean_inference
from facdigger.training.runtime import write_json

CANDIDATES = ("statistics_linear", "statistics_mlp", "finance")


def paired_daily(left: pl.DataFrame, right: pl.DataFrame, *, sparse: bool = False) -> dict:
    keys = ["asof_date", "horizon", "labelled_rows", "computational_rows"]
    left, right = (x.filter(pl.col("horizon") == 5).sort("asof_date") for x in (left, right))
    if not left.select(keys).equals(right.select(keys)):
        raise DataContractError("paired comparison has different dates/support; no intersection")
    if left["rank_ic"].null_count() or right["rank_ic"].null_count():
        return {
            "status": "undefined_rank_ic",
            "expected_dates": left.height,
            "reason": "constant/invalid ranks retained; no successful-date intersection",
        }
    delta = (left["rank_ic"] - right["rank_ic"]).to_numpy()
    if not np.isfinite(delta).all():
        raise DataContractError("non-finite paired diagnostic series")
    result = {"n": len(delta), "mean_delta": float(delta.mean()), "reference_delta": 0.001}
    if sparse:
        result["inference"] = "descriptive sparse F panel; no daily-lag inference"
        return result
    result["HAC"] = {}
    for lag in (5, 20, 60):
        for null in (0.0, 0.001):
            stats = newey_west_mean_inference(delta.tolist(), lag, null_mean=null)
            se = stats["standard_error"]
            stats["two_sided_95_interval"] = (
                [stats["mean"] - 1.959963984540054 * se, stats["mean"] + 1.959963984540054 * se]
                if se is not None
                else None
            )
            result["HAC"][f"lag{lag}-null{null}"] = stats
    result["stride5_all_offsets"] = [
        non_overlapping_mean_inference(delta.tolist(), stride=5, offset=i) for i in range(5)
    ]
    return result


def summarize_prefix(root: Path, output: Path) -> dict:
    audits = {
        name: json.loads((root / name / "diagnostic.json").read_text()) for name in CANDIDATES
    }
    protocol = audits[CANDIDATES[0]]["identity"]["data_protocol"]
    if any(a["identity"]["data_protocol"] != protocol for a in audits.values()):
        raise DataContractError("C candidate data protocols differ")
    config = audits[CANDIDATES[0]]["identity"]["config"]
    if any(a["identity"]["config"] != config for a in audits.values()):
        raise DataContractError("C candidate training conditions differ")
    result = {
        "status": "complete"
        if all(a["status"] == "observations_complete" for a in audits.values())
        else "incomplete",
        "limitations": [
            "two epochs, one seed/fold; not fully trained performance",
            "S selects checkpoints; V is development evidence, not holdout",
            "daily intervals do not include training seed uncertainty",
        ],
        "candidates": {},
        "paired": {},
    }
    tables = {}
    for name, audit in audits.items():
        summary = {
            "status": audit["status"],
            "training_status": audit["training_status"],
            "attempts": audit["attempts"],
            "best_epoch": audit.get("best_epoch"),
            "relative_parameter_change": audit.get("relative_parameter_change"),
            "curves": {},
            "monthly_primary_ic": {},
        }
        for file in sorted((root / name).glob("epoch-*-*-daily.parquet")):
            table = pl.read_parquet(file)
            label = file.stem.removesuffix("-daily")
            tables[(name, label)] = table
            summary["curves"][label] = (
                table.group_by("horizon")
                .agg(
                    pl.len().alias("expected_dates"),
                    pl.col("rank_ic").count().alias("finite_ic_dates"),
                    pl.col("rank_ic").mean(),
                    pl.col("raw_ic").mean(),
                    pl.col("surrogate_correlation").mean(),
                    pl.col("score_std").mean(),
                    pl.col("weighted_available_loss").mean(),
                    pl.col("weighted_available_scale_penalty").mean(),
                )
                .sort("horizon")
                .to_dicts()
            )
            summary["monthly_primary_ic"][label] = (
                table.filter(pl.col("horizon") == 5)
                .group_by(pl.col("asof_date").dt.strftime("%Y-%m").alias("month"))
                .agg(pl.col("rank_ic").mean(), pl.len().alias("dates"))
                .sort("month")
                .to_dicts()
            )
        result["candidates"][name] = summary
    # Include every predetermined pair; no winner-dependent choice of comparator.
    for right, left in combinations(CANDIDATES, 2):
        pair = result["paired"][f"{left}-minus-{right}"] = {}
        for phase in ("F", "S", "V"):
            for rule in ("epoch2", "S-best"):
                le = 2 if rule == "epoch2" else audits[left].get("best_epoch")
                re = 2 if rule == "epoch2" else audits[right].get("best_epoch")
                keys = ((left, f"epoch-{le}-{phase}"), (right, f"epoch-{re}-{phase}"))
                if all(k in tables for k in keys):
                    pair[f"{phase}-{rule}"] = paired_daily(
                        tables[keys[0]], tables[keys[1]], sparse=phase == "F"
                    )
                else:
                    pair[f"{phase}-{rule}"] = {"status": "missing_observation"}
    write_json(output, result)
    return result
