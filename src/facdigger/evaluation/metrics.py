"""Deterministic cross-sectional factor metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def daily_information_coefficients(
    frame: pl.DataFrame, score_column: str = "score_raw"
) -> pl.DataFrame:
    records: list[dict[str, Any]] = []
    usable = frame.filter(pl.col(score_column).is_not_null())
    for group in usable.partition_by("asof_date", maintain_order=True):
        scores = group[score_column].to_numpy().astype(np.float64)
        targets = group["target"].to_numpy().astype(np.float64)
        records.append(
            {
                "asof_date": group["asof_date"][0],
                "n": len(scores),
                "ic": _correlation(scores, targets),
                "rank_ic": _correlation(_average_ranks(scores), _average_ranks(targets)),
            }
        )
    return pl.DataFrame(
        records,
        schema={"asof_date": pl.Date, "n": pl.Int64, "ic": pl.Float64, "rank_ic": pl.Float64},
    )


def _aggregate_series(values: list[float], annualization: int) -> dict[str, float | int | None]:
    if not values:
        return {"dates": 0, "mean": None, "std": None, "ir": None, "positive_ratio": None}
    array = np.asarray(values, dtype=np.float64)
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    return {
        "dates": len(array),
        "mean": float(np.mean(array)),
        "std": std,
        "ir": float(np.mean(array) / std * math.sqrt(annualization)) if std > 0 else None,
        "positive_ratio": float(np.mean(array > 0)),
    }


def aggregate_ic(daily: pl.DataFrame, annualization: int = 252) -> dict[str, Any]:
    return {
        "ic": _aggregate_series(daily["ic"].drop_nulls().to_list(), annualization),
        "rank_ic": _aggregate_series(daily["rank_ic"].drop_nulls().to_list(), annualization),
    }


def daily_quantile_portfolio(
    frame: pl.DataFrame,
    *,
    score_column: str = "score_raw",
    quantiles: int = 5,
    costs_bps: list[float] | None = None,
) -> pl.DataFrame:
    costs = costs_bps or [0.0, 10.0, 20.0, 50.0]
    previous_weights: dict[str, float] = {}
    records: list[dict[str, Any]] = []
    usable = frame.filter(pl.col(score_column).is_not_null())
    for group in usable.partition_by("asof_date", maintain_order=True):
        ordered = group.sort([score_column, "security_id"])
        count = ordered.height
        if count < 2:
            continue
        groups = min(quantiles, count)
        bottom_count = max(1, count // groups)
        top_count = max(1, count // groups)
        bottom = ordered.head(bottom_count)
        top = ordered.tail(top_count)
        weights = {
            security_id: -1.0 / bottom_count for security_id in bottom["security_id"].to_list()
        }
        weights.update(
            {security_id: 1.0 / top_count for security_id in top["security_id"].to_list()}
        )
        all_ids = set(previous_weights) | set(weights)
        turnover = (
            0.5
            * sum(abs(weights.get(key, 0.0) - previous_weights.get(key, 0.0)) for key in all_ids)
            if previous_weights
            else None
        )
        gross = float(top["target"].mean() - bottom["target"].mean())
        record: dict[str, Any] = {
            "asof_date": ordered["asof_date"][0],
            "n": count,
            "groups": groups,
            "gross_q_high_minus_low": gross,
            "turnover": turnover,
        }
        for cost in costs:
            key = f"net_{cost:g}bps"
            record[key] = gross if turnover is None else gross - turnover * cost / 10_000
        records.append(record)
        previous_weights = weights
    if not records:
        return pl.DataFrame()
    return pl.DataFrame(records)


def aggregate_quantiles(daily: pl.DataFrame) -> dict[str, Any]:
    if daily.is_empty():
        return {"dates": 0}
    value_columns = [
        column
        for column in daily.columns
        if column == "gross_q_high_minus_low" or column.startswith("net_")
    ]
    result: dict[str, Any] = {
        "dates": daily.height,
        "mean_turnover": (
            float(daily["turnover"].drop_nulls().mean())
            if daily["turnover"].drop_nulls().len()
            else None
        ),
    }
    for column in value_columns:
        values = daily[column].drop_nulls()
        result[column] = float(values.mean()) if values.len() else None
    return result


def evaluate_score(
    frame: pl.DataFrame, score_column: str, costs_bps: list[float]
) -> dict[str, Any]:
    daily_ic = daily_information_coefficients(frame, score_column)
    daily_quantiles = daily_quantile_portfolio(
        frame, score_column=score_column, costs_bps=costs_bps
    )
    return {
        **aggregate_ic(daily_ic),
        "portfolio": aggregate_quantiles(daily_quantiles),
        "daily_ic": daily_ic.to_dicts(),
        "daily_portfolio": daily_quantiles.to_dicts(),
    }


def evaluate_predictions(frame: pl.DataFrame, costs_bps: list[float]) -> dict[str, Any]:
    cross_section_sizes = frame.group_by("asof_date").len()["len"]
    cross_section = {
        "dates": cross_section_sizes.len(),
        "minimum": int(cross_section_sizes.min()),
        "median": float(cross_section_sizes.median()),
        "mean": float(cross_section_sizes.mean()),
        "maximum": int(cross_section_sizes.max()),
        "research_ready": bool(
            cross_section_sizes.len() >= 20 and float(cross_section_sizes.median()) >= 20
        ),
        "research_ready_rule": "at least 20 dates and median cross-section >= 20",
    }
    raw = evaluate_score(frame, "score_raw", costs_bps)
    neutralized_rows = frame.filter(pl.col("score_neutralized").is_not_null()).height
    neutralized = (
        evaluate_score(frame, "score_neutralized", costs_bps) if neutralized_rows else None
    )
    by_year: dict[str, Any] = {}
    for year in frame["asof_date"].dt.year().unique().sort().to_list():
        by_year[str(year)] = evaluate_score(
            frame.filter(pl.col("asof_date").dt.year() == year), "score_raw", costs_bps
        )
        by_year[str(year)].pop("daily_ic")
        by_year[str(year)].pop("daily_portfolio")
    by_industry: dict[str, Any] = {}
    industry_rows = frame.filter(pl.col("industry_code").is_not_null())
    for industry in industry_rows["industry_code"].unique().sort().to_list():
        group_metrics = evaluate_score(
            industry_rows.filter(pl.col("industry_code") == industry),
            "score_raw",
            costs_bps,
        )
        group_metrics.pop("daily_ic")
        group_metrics.pop("daily_portfolio")
        by_industry[str(industry)] = group_metrics

    size_frames: list[pl.DataFrame] = []
    complete_size = frame.filter(pl.col("log_float_market_cap").is_not_null())
    for group in complete_size.partition_by("asof_date", maintain_order=True):
        ordered = group.sort(["log_float_market_cap", "security_id"])
        count = ordered.height
        labels = [("small", "mid", "large")[min(index * 3 // count, 2)] for index in range(count)]
        size_frames.append(ordered.with_columns(pl.Series("size_bucket", labels)))
    by_size: dict[str, Any] = {}
    if size_frames:
        size_panel = pl.concat(size_frames)
        for bucket in ["small", "mid", "large"]:
            bucket_frame = size_panel.filter(pl.col("size_bucket") == bucket)
            if bucket_frame.is_empty():
                continue
            group_metrics = evaluate_score(bucket_frame, "score_raw", costs_bps)
            group_metrics.pop("daily_ic")
            group_metrics.pop("daily_portfolio")
            by_size[bucket] = group_metrics
    return {
        "raw": raw,
        "neutralized": neutralized,
        "cross_section": cross_section,
        "stability": {
            "by_year": by_year,
            "by_industry": by_industry,
            "by_size": by_size,
            "industry_rows": industry_rows.height,
            "size_rows": complete_size.height,
        },
    }
