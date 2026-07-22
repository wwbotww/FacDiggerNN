"""Point-in-time cross-sectional score neutralization."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def _neutralize_date(rows: list[dict[str, Any]]) -> tuple[dict[str, float], str]:
    complete = [
        row
        for row in rows
        if row.get("industry_code") is not None
        and row.get("log_float_market_cap") is not None
        and np.isfinite(row["log_float_market_cap"])
    ]
    if len(complete) < 3:
        return {}, "unavailable_missing_point_in_time_exposures"
    industries = sorted({str(row["industry_code"]) for row in complete})
    columns: list[np.ndarray] = [np.ones(len(complete), dtype=np.float64)]
    market_cap = np.asarray(
        [float(row["log_float_market_cap"]) for row in complete], dtype=np.float64
    )
    if np.std(market_cap) > 0:
        columns.append((market_cap - market_cap.mean()) / market_cap.std())
    for industry in industries[1:]:
        columns.append(
            np.asarray(
                [float(str(row["industry_code"]) == industry) for row in complete],
                dtype=np.float64,
            )
        )
    design = np.column_stack(columns)
    if len(complete) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        return {}, "unavailable_insufficient_cross_section"
    scores = np.asarray([float(row["score_raw"]) for row in complete], dtype=np.float64)
    coefficients, *_ = np.linalg.lstsq(design, scores, rcond=None)
    residuals = scores - design @ coefficients
    return {
        f"{row['security_id']}|{row['asof_date']}": float(residual)
        for row, residual in zip(complete, residuals, strict=True)
    }, "industry_and_log_market_cap"


def neutralize_predictions(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Add neutralized scores without silently substituting unavailable exposures."""

    residual_by_key: dict[str, float] = {}
    quality_counts: dict[str, int] = {}
    for group in frame.partition_by("asof_date", maintain_order=True):
        residuals, quality = _neutralize_date(group.to_dicts())
        residual_by_key.update(residuals)
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    keys = [
        f"{security_id}|{asof_date}"
        for security_id, asof_date in frame.select("security_id", "asof_date").iter_rows()
    ]
    values = [residual_by_key.get(key) for key in keys]
    result = frame.with_columns(pl.Series("score_neutralized", values, dtype=pl.Float64))
    return result, {
        "date_quality_counts": quality_counts,
        "available_rows": sum(value is not None for value in values),
        "total_rows": len(values),
    }
