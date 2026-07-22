"""Train-only robust feature scaling."""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl

from facdigger.data.contracts import DataContractError


def fit_train_robust_scaler(
    features: pl.DataFrame,
    channels: list[str],
    train_end: date,
    winsor_lower: float = 0.005,
    winsor_upper: float = 0.995,
) -> dict[str, Any]:
    if not 0 <= winsor_lower < winsor_upper <= 1:
        raise ValueError("winsor quantiles must satisfy 0 <= lower < upper <= 1")
    training = features.filter(pl.col("trade_date") <= pl.lit(train_end))
    parameters: dict[str, dict[str, float]] = {}
    for channel in channels:
        values = training[channel].drop_nulls()
        if values.is_empty():
            raise DataContractError(f"No training observations available for feature {channel}")
        lower = float(values.quantile(winsor_lower, interpolation="linear"))
        upper = float(values.quantile(winsor_upper, interpolation="linear"))
        clipped = values.clip(lower, upper)
        median = float(clipped.median())
        q25 = float(clipped.quantile(0.25, interpolation="linear"))
        q75 = float(clipped.quantile(0.75, interpolation="linear"))
        scale = (q75 - q25) / 1.349
        if not scale > 1e-12:
            scale = 1.0
        parameters[channel] = {
            "winsor_lower_value": lower,
            "winsor_upper_value": upper,
            "median": median,
            "scale": scale,
            "train_observations": len(values),
        }
    return {
        "method": "train_global_robust",
        "fit_end": train_end.isoformat(),
        "winsor_lower_quantile": winsor_lower,
        "winsor_upper_quantile": winsor_upper,
        "channels": parameters,
    }


def apply_robust_scaler(
    features: pl.DataFrame,
    parameters: dict[str, Any],
) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for channel, values in parameters["channels"].items():
        expressions.append(
            (
                (
                    pl.col(channel).clip(values["winsor_lower_value"], values["winsor_upper_value"])
                    - values["median"]
                )
                / values["scale"]
            ).alias(channel)
        )
    return features.with_columns(expressions)
