"""One feature construction and frozen-transform path for training and inference."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date
from typing import Any

import polars as pl

from facdigger.data.config import DEFAULT_CHANNELS, MARKET_CONTEXT_CHANNELS, RANK_CHANNELS
from facdigger.data.contracts import DataContractError
from facdigger.features.cross_sectional import (
    append_cross_sectional_ranks,
    build_market_context_features,
)
from facdigger.features.price_volume import build_price_volume_features
from facdigger.features.scaling import apply_robust_scaler, fit_train_robust_scaler


def build_raw_feature_tables(
    bars: pl.DataFrame, universe: pl.DataFrame, *, feature_set: str
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    price_volume = build_price_volume_features(bars, universe)
    if feature_set == "price_volume_v1":
        return price_volume, None
    if feature_set == "finance_transformer":
        return (
            append_cross_sectional_ranks(price_volume, universe),
            build_market_context_features(price_volume, universe),
        )
    raise DataContractError(f"unsupported inference feature set: {feature_set}")


def fit_feature_scaler(
    local: pl.DataFrame,
    market: pl.DataFrame | None,
    *,
    channels: list[str],
    train_end: date,
    winsor_lower: float,
    winsor_upper: float,
) -> dict[str, Any]:
    def fit(frame: pl.DataFrame, selected: list[str]) -> dict[str, Any]:
        return fit_train_robust_scaler(
            frame, selected, train_end, winsor_lower=winsor_lower, winsor_upper=winsor_upper
        )

    if market is None:
        return fit(local, channels)
    return {
        "method": "finance_transformer_train_global_robust",
        "local": fit(local, DEFAULT_CHANNELS),
        "market": fit(market, MARKET_CONTEXT_CHANNELS),
        "rank_channels": RANK_CHANNELS,
    }


def validate_feature_scaler(
    scaler: Mapping[str, Any],
    *,
    feature_set: str,
    channels: list[str],
    market_channels: list[str],
) -> None:
    if feature_set == "price_volume_v1":
        if market_channels:
            raise DataContractError("price/volume features cannot declare market channels")
        states = [(scaler, channels)]
    elif feature_set == "finance_transformer":
        if (
            channels != [*DEFAULT_CHANNELS, *RANK_CHANNELS]
            or market_channels != MARKET_CONTEXT_CHANNELS
            or scaler.get("method") != "finance_transformer_train_global_robust"
            or scaler.get("rank_channels") != RANK_CHANNELS
        ):
            raise DataContractError("finance Transformer scaler or channel contract differs")
        states = [(scaler.get("local"), DEFAULT_CHANNELS), (scaler.get("market"), market_channels)]
    else:
        raise DataContractError(f"unsupported inference feature set: {feature_set}")
    for state, expected_channels in states:
        if not isinstance(state, Mapping) or not isinstance(state.get("channels"), Mapping):
            raise DataContractError("frozen scaler must declare its channel parameters")
        if set(state["channels"]) != set(expected_channels):
            raise DataContractError("release scaler channels differ from its feature contract")
        for parameters in state["channels"].values():
            if not isinstance(parameters, Mapping):
                raise DataContractError("frozen scaler parameters must be a mapping")
            keys = ["winsor_lower_value", "winsor_upper_value", "median", "scale"]
            try:
                values = [float(parameters[k]) for k in keys]
            except (KeyError, TypeError, ValueError) as exc:
                raise DataContractError("frozen scaler has incomplete numeric parameters") from exc
            if (
                not all(math.isfinite(v) for v in values)
                or values[-1] <= 0
                or values[0] > values[1]
            ):
                raise DataContractError("frozen scaler has invalid numeric parameters")


def apply_feature_scaler(
    local: pl.DataFrame,
    market: pl.DataFrame | None,
    scaler: dict[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    if market is None:
        return apply_robust_scaler(local, scaler), None
    return (
        apply_robust_scaler(local, scaler["local"]),
        apply_robust_scaler(market, scaler["market"]),
    )
