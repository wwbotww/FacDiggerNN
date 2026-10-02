"""Fixed-state Finance diagnostics; no checkpoint selection or holdout access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from facdigger.data.contracts import DataContractError
from facdigger.datasets.index import align_labelled_samples
from facdigger.datasets.window import (
    FinanceTransformerInferenceWindowDataset,
    MarketFeatureStore,
    SecurityFeatureStore,
)
from facdigger.models.finance_scoring import predict_finance_horizons
from facdigger.training.common import (
    load_required_market_features,
    load_required_snapshot_features,
)
from facdigger.training.finance_transformer_engine import _multi_horizon_loss
from facdigger.training.ranking import average_ranks, cross_sectional_rank_targets


def fixed_dates(rows: pl.DataFrame, *, bins: int = 5, per_bin: int = 12) -> list[Any]:
    """Only dates determine the panel; never scores, labels, or coverage."""
    dates = rows["asof_date"].unique().sort().to_list()
    if len(dates) < bins * per_bin:
        raise DataContractError("insufficient dates for the preregistered panel")
    return [
        part[int(i)]
        for part in np.array_split(np.asarray(dates, dtype=object), bins)
        for i in np.linspace(0, len(part) - 1, per_bin, dtype=int)
    ]


def computation_rows(labelled: pl.DataFrame) -> pl.DataFrame:
    """Original legacy C=L pool, without supervised fields."""
    return labelled.select(
        "sample_id", "security_id", "symbol", "asof_date", "feature_start", "feature_end"
    ).with_columns(pl.lit(True).alias("eligible"))


def window_datasets(
    snapshot: Path,
    manifest: dict,
    config: Any,
    pools: dict[str, pl.DataFrame],
) -> dict[str, FinanceTransformerInferenceWindowDataset]:
    required = pl.concat(list(pools.values()), how="diagonal_relaxed").unique(
        ["asof_date", "security_id"]
    )
    features = SecurityFeatureStore(
        features=load_required_snapshot_features(snapshot, manifest, required),
        channels=config.channels,
        presorted=True,
    )
    market = MarketFeatureStore(
        features=load_required_market_features(snapshot, manifest, required),
        channels=config.market_channels,
    )
    return {
        key: FinanceTransformerInferenceWindowDataset(
            feature_store=features,
            market_store=market,
            inference_index=rows,
            channels=config.channels,
            market_channels=config.market_channels,
            context_length=int(manifest["config"]["features"]["context_length"]),
            primary_horizon=config.primary_horizon,
        )
        for key, rows in pools.items()
    }


def correlation(left: np.ndarray, right: np.ndarray, *, rank: bool = False) -> float | None:
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return None
    if rank:
        left, right = average_ranks(left), average_ranks(right)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def diagnostic_daily(
    scores: np.ndarray,
    pool: pl.DataFrame,
    labelled: pl.DataFrame,
    config: Any,
    *,
    horizons: tuple[int, ...] | None = None,
) -> pl.DataFrame:
    """Project full C onto fixed L. Missing scores fail, never shrink support."""
    horizons = horizons or tuple(config.horizons)
    aligned = align_labelled_samples(pool, labelled.sort("asof_date", "security_id"))
    if scores.shape != (pool.height, len(horizons)) or not np.isfinite(scores).all():
        raise DataContractError("diagnostic predictions are incomplete or non-finite")
    selected = scores[aligned["_computational_row"].to_numpy()]
    objective = config.training.objective
    counts = pool.group_by("asof_date").len()
    pool_counts = dict(counts.iter_rows())
    result = []
    offset = 0
    for (asof,), day in aligned.group_by("asof_date", maintain_order=True):
        n = day.height
        values = selected[offset : offset + n]
        offset += n
        ranks = np.stack(
            [
                cross_sectional_rank_targets(
                    day[f"target_{h}"].to_numpy(),
                    [asof] * n,
                    minimum_cross_section_size=objective.minimum_cross_section_size,
                )
                for h in horizons
            ],
            axis=1,
        )
        _, loss = _multi_horizon_loss(
            torch.tensor(values, dtype=torch.float32),
            torch.from_numpy(ranks),
            horizons=horizons,
            horizon_weights=objective.horizon_weights,
            epsilon=objective.epsilon,
            scale_regularization=objective.scale_regularization,
        )
        for col, h in enumerate(horizons):
            result.append(
                {
                    "asof_date": asof,
                    "horizon": h,
                    "labelled_rows": n,
                    "computational_rows": pool_counts[asof],
                    "prediction_coverage": 1.0,
                    "rank_ic": correlation(
                        values[:, col], day[f"target_{h}"].to_numpy(), rank=True
                    ),
                    "raw_ic": correlation(values[:, col], day[f"target_{h}"].to_numpy()),
                    "surrogate_correlation": 1.0 - loss[f"rank_loss_{h}"],
                    "score_std": loss[f"score_std_{h}"],
                    # Total loss is only the full objective when all three heads are present.
                    "weighted_available_loss": loss["loss"],
                    "weighted_available_scale_penalty": loss["scale_penalty"],
                    "full_objective_available": horizons == tuple(config.horizons),
                }
            )
    return pl.DataFrame(result)


def score_panel(
    model: Any, dataset: Any, labelled: pl.DataFrame, config: Any, *, check_stop=None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    scores = predict_finance_horizons(
        model,
        dataset,
        batch_size=config.training.batch_size,
        device=str(next(model.parameters()).device),
        precision=config.training.precision,
        num_workers=0,
        check_stop=check_stop,
    )
    daily = diagnostic_daily(scores, dataset.sample_rows, labelled, config)
    predictions = dataset.sample_rows.select("sample_id", "security_id", "asof_date").with_columns(
        *[pl.Series(f"score_{h}", scores[:, col]) for col, h in enumerate(config.horizons)]
    )
    return daily, predictions


def statistics_style_exposures(
    predictions: pl.DataFrame,
    cached: Any,
    labelled: pl.DataFrame,
) -> pl.DataFrame:
    """Observed input proxies, not executable return factors or new fitted features."""
    rows = cached.sample_rows
    if not predictions.select("sample_id", "security_id", "asof_date").equals(
        rows.select("sample_id", "security_id", "asof_date")
    ):
        raise DataContractError("style diagnostics and prediction computation keys differ")
    channels = len(cached.channels)
    aligned = align_labelled_samples(rows, labelled.sort("asof_date", "security_id"))
    indices = aligned["_computational_row"].to_numpy()
    # Cache ordering: each window has mean/std/min/max/ratio, then latest values.
    close = cached.channels.index("r_close")
    vol = cached.channels.index("vol20")
    positions = cached.positions[indices]
    proxies = {
        "reversal_scaled_mean_5": -cached.values[
            positions, cached.windows.index(5) * 5 * channels + close
        ],
        "momentum_scaled_mean_20": cached.values[
            positions, cached.windows.index(20) * 5 * channels + close
        ],
        "volatility_latest_vol20": cached.values[
            positions, cached.values.shape[1] - channels + vol
        ],
    }
    result = []
    cursor = 0
    for (day,), part in aligned.group_by("asof_date", maintain_order=True):
        stop = cursor + part.height
        score = predictions["score_5"].to_numpy()[indices[cursor:stop]]
        for name, values in proxies.items():
            result.append(
                {
                    "asof_date": day,
                    "style": name,
                    "labelled_rows": part.height,
                    "score_rank_exposure": correlation(score, values[cursor:stop], rank=True),
                    "style_rank_ic": correlation(
                        values[cursor:stop], part["target_5"].to_numpy(), rank=True
                    ),
                }
            )
        cursor = stop
    return pl.DataFrame(result)


def evaluate_diagnostic_validation(
    snapshot: Path,
    manifest: dict,
    predictions: pl.DataFrame,
    labelled: pl.DataFrame,
    config: Any,
    *,
    model_id: str,
    checkpoint_hash: str,
    destination: Path,
) -> None:
    """Use the existing prediction contract/evaluator on original V support only."""
    from facdigger.evaluation.contracts import prediction_coverage
    from facdigger.evaluation.metrics import evaluate_predictions
    from facdigger.training.common import (
        apply_source_readiness_gate,
        build_prediction_frame,
        load_source_provenance,
    )
    from facdigger.training.runtime import write_json

    if labelled["split"].unique().to_list() != ["valid"]:
        raise DataContractError("diagnostic evaluation requires validation, never holdout")
    support = labelled.sort("asof_date", "security_id")
    scored = support.join(
        predictions.select("security_id", "asof_date", "score_5"),
        on=["security_id", "asof_date"],
        how="left",
        validate="1:1",
    )
    if scored["score_5"].null_count() or not scored["score_5"].is_finite().all():
        raise DataContractError("missing diagnostic V predictions on fixed support")
    metadata = (
        pl.scan_parquet(snapshot / "sample_metadata.parquet")
        .filter(pl.col("sample_id").is_in(support["sample_id"].implode()))
        .collect()
    )
    frame, neutralization = build_prediction_frame(
        scored,
        metadata,
        scored["score_5"].to_numpy(),
        model_id=model_id,
        checkpoint_hash=checkpoint_hash,
        dataset_id=manifest["dataset_id"],
    )
    coverage = prediction_coverage(frame, support, split="valid", minimum=config.minimum_coverage)
    provenance = load_source_provenance(snapshot, manifest)
    metrics = apply_source_readiness_gate(evaluate_predictions(frame, config.costs_bps), provenance)
    frame.write_parquet(destination.with_suffix(".parquet"))
    write_json(
        destination.with_suffix(".json"),
        {
            "evaluation_split": "valid",
            "purpose": "diagnostic; not a release or independent test",
            "coverage": coverage,
            "neutralization": neutralization,
            "source_provenance": provenance,
            "metrics": metrics,
        },
    )
