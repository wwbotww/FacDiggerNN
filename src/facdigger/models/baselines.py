"""Multiscale tabular features and E0 model implementations."""

from __future__ import annotations

import copy
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from facdigger.datasets.sampler import FullDateBatchSampler
from facdigger.training.e0_config import LightGBMBaselineConfig, MLPBaselineConfig
from facdigger.training.ranking import (
    RANKING_OBJECTIVE,
    TARGET_TRANSFORM,
    CrossSectionalRankingConfig,
    cross_sectional_rank_correlation_loss,
    cross_sectional_rank_targets,
    grouped_rank_ic_audit,
)


def build_multiscale_features(
    features: pl.DataFrame,
    sample_index: pl.DataFrame,
    *,
    channels: list[str],
    windows: list[int],
    context_length: int,
) -> tuple[pl.DataFrame, list[str]]:
    """Build tabular features one security at a time and retain only sample rows."""

    usable_windows = [window for window in windows if window <= context_length]
    if not usable_windows:
        raise ValueError("at least one statistics window must fit within context_length")
    expressions: list[pl.Expr] = []
    feature_columns: list[str] = []
    for channel in channels:
        last_name = f"{channel}__last"
        expressions.append(pl.col(channel).cast(pl.Float32).alias(last_name))
        feature_columns.append(last_name)
        for window in usable_windows:
            for statistic in ("mean", "std", "min", "max"):
                name = f"{channel}__{statistic}_{window}"
                source = pl.col(channel)
                if statistic == "mean":
                    expression = source.rolling_mean(window, min_samples=window)
                elif statistic == "std":
                    expression = source.rolling_std(window, min_samples=window, ddof=0)
                elif statistic == "min":
                    expression = source.rolling_min(window, min_samples=window)
                else:
                    expression = source.rolling_max(window, min_samples=window)
                expressions.append(expression.cast(pl.Float32).alias(name))
                feature_columns.append(name)

    sample_columns = [
        "sample_id",
        "security_id",
        "symbol",
        "asof_date",
        "split",
        "target",
    ]
    samples = sample_index.select(sample_columns)
    sample_groups = {
        str(group["security_id"][0]): group
        for _, group in samples.group_by("security_id", maintain_order=True)
    }
    relevant = features.filter(pl.col("security_id").is_in(list(sample_groups))).select(
        "security_id", "trade_date", *channels
    )
    parts: list[pl.DataFrame] = []
    for _, block in relevant.group_by("security_id", maintain_order=True):
        security_id = str(block["security_id"][0])
        security_samples = sample_groups[security_id]
        statistics = (
            block.sort("trade_date")
            .with_columns(expressions)
            .select(
                "security_id",
                pl.col("trade_date").alias("asof_date"),
                *feature_columns,
            )
        )
        selected = security_samples.join(
            statistics,
            on=["security_id", "asof_date"],
            how="inner",
            validate="1:1",
        )
        if selected.height:
            parts.append(selected)
    if not parts:
        return samples.head(0), feature_columns
    return pl.concat(parts, how="vertical", rechunk=False), feature_columns


def build_multiscale_inference_features(
    features: pl.DataFrame,
    inference_index: pl.DataFrame,
    *,
    channels: list[str],
    windows: list[int],
    context_length: int,
) -> tuple[pl.DataFrame, list[str]]:
    """Rebuild E0 statistics for a target-free inference index."""

    placeholder = inference_index.select(
        "sample_id", "security_id", "symbol", "asof_date"
    ).with_columns(
        pl.lit("inference").alias("split"),
        pl.lit(0.0, dtype=pl.Float64).alias("target"),
    )
    tabular, columns = build_multiscale_features(
        features,
        placeholder,
        channels=channels,
        windows=windows,
        context_length=context_length,
    )
    return tabular.drop("split", "target"), columns


@dataclass(frozen=True)
class TabularPreprocessor:
    feature_columns: list[str]
    means: list[float]
    scales: list[float]

    @classmethod
    def fit(cls, frame: pl.DataFrame, feature_columns: list[str]) -> TabularPreprocessor:
        expressions: list[pl.Expr] = []
        for column in feature_columns:
            finite = pl.col(column).filter(pl.col(column).is_finite())
            expressions.extend(
                [
                    finite.mean().alias(f"{column}__mean"),
                    finite.std(ddof=0).alias(f"{column}__std"),
                ]
            )
        statistics = frame.select(expressions).row(0, named=True)
        means = []
        scales = []
        for column in feature_columns:
            mean = statistics[f"{column}__mean"]
            scale = statistics[f"{column}__std"]
            means.append(float(mean) if mean is not None else 0.0)
            numeric_scale = float(scale) if scale is not None else 0.0
            scales.append(numeric_scale if numeric_scale > 1e-12 else 1.0)
        return cls(feature_columns, means, scales)

    def _fill(self, frame: pl.DataFrame, destination: np.ndarray) -> None:
        expected = (frame.height, len(self.feature_columns) * 2)
        if destination.shape != expected or destination.dtype != np.float32:
            raise ValueError(
                f"preprocessing destination must be float32 with shape {expected}"
            )
        width = len(self.feature_columns)
        for index, (name, mean, scale) in enumerate(
            zip(self.feature_columns, self.means, self.scales, strict=True)
        ):
            values = frame[name].to_numpy()
            observed = np.isfinite(values)
            normalized = destination[:, index]
            np.subtract(values, mean, out=normalized, casting="unsafe")
            np.divide(normalized, scale, out=normalized)
            normalized[~observed] = 0.0
            destination[:, width + index] = observed

    def transform(self, frame: pl.DataFrame) -> np.ndarray:
        result = np.empty(
            (frame.height, len(self.feature_columns) * 2),
            dtype=np.float32,
        )
        self._fill(frame, result)
        return result

    def transform_to_npy(self, frame: pl.DataFrame, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        result = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=(frame.height, len(self.feature_columns) * 2),
        )
        self._fill(frame, result)
        result.flush()
        del result
        return path

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_columns": self.feature_columns,
            "means": self.means,
            "scales": self.scales,
            "missing_policy": "train_standardize_then_zero_fill_plus_observed_mask",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TabularPreprocessor:
        feature_columns = [str(value) for value in payload["feature_columns"]]
        means = [float(value) for value in payload["means"]]
        scales = [float(value) for value in payload["scales"]]
        if not feature_columns or not (
            len(feature_columns) == len(means) == len(scales)
        ):
            raise ValueError("invalid tabular preprocessing checkpoint")
        if any(scale <= 0 for scale in scales):
            raise ValueError("tabular preprocessing scales must be positive")
        return cls(feature_columns=feature_columns, means=means, scales=scales)


def build_mlp_model(
    *, input_dim: int, hidden_dims: list[int], dropout: float, device: str
) -> Any:
    from torch import nn

    layers: list[nn.Module] = []
    width = input_dim
    for hidden in hidden_dims:
        layers.extend([nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout)])
        width = hidden
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers).to(device)


def _torch_device(preference: str) -> str:
    import torch

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for E0 MLP but is unavailable")
        return "cuda"
    if preference == "auto" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    train_dates: list[Any],
    valid_dates: list[Any],
    *,
    config: MLPBaselineConfig,
    objective: CrossSectionalRankingConfig,
    seed: int,
    checkpoint_path: Path,
    preprocessing: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _torch_device(config.device)
    model = build_mlp_model(
        input_dim=train_x.shape[1],
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_target_ranks = cross_sectional_rank_targets(
        train_y,
        train_dates,
        minimum_cross_section_size=objective.minimum_cross_section_size,
    )
    sampler = FullDateBatchSampler(
        train_dates,
        shuffle=True,
        seed=seed,
        minimum_group_size=objective.minimum_cross_section_size,
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_target_ranks)),
        batch_sampler=sampler,
        num_workers=0,
    )
    valid_features = torch.from_numpy(valid_x)
    best_rank_ic = float("-inf")
    best_selection_audit: dict[str, Any] = {}
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float | int]] = []
    stale_epochs = 0
    for epoch in range(1, config.max_epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_batches = 0
        total_score_std = 0.0
        for batch_x, batch_target_rank in loader:
            batch_x = batch_x.to(device)
            batch_target_rank = batch_target_rank.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x).squeeze(-1)
            loss = cross_sectional_rank_correlation_loss(
                prediction, batch_target_rank, epsilon=objective.epsilon
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_batches += 1
            total_score_std += float(prediction.detach().float().std(unbiased=False).cpu())
        model.eval()
        with torch.no_grad():
            valid_prediction = torch.cat(
                [
                    model(batch.to(device)).squeeze(-1).float().cpu()
                    for batch in valid_features.split(config.batch_size)
                ]
            )
        selection = grouped_rank_ic_audit(
            valid_prediction.detach().float().cpu().numpy(),
            valid_y,
            valid_dates,
            minimum_cross_section_size=objective.minimum_cross_section_size,
            minimum_dates=objective.minimum_selection_dates,
            minimum_coverage=objective.minimum_selection_coverage,
        )
        selection_rank_ic = float(selection["mean_rank_ic"])
        train_loss = total_loss / max(total_batches, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_score_std": total_score_std / max(total_batches, 1),
                "complete_date_steps": total_batches,
                "selection_mean_rank_ic": selection_rank_ic,
                "selection_valid_dates": selection["valid_dates"],
                "selection_skipped_dates": selection["skipped_dates"],
                "selection_score_std": selection["score_std"],
            }
        )
        if selection_rank_ic > best_rank_ic + 1e-12:
            best_rank_ic = selection_rank_ic
            best_selection_audit = selection
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("E0 MLP did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 2,
            "model_type": "mlp",
            "objective": RANKING_OBJECTIVE,
            "target_transform": TARGET_TRANSFORM,
            "input_dim": train_x.shape[1],
            "hidden_dims": config.hidden_dims,
            "dropout": config.dropout,
            "state_dict": best_state,
            "preprocessing": preprocessing,
            "best_epoch": best_epoch,
            "best_selection_rank_ic": best_rank_ic,
            "best_selection_audit": best_selection_audit,
            "seed": seed,
        },
        checkpoint_path,
    )
    return model, {
        "device": device,
        "objective": RANKING_OBJECTIVE,
        "target_transform": TARGET_TRANSFORM,
        "optimization_unit": "one_complete_asof_date_cross_section",
        "prediction_batch_size": config.batch_size,
        "training_dates_per_epoch": len(loader),
        "best_epoch": best_epoch,
        "best_selection_rank_ic": best_rank_ic,
        "best_selection_audit": best_selection_audit,
        "epochs_ran": len(history),
        "history": history,
    }


def predict_mlp(model: Any, values: np.ndarray, device: str) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        result = model(torch.from_numpy(values).to(device)).squeeze(-1).cpu().numpy()
    return result.astype(np.float64)


def load_mlp_checkpoint(
    checkpoint_path: Path, *, device: str
) -> tuple[Any, TabularPreprocessor, dict[str, Any]]:
    import torch

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("model_type") != "mlp":
        raise ValueError("checkpoint is not an E0 MLP model")
    model = build_mlp_model(
        input_dim=int(payload["input_dim"]),
        hidden_dims=[int(value) for value in payload["hidden_dims"]],
        dropout=float(payload["dropout"]),
        device=device,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    preprocessor = TabularPreprocessor.from_dict(payload["preprocessing"])
    if int(payload["input_dim"]) != len(preprocessor.feature_columns) * 2:
        raise ValueError("MLP checkpoint input_dim disagrees with preprocessing")
    return model, preprocessor, payload


def predict_lightgbm_checkpoint(
    checkpoint_path: Path, values: np.ndarray, *, timeout_seconds: int = 1800
) -> np.ndarray:
    """Predict in a clean process to isolate native OpenMP runtimes."""

    import tempfile

    with tempfile.TemporaryDirectory(prefix="facdigger-lgb-predict-") as temporary:
        root = Path(temporary)
        input_path = root / "features.npy"
        output_path = root / "scores.npy"
        np.save(input_path, values)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "facdigger.models.lightgbm_predict_worker",
                "--input",
                str(input_path),
                "--checkpoint",
                str(checkpoint_path),
                "--output",
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            final_line = detail[-1] if detail else "unknown worker failure"
            raise RuntimeError(f"isolated LightGBM prediction failed: {final_line}")
        return np.load(output_path).astype(np.float64)


def train_lightgbm_from_files(
    *,
    train_x: Path,
    train_y: Path,
    valid_x: Path,
    valid_y: Path,
    train_target_rank: Path,
    valid_target_rank: Path,
    train_group: Path,
    valid_group: Path,
    evaluation_x: Path,
    config: LightGBMBaselineConfig,
    seed: int,
    checkpoint_path: Path,
    preprocessing: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train in a clean process without duplicating input matrices in the parent."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path = checkpoint_path.with_suffix(".scores.npy")
    audit_path = checkpoint_path.with_suffix(".audit.json")
    config_path = checkpoint_path.with_suffix(".config.json")
    config_path.write_text(
        json.dumps({"model": config.model_dump(), "seed": seed}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "facdigger.models.lightgbm_worker",
                "--train-x",
                str(train_x),
                "--train-y",
                str(train_y),
                "--valid-x",
                str(valid_x),
                "--valid-y",
                str(valid_y),
                "--train-target-rank",
                str(train_target_rank),
                "--valid-target-rank",
                str(valid_target_rank),
                "--train-group",
                str(train_group),
                "--valid-group",
                str(valid_group),
                "--evaluation-x",
                str(evaluation_x),
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--scores",
                str(scores_path),
                "--audit",
                str(audit_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            final_line = detail[-1] if detail else "unknown worker failure"
            raise RuntimeError(f"isolated LightGBM worker failed: {final_line}")
        scores = np.load(scores_path).astype(np.float64)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    finally:
        for temporary in [scores_path, audit_path, config_path]:
            temporary.unlink(missing_ok=True)
    checkpoint_path.with_suffix(".preprocessing.json").write_text(
        json.dumps(preprocessing, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return scores, audit
