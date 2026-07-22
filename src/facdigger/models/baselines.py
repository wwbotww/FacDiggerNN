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

from facdigger.training.e0_config import LightGBMBaselineConfig, MLPBaselineConfig


def build_multiscale_features(
    features: pl.DataFrame,
    sample_index: pl.DataFrame,
    *,
    channels: list[str],
    windows: list[int],
    context_length: int,
) -> tuple[pl.DataFrame, list[str]]:
    usable_windows = [window for window in windows if window <= context_length]
    if not usable_windows:
        raise ValueError("at least one statistics window must fit within context_length")
    expressions: list[pl.Expr] = []
    feature_columns: list[str] = []
    for channel in channels:
        last_name = f"{channel}__last"
        expressions.append(pl.col(channel).alias(last_name))
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
                expressions.append(expression.over("security_id").alias(name))
                feature_columns.append(name)
    tabular = (
        features.sort(["security_id", "trade_date"])
        .with_columns(expressions)
        .select(
            "security_id",
            pl.col("trade_date").alias("asof_date"),
            *feature_columns,
        )
        .join(
            sample_index.select(
                "sample_id",
                "security_id",
                "symbol",
                "asof_date",
                "split",
                "target",
            ),
            on=["security_id", "asof_date"],
            how="inner",
            validate="1:1",
        )
        .sort(["asof_date", "security_id"])
    )
    return tabular, feature_columns


@dataclass(frozen=True)
class TabularPreprocessor:
    feature_columns: list[str]
    means: list[float]
    scales: list[float]

    @classmethod
    def fit(cls, frame: pl.DataFrame, feature_columns: list[str]) -> TabularPreprocessor:
        values = frame.select(feature_columns).to_numpy().astype(np.float64)
        means: list[float] = []
        scales: list[float] = []
        for column in values.T:
            observed = column[np.isfinite(column)]
            if not len(observed):
                means.append(0.0)
                scales.append(1.0)
                continue
            means.append(float(np.mean(observed)))
            scale = float(np.std(observed))
            scales.append(scale if scale > 1e-12 else 1.0)
        return cls(feature_columns, means, scales)

    def transform(self, frame: pl.DataFrame) -> np.ndarray:
        values = frame.select(self.feature_columns).to_numpy().astype(np.float64)
        observed = np.isfinite(values)
        normalized = (values - np.asarray(self.means)) / np.asarray(self.scales)
        normalized = np.where(observed, normalized, 0.0)
        return np.concatenate([normalized, observed.astype(np.float64)], axis=1).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_columns": self.feature_columns,
            "means": self.means,
            "scales": self.scales,
            "missing_policy": "train_standardize_then_zero_fill_plus_observed_mask",
        }


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
    *,
    config: MLPBaselineConfig,
    seed: int,
    checkpoint_path: Path,
    preprocessing: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _torch_device(config.device)
    layers: list[nn.Module] = []
    width = train_x.shape[1]
    for hidden in config.hidden_dims:
        layers.extend([nn.Linear(width, hidden), nn.GELU(), nn.Dropout(config.dropout)])
        width = hidden
    layers.append(nn.Linear(width, 1))
    model = nn.Sequential(*layers).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = nn.HuberLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32))),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    valid_features = torch.from_numpy(valid_x).to(device)
    valid_targets = torch.from_numpy(valid_y.astype(np.float32)).to(device)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float | int]] = []
    stale_epochs = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x).squeeze(-1)
            loss = loss_function(prediction, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_y)
            total_rows += len(batch_y)
        model.eval()
        with torch.no_grad():
            valid_prediction = model(valid_features).squeeze(-1)
            valid_loss = float(loss_function(valid_prediction, valid_targets).cpu())
        train_loss = total_loss / max(total_rows, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        if valid_loss < best_loss - 1e-12:
            best_loss = valid_loss
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
            "model_type": "mlp",
            "input_dim": train_x.shape[1],
            "hidden_dims": config.hidden_dims,
            "dropout": config.dropout,
            "state_dict": best_state,
            "preprocessing": preprocessing,
            "best_epoch": best_epoch,
            "best_valid_loss": best_loss,
            "seed": seed,
        },
        checkpoint_path,
    )
    return model, {
        "device": device,
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "epochs_ran": len(history),
        "history": history,
    }


def predict_mlp(model: Any, values: np.ndarray, device: str) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        result = model(torch.from_numpy(values).to(device)).squeeze(-1).cpu().numpy()
    return result.astype(np.float64)


def train_lightgbm(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    evaluation_x: np.ndarray,
    *,
    config: LightGBMBaselineConfig,
    seed: int,
    checkpoint_path: Path,
    preprocessing: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train in a clean process to isolate native OpenMP runtimes on macOS."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    input_path = checkpoint_path.with_suffix(".input.npz")
    scores_path = checkpoint_path.with_suffix(".scores.npy")
    audit_path = checkpoint_path.with_suffix(".audit.json")
    config_path = checkpoint_path.with_suffix(".config.json")
    np.savez_compressed(
        input_path,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        evaluation_x=evaluation_x,
    )
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
                "--input",
                str(input_path),
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
        for temporary in [input_path, scores_path, audit_path, config_path]:
            temporary.unlink(missing_ok=True)
    checkpoint_path.with_suffix(".preprocessing.json").write_text(
        json.dumps(preprocessing, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return scores, audit
