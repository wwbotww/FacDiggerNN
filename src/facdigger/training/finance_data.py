"""Finance selection inputs and the data semantics bound to training state."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from facdigger.data.config import FinanceSelectionConfig
from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.datasets.splits import build_finance_selection_indices
from facdigger.training.finance_pretrain_config import FinancePretrainingExperimentConfig
from facdigger.training.finance_transformer_config import FinanceTransformerExperimentConfig

SUPERVISED_UNIVERSE = "target_free_inference_index"
PRETRAINING_UNIVERSE = "pretraining_index_before_probe_selection"
SUPERVISED_LABEL_SUPPORT = "complete_horizon_labels_on_original_phase_dates"
PROBE_LABEL_SUPPORT = "probe_complete_horizon_labels"


def _artifact_path(dataset_dir: Path, manifest: dict[str, Any], name: str) -> Path:
    relative = manifest.get("artifacts", {}).get(name)
    if not isinstance(relative, str):
        raise DataContractError(f"finance snapshot requires {name}; rebuild the snapshot")
    path = (dataset_dir / relative).resolve()
    if not path.is_relative_to(dataset_dir.resolve()) or not path.is_file():
        raise DataContractError(f"finance snapshot artifact is missing or outside snapshot: {name}")
    return path


def load_finance_selection(
    dataset_dir: str | Path,
    manifest: dict[str, Any],
    *,
    sample_index: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Reconstruct the one selection plan and verify actual preprocessing/SSL bounds."""
    root = Path(dataset_dir)
    config = manifest["config"]
    if config["features"]["name"] != "finance_transformer":
        raise DataContractError("finance selection requires finance_transformer features")
    if config.get("finance_selection") is None:
        raise DataContractError("finance selection plan is missing; rebuild a new snapshot")
    selection = FinanceSelectionConfig.model_validate(config["finance_selection"])
    saved_plan = json.loads(
        _artifact_path(root, manifest, "finance_selection_plan").read_text(encoding="utf-8")
    )
    if sample_index is None:
        sample_index = pl.read_parquet(_artifact_path(root, manifest, "sample_index"))
    calendar = (
        pl.scan_parquet(_artifact_path(root, manifest, "market_features"))
        .select("trade_date")
        .unique()
        .sort("trade_date")
        .collect()["trade_date"]
        .to_list()
    )
    supervised, probe, plan = build_finance_selection_indices(sample_index, calendar, selection)
    if saved_plan != plan:
        raise DataContractError("finance selection plan differs from snapshot rows/calendar")
    scaler = json.loads(_artifact_path(root, manifest, "scaler").read_text(encoding="utf-8"))
    if scaler.get("method") != "finance_transformer_train_global_robust" or any(
        scaler.get(part, {}).get("fit_end") != plan["a_end"] for part in ("local", "market")
    ):
        raise DataContractError("finance scaler fit_end differs from the pre-probe selection plan")
    a_end = date.fromisoformat(plan["a_end"])
    p_start, s_start = (date.fromisoformat(plan[key]) for key in ("p_start", "s_start"))
    if not a_end < p_start < s_start:
        raise DataContractError("finance selection boundaries are not chronological")
    ssl = pl.scan_parquet(_artifact_path(root, manifest, "pretraining_index"))
    names = ssl.collect_schema().names()
    if any(name.startswith(("target", "label")) or name == "split" for name in names):
        raise DataContractError("finance pretraining index contains supervised columns")
    valid = (
        (pl.col("feature_end") == pl.col("asof_date"))
        & (pl.col("asof_date") < pl.col("future_start"))
        & (pl.col("future_start") <= pl.col("future_end"))
        & (pl.col("future_end") <= a_end)
    ).fill_null(False)
    audit = ssl.select(pl.len().alias("rows"), (~valid).sum().alias("invalid")).collect().row(0)
    if not audit[0] or audit[1]:
        raise DataContractError("finance pretraining index crosses the pre-probe boundary")
    return supervised, probe, plan


def finance_data_protocol(
    dataset_dir: str | Path,
    manifest: dict[str, Any],
    config: FinanceTransformerExperimentConfig | FinancePretrainingExperimentConfig,
    *,
    sample_index: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """Resolve experiment assertions before a run can write state or reuse artifacts."""
    _, _, plan = load_finance_selection(dataset_dir, manifest, sample_index=sample_index)
    features = manifest["config"]["features"]
    if (
        config.channels != features["channels"]
        or config.market_channels != features["market_channels"]
    ):
        raise DataContractError("finance experiment channels differ from snapshot")
    if config.model.statistics_windows[-1] > features["context_length"]:
        raise DataContractError("model statistics window exceeds snapshot context")
    if isinstance(config, FinancePretrainingExperimentConfig):
        expected = {
            "probe_fit_dates": config.training.probe.fit_dates,
            "probe_selection_dates": config.training.probe.selection_dates,
            "future_horizon": config.future_horizon,
        }
        universe, support = PRETRAINING_UNIVERSE, PROBE_LABEL_SUPPORT
    else:
        expected = {"supervised_selection_fraction": config.selection_fraction}
        universe, support = SUPERVISED_UNIVERSE, SUPERVISED_LABEL_SUPPORT
    FinanceSelectionConfig.model_validate(plan["parameters"]).validate_expectations(**expected)
    label = manifest["config"]["label"]
    horizons = sorted([label["horizon"], *label.get("auxiliary_horizons", [])])
    if isinstance(config, FinanceTransformerExperimentConfig):
        if config.horizons != horizons or config.primary_horizon != label["horizon"]:
            raise DataContractError("finance experiment horizons differ from snapshot")
    elif horizons != [1, 5, 20] or label["horizon"] != 5:
        raise DataContractError("finance pretraining probe requires 1/5/20 labels with primary 5")
    return {
        "dataset_id": manifest["dataset_id"],
        "selection_plan": plan,
        "feature_scaler_sha256": sha256_file(_artifact_path(Path(dataset_dir), manifest, "scaler")),
        "computational_universe": universe,
        "label_support": support,
    }


def validate_encoder_data_protocol(
    encoder_payload: dict[str, Any], expected: dict[str, Any]
) -> None:
    actual = encoder_payload.get("data_protocol")
    if not isinstance(actual, dict) or (
        actual.get("computational_universe") != PRETRAINING_UNIVERSE
        or actual.get("label_support") != PROBE_LABEL_SUPPORT
        or any(
            actual.get(key) != expected.get(key)
            for key in ("dataset_id", "selection_plan", "feature_scaler_sha256")
        )
    ):
        raise DataContractError("pretrained encoder data/scaler/selection protocol differs")
