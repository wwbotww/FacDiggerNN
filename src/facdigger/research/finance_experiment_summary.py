"""Read committed full F/S observations without training, reselection or V access."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.research.finance_experiment_config import cell_config
from facdigger.research.finance_experiment_plan import (
    clean_git,
    disjoint_output,
    read_experiment_plan,
)
from facdigger.training.runtime import write_json

METRICS = (
    "rank_ic",
    "raw_ic",
    "surrogate_correlation",
    "score_std",
    "weighted_available_loss",
    "weighted_available_scale_penalty",
)
SUPPORT = ["asof_date", "horizon", "computational_rows", "labelled_rows"]


def checked_file(root: Path, name: str, expected: str, hashes: dict) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise DataContractError("summary source is missing or outside its cell")
    actual = sha256_file(path)
    if actual != expected:
        raise DataContractError(f"summary source checksum differs: {path}")
    hashes[str(path)] = actual
    return path


def read_completed_cell(source: Path, plan: dict, cell: dict, hashes: dict) -> tuple[dict, list]:
    root = source / "cells" / cell["cell_id"]
    path = root / "cell.json"
    before = sha256_file(path)
    audit = json.loads(path.read_text())
    expected = {
        "plan_hash": plan["plan_hash"],
        "cell": cell,
        "config": cell_config(plan["identity"]["supervised_config"], cell["seed"]).model_dump(
            mode="json",
        ),
        "data_protocol": plan["identity"]["folds"][cell["fold_id"]]["data_protocol"],
        "execution": json.loads((source / "execution.json").read_text()),
    }
    if (
        audit.get("status") != "complete"
        or audit["identity"] != expected
        or audit.get("validation_used") is not False
        or audit.get("holdout_used") is not False
    ):
        raise DataContractError("summary requires complete cells with unchanged F/S identities")
    for name in ("history.json", "checkpoints/best.pt", "checkpoints/last.pt"):
        checked_file(root, name, audit["artifacts"][name], hashes)
    history = json.loads((root / "history.json").read_text())
    if not history or [row["epoch"] for row in history] != list(range(1, audit["epoch"] + 1)):
        raise DataContractError("complete history has missing epochs")
    best = history[0]
    for row in history[1:]:
        if row["selection"]["selection_score"] > best["selection"]["selection_score"] + 1e-12:
            best = row
    if (
        best["epoch"] != audit["best_epoch"]
        or best["selection"]["selection_score"] != audit["best_selection_score"]
        or history[-1]["global_step"] != audit["global_step"]
    ):
        raise DataContractError("native S selection differs from complete history")
    checked_file(root, "cell.json", before, hashes)
    return audit, history


def summarize_experiments(
    sources: dict[str, Path],
    output: Path,
    *,
    repository_root: Path,
    comparisons: list[dict] | None = None,
) -> dict:
    """All cells, epochs and seeds; comparisons name explicit source/candidate pairs."""
    git = clean_git(repository_root)
    if not sources or output.exists():
        raise DataContractError("summary requires sources and a new output directory")
    for source in sources.values():
        disjoint_output(output, source)
    hashes, supports, trajectories, segments, cells, plans = {}, {}, [], [], [], {}
    training = []
    executions, protocols = set(), {}
    for source_id, source in sources.items():
        plan = read_experiment_plan(source)
        plans[source_id] = {
            "path": str(source.resolve()),
            "plan_hash": plan["plan_hash"],
            "training_commit": plan["identity"]["code_commit"],
        }
        for name in ("plan.json", "execution.json"):
            hashes[str((source / name).resolve())] = sha256_file(source / name)
        executions.add(
            json.dumps(json.loads((source / "execution.json").read_text()), sort_keys=True)
        )
        for fold, binding in plan["identity"]["folds"].items():
            disjoint_output(output, Path(binding["path"]))
            common = {k: binding[k] for k in ("dataset_id", "checksums_sha256", "data_protocol")}
            if fold in protocols and protocols[fold] != common:
                raise DataContractError("summary sources use different fold data boundaries")
            protocols[fold] = common
        for cell in plan["identity"]["cells"]:
            root = source / "cells" / cell["cell_id"]
            audit, history = read_completed_cell(source, plan, cell, hashes)
            for entry in history:
                training.append(
                    {
                        "source": source_id,
                        **cell,
                        **{
                            key: value
                            for key, value in entry.items()
                            if isinstance(value, (int, float)) or value is None
                        },
                        **{
                            f"native_S_{key}": value
                            for key, value in entry["selection"].items()
                            if isinstance(value, (int, float)) or value is None
                        },
                        **{f"lr_{key}": value for key, value in entry["learning_rates"].items()},
                    }
                )
            native = {row["epoch"]: row for row in history}
            metrics = {}
            for epoch in range(audit["epoch"] + 1):
                marker_name = f"observations/epoch-{epoch}.json"
                marker = json.loads(
                    checked_file(
                        root,
                        marker_name,
                        audit["artifacts"][marker_name],
                        hashes,
                    ).read_text()
                )
                if (
                    marker["identity"] != audit["identity"]
                    or marker["epoch"] != epoch
                    or marker["optimizer_updates"] != 0
                ):
                    raise DataContractError("observation binding or zero-update contract differs")
                for phase in ("F", "S"):
                    name = f"epoch-{epoch}-{phase}-daily.parquet"
                    if marker["files"][name] != audit["artifacts"][name]:
                        raise DataContractError("daily observation inventory differs")
                    table = pl.read_parquet(
                        checked_file(
                            root,
                            name,
                            marker["files"][name],
                            hashes,
                        )
                    ).sort("asof_date", "horizon")
                    expected = plan["identity"]["folds"][cell["fold_id"]]["support"][phase]
                    horizons = plan["identity"]["supervised_config"]["horizons"]
                    if (
                        table.height != expected["dates"] * len(horizons)
                        or sorted(table["horizon"].unique().to_list()) != sorted(horizons)
                        or table.select("asof_date", "horizon").unique().height != table.height
                        or not table["full_objective_available"].all()
                        or table["prediction_coverage"].min() != 1.0
                    ):
                        raise DataContractError("incomplete F/S daily support; no intersection")
                    key = cell["fold_id"], phase
                    support = table.select(SUPPORT)
                    if key in supports and not supports[key].equals(support):
                        raise DataContractError(
                            "F/S support differs across epochs, cells or sources"
                        )
                    supports[key] = support
                    for horizon in horizons:
                        part = table.filter(pl.col("horizon") == horizon)
                        for field in ("computational_rows", "labelled_rows"):
                            if part[field].sum() != expected[field]:
                                raise DataContractError("daily C/L differs from frozen support")
                        row = {
                            "source": source_id,
                            **cell,
                            "epoch": epoch,
                            "phase": phase,
                            "horizon": horizon,
                            "dates": part.height,
                        }
                        for field in METRICS:
                            values = part[field].to_numpy()
                            if not np.isfinite(values).all():
                                raise DataContractError(
                                    f"undefined {field}; retain failure, no filtering"
                                )
                            row[field] = float(values.mean())
                        if phase == "S":
                            objective = plan["identity"]["supervised_config"]["training"][
                                "objective"
                            ]
                            pieces = np.array_split(
                                part["rank_ic"].to_numpy(), objective["selection_subperiods"]
                            )
                            row["diagnostic_selection_score"] = row["rank_ic"] - (
                                objective["selection_stability_penalty"]
                                * float(np.std([x.mean() for x in pieces]))
                            )
                        trajectories.append(row)
                        metrics[epoch, phase, horizon] = row
                        if horizon == 5:
                            dates = part["asof_date"].to_list()
                            blocks = [
                                (f"block-{i + 1}", indices)
                                for i, indices in enumerate(
                                    np.array_split(
                                        np.arange(len(dates)),
                                        min(len(dates), 3 if phase == "S" else 5),
                                    ),
                                )
                            ]
                            blocks += [
                                (
                                    str(year),
                                    np.array(
                                        [i for i, day in enumerate(dates) if day.year == year]
                                    ),
                                )
                                for year in sorted({day.year for day in dates})
                            ]
                            for label, indices in blocks:
                                segments.append(
                                    {
                                        "source": source_id,
                                        **cell,
                                        "epoch": epoch,
                                        "phase": phase,
                                        "segment": label,
                                        "dates": len(indices),
                                        "first_date": str(dates[indices[0]]),
                                        "last_date": str(dates[indices[-1]]),
                                        "rank_ic": float(
                                            part["rank_ic"].to_numpy()[indices].mean()
                                        ),
                                    }
                                )
            summary = {
                "source": source_id,
                **cell,
                "best_epoch": audit["best_epoch"],
                "final_epoch": audit["epoch"],
                "native_best_selection_score": audit["best_selection_score"],
                "amp_skipped_steps": sum(
                    row.get("amp_skipped_optimizer_steps", 0) for row in history
                ),
                "mean_gradient_clip_ratio": float(
                    np.mean([row["gradient_clip_ratio"] for row in history])
                ),
                "max_gradient_norm": max(row["max_pre_clip_gradient_norm"] for row in history),
            }
            for label, epoch in [(f"e{e}", e) for e in (0, 1, 2, 6) if e <= audit["epoch"]] + [
                ("best", audit["best_epoch"]),
                ("last", audit["epoch"]),
            ]:
                for phase in ("F", "S"):
                    for field in METRICS:
                        summary[f"{label}_{phase}_{field}"] = metrics[epoch, phase, 5][field]
                summary[f"{label}_native_selection_score"] = (
                    native[epoch]["selection"]["selection_score"] if epoch else None
                )
            cells.append(summary)
    if len(executions) != 1:
        raise DataContractError("paired summary numerical execution environments differ")
    paired = []
    for pair in comparisons or []:
        arms = []
        for arm in ("candidate", "control"):
            binding = pair[arm]
            arms.append(
                {
                    (row["fold_id"], row["seed"]): row
                    for row in cells
                    if row["source"] == binding["source"]
                    and row["candidate"] == binding["candidate"]
                }
            )
        if not arms[0] or arms[0].keys() != arms[1].keys():
            raise DataContractError("paired summary requires all paired folds and seeds")
        for key, left in arms[0].items():
            right = arms[1][key]
            for metric in (
                "native_best_selection_score",
                "best_F_rank_ic",
                "best_S_rank_ic",
                "last_F_rank_ic",
                "last_S_rank_ic",
            ):
                paired.append(
                    {
                        "comparison": pair["name"],
                        "fold_id": key[0],
                        "seed": key[1],
                        "metric": metric,
                        "delta": left[metric] - right[metric],
                    }
                )
    # Recheck audit/inputs after analysis so an active writer cannot produce a mixed report.
    for path, expected_hash in hashes.items():
        if sha256_file(Path(path)) != expected_hash:
            raise DataContractError("summary source changed during analysis")
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (
        ("cells", cells),
        ("trajectory", trajectories),
        ("segments", segments),
        ("paired", paired),
        ("training", training),
    ):
        if rows:
            pl.from_dicts(rows, infer_schema_length=None).write_csv(output / f"{name}.csv")
    curve = (
        pl.from_dicts(trajectories, infer_schema_length=None)
        .group_by(
            "source",
            "fold_id",
            "candidate",
            "epoch",
            "phase",
            "horizon",
        )
        .agg(pl.len().alias("seeds_present"), *[pl.col(field).mean() for field in METRICS])
        .sort(
            "source",
            "fold_id",
            "candidate",
            "epoch",
            "phase",
            "horizon",
        )
    )
    curve.write_csv(output / "curves.csv")
    write_json(output / "source-hashes.json", hashes)
    report = {
        "analysis_commit": git["commit"],
        "sources": plans,
        "cells": len(cells),
        "validation_used": False,
        "holdout_used": False,
        "paired_comparisons": comparisons or [],
        "common_epoch_range": [0, min(row["final_epoch"] for row in cells)],
        "limitations": [
            "S selects native checkpoints; FP32 observations never reselect",
            "Curves report seeds_present; later epochs may have fewer seeds",
            "Seeds share dates and F/S fold periods overlap; descriptive evidence only",
            "Hashes cover checkpoints, history, markers and daily tables; "
            "predictions/chunks are not fully rescanned",
            "PIT/source_research_ready limitations remain; no investable validity claim",
        ],
    }
    write_json(output / "report.json", report)
    return report
