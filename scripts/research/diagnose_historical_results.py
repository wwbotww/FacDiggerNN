"""Read-only validation diagnostics for completed M6 and Finance comparison runs.

No checkpoint is loaded and no holdout is read. Seed averages below average daily
ICs, not model scores. Cross-generation comparisons are descriptive: historical
models were not trained under a matched protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath

import numpy as np
import polars as pl

from facdigger.research.statistics import panel_mean_inference

KEYS = ["asof_date", "security_id"]


def read_predictions(path: Path) -> pl.DataFrame:
    frame = pl.read_parquet(path, columns=[*KEYS, "split", "score_raw", "target"])
    if set(frame["split"].unique()) != {"valid"}:
        raise ValueError(f"Expected validation predictions only: {path}")
    if frame.select(KEYS).is_duplicated().any():
        raise ValueError(f"Duplicate prediction keys: {path}")
    for name in ["score_raw", "target"]:
        if frame[name].null_count() or not frame[name].is_finite().all():
            raise ValueError(f"Non-finite {name}: {path}")
    return frame.drop("split").sort(KEYS)


def daily_ic(frame: pl.DataFrame, score: str = "score_raw") -> pl.DataFrame:
    daily = (
        frame.group_by("asof_date")
        .agg(
            pl.len().alias("n"),
            pl.corr(pl.col(score).rank("average"), pl.col("target").rank("average")).alias(
                "rank_ic"
            ),
        )
        .sort("asof_date")
    )
    if daily["rank_ic"].null_count() or not daily["rank_ic"].is_finite().all():
        raise ValueError("Undefined daily IC; inspect score/target variation")
    return daily


def summarize(frame: pl.DataFrame) -> dict:
    return {
        "dates": frame.height,
        "mean_rank_ic": frame["rank_ic"].mean(),
        "positive_date_ratio": (frame["rank_ic"] > 0).mean(),
        "first_date": str(frame["asof_date"].min()),
        "last_date": str(frame["asof_date"].max()),
        "by_year": {
            str(year): {
                "dates": group.height,
                "mean_rank_ic": group["rank_ic"].mean(),
            }
            for (year,), group in frame.with_columns(
                pl.col("asof_date").dt.year().alias("year")
            ).group_by("year", maintain_order=True)
        },
    }


def paired_inference(groups: list[list[float]], *, null_mean: float) -> dict:
    hac = {}
    for lag in [5, 20, 60]:
        result = panel_mean_inference(
            groups, hac_lags=lag, stride=5, offset=0, null_mean=null_mean
        )["hac"]
        se = result["standard_error"]
        result["two_sided_95_interval"] = (
            [result["mean"] - 1.959963984540054 * se, result["mean"] + 1.959963984540054 * se]
            if se is not None
            else None
        )
        hac[str(lag)] = result
    return {
        "date_weighted_mean": float(np.concatenate(groups).mean()),
        "equal_fold_mean": float(np.mean([np.mean(g) for g in groups])),
        "hac_sensitivity_exploratory": hac,
        "nonoverlapping_means_all_offsets": [
            float(np.concatenate([np.asarray(g)[offset::5] for g in groups]).mean())
            for offset in range(5)
        ],
    }


def diagnose(old_run: Path, finance_run: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    records, evidence, cells = [], [], []
    frames = {}
    paths = []
    matrix_cells = {}
    for family, base, field in [("old", old_run, "validation"), ("finance", finance_run, "stages")]:
        matrix = json.loads((base / "matrix.json").read_text())[field]
        for cell in matrix:
            if cell["status"] != "complete":
                raise ValueError(f"Incomplete matrix cell: {cell}")
            model = cell["model_key"] if family == "old" else cell["stage"]
            if model == "pretraining":
                continue
            run_name = PureWindowsPath(cell["run_dir"]).name
            folder = (
                base
                / "runs"
                / "validation"
                / cell["fold_id"]
                / model
                / f"seed-{cell['seed']}"
                / run_name
                if family == "old"
                else base / "runs" / cell["fold_id"] / model / run_name
            )
            for filename, hash_field in [
                ("predictions.parquet", "predictions_sha256"),
                ("metrics.json", "metrics_sha256"),
                ("manifest.json", "manifest_sha256"),
            ]:
                if hash_field in cell:
                    digest = hashlib.sha256((folder / filename).read_bytes()).hexdigest()
                    if digest != cell[hash_field]:
                        raise ValueError(f"Matrix hash mismatch: {folder / filename}")
            paths.append(folder / "predictions.parquet")
            matrix_cells[folder / "predictions.parquet"] = cell
    for path in sorted(paths):
        is_old = path.is_relative_to(old_run)
        base = old_run if is_old else finance_run
        parts = path.relative_to(base).parts
        fold = next(v for v in parts if v in {"wf1", "wf2", "wf3"})
        model = parts[parts.index(fold) + 1]
        seed = int(next(v.split("-")[1] for v in parts if v.startswith("seed-"))) if is_old else 42
        key = (fold, model, seed)
        if key in frames:
            raise ValueError(f"Duplicate completed cell: {key}")
        manifest = json.loads(path.with_name("manifest.json").read_text())
        if manifest.get("status") != "complete":
            raise ValueError(f"Incomplete run: {path}")
        cell = matrix_cells[path]
        if manifest["seed"] != seed or cell["seed"] != seed:
            raise ValueError(f"Seed mismatch: {path}")
        if manifest["run_id"] != path.parent.name or manifest["evaluation_split"] != "valid":
            raise ValueError(f"Run or split mismatch: {path}")
        if "predictions_sha256" in manifest:
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["predictions_sha256"]:
                raise ValueError(f"Manifest prediction hash mismatch: {path}")
        frame = read_predictions(path)
        frames[key] = frame
        daily = daily_ic(frame)
        reported = json.loads(path.with_name("metrics.json").read_text())
        for field in ["run_id", "dataset_id", "evaluation_split"]:
            if reported[field] != manifest[field]:
                raise ValueError(f"Metric identity mismatch ({field}): {path}")
        expected = (
            pl.DataFrame(reported["metrics"]["raw"]["daily_ic"])
            .select(pl.col("asof_date").str.to_date(), "rank_ic")
            .sort("asof_date")
        )
        if daily["asof_date"].to_list() != expected["asof_date"].to_list():
            raise ValueError(f"Reported dates differ: {path}")
        if expected["rank_ic"].null_count() or not expected["rank_ic"].is_finite().all():
            raise ValueError(f"Non-finite reported IC: {path}")
        error = float(np.max(np.abs(daily["rank_ic"].to_numpy() - expected["rank_ic"].to_numpy())))
        if error > 1e-12:
            raise ValueError(f"Reported IC differs: {path}, {error}")
        records.append(
            daily.with_columns(
                pl.lit(fold).alias("fold"), pl.lit(model).alias("model"), pl.lit(seed).alias("seed")
            )
        )
        cells.append(
            {
                "fold": fold,
                "model": model,
                "seed": seed,
                "rows": frame.height,
                "reported_daily_max_error": error,
                **summarize(daily),
            }
        )
        for name in [
            "predictions.parquet",
            "metrics.json",
            "manifest.json",
            "resolved_config.yaml",
        ]:
            p = path.with_name(name)
            if p.exists():
                evidence.append(
                    {
                        "path": str(p.relative_to(base)),
                        "family": "old" if is_old else "finance",
                        "bytes": p.stat().st_size,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    }
                )
    expected_keys = {
        (fold, model, seed)
        for fold in ["wf1", "wf2", "wf3"]
        for model in ["e0", "e1", "e2", "e3"]
        for seed in [17, 42, 73]
    }
    expected_keys.update(
        (fold, model, 42)
        for fold in ["wf1", "wf2", "wf3"]
        for model in ["scratch", "finance_pretrained"]
    )
    if set(frames) != expected_keys:
        raise ValueError("Expected exactly 36 old and 6 Finance cells with original seeds")
    for fold in ["wf1", "wf2", "wf3"]:
        for family in [{"e0", "e1", "e2", "e3"}, {"scratch", "finance_pretrained"}]:
            support = None
            for (cell_fold, model, _seed), frame in frames.items():
                if cell_fold == fold and model in family:
                    candidate = frame.select(*KEYS, "target")
                    if support is not None and not candidate.equals(support):
                        raise ValueError(
                            f"Family prediction support/labels differ: {fold}, {model}"
                        )
                    support = candidate
    all_daily = pl.concat(records)
    all_daily.write_csv(output / "daily_ic.csv")
    pooled = (
        all_daily.group_by(["model", "fold", "asof_date"])
        .agg(pl.col("rank_ic").mean())
        .sort(["model", "fold", "asof_date"])
    )
    models = {
        model: summarize(group) for (model,), group in pooled.group_by("model", maintain_order=True)
    }
    seeds = []
    for (model, seed), group in all_daily.group_by(["model", "seed"]):
        seeds.append({"model": model, "seed": seed, **summarize(group)})
    pairs = {}
    for left, right in [("finance_pretrained", "scratch"), ("e1", "e0"), ("e3", "e1")]:
        groups = []
        for fold in ["wf1", "wf2", "wf3"]:
            a = pooled.filter((pl.col("model") == left) & (pl.col("fold") == fold))
            b = pooled.filter((pl.col("model") == right) & (pl.col("fold") == fold))
            joined = a.join(b, on="asof_date", suffix="_b", validate="1:1").sort("asof_date")
            if joined.height != a.height or joined.height != b.height:
                raise ValueError("Unmatched dates in paired comparison")
            groups.append((joined["rank_ic"] - joined["rank_ic_b"]).to_list())
        pairs[f"{left}_minus_{right}"] = paired_inference(
            groups, null_mean=0.001 if left == "finance_pretrained" else 0.0
        )
    common_cells, correlations, common_daily = [], [], []
    for fold in ["wf1", "wf2", "wf3"]:
        names = ["scratch", "finance_pretrained", "e0", "e1", "e3"]
        anchor = frames[(fold, "scratch", 42)]
        joined = anchor.select(*KEYS, "target", pl.col("score_raw").alias("scratch"))
        target_errors = {}
        for model in names[1:]:
            frame = frames[(fold, model, 42)]
            joined = joined.join(
                frame.rename({"score_raw": model, "target": f"target_{model}"}),
                on=KEYS,
                validate="1:1",
            )
            target_errors[model] = float((joined["target"] - joined[f"target_{model}"]).abs().max())
        # Require identical evaluation labels before comparing old and new scores.
        if any(v > 1e-12 for v in target_errors.values()):
            raise ValueError(f"Different historical labels in {fold}: {target_errors}")
        for model in names:
            daily = daily_ic(joined, model)
            common_cells.append(
                {
                    "fold": fold,
                    "model": model,
                    "seed": 42,
                    "common_rows": joined.height,
                    "original_rows": frames[(fold, model, 42)].height,
                    "excluded_fraction": 1 - joined.height / frames[(fold, model, 42)].height,
                    **summarize(daily),
                }
            )
            common_daily.append(
                daily.with_columns(pl.lit(fold).alias("fold"), pl.lit(model).alias("model"))
            )
        for left, right in [
            ("scratch", "finance_pretrained"),
            ("scratch", "e0"),
            ("finance_pretrained", "e0"),
            ("scratch", "e1"),
        ]:
            scores = joined.group_by("asof_date").agg(
                pl.corr(pl.col(left).rank("average"), pl.col(right).rank("average")).alias(
                    "score_rank_correlation"
                )
            )
            correlations.append(
                {
                    "fold": fold,
                    "left": left,
                    "right": right,
                    "mean_daily_score_rank_correlation": scores["score_rank_correlation"].mean(),
                }
            )
    common = pl.concat(common_daily)
    common.write_csv(output / "common_daily_ic_seed42.csv")
    result = {
        "scope": "Existing validation only; exploratory, no new confirmation test",
        "interpretation": {
            "training_protocol_matched_across_generations": False,
            "common_keys": "Same evaluation labels and keys, not a matched training intervention",
            "seed_average": "Average of daily ICs, not ensemble prediction scores",
            "uncertainty": (
                "Conditional temporal HAC; does not estimate full training-seed uncertainty"
            ),
        },
        "old_run": old_run.name,
        "finance_run": finance_run.name,
        "cells": cells,
        "models_seed_averaged_daily_ic": models,
        "seed_sensitivity": sorted(seeds, key=lambda x: (x["model"], x["seed"])),
        "paired_daily_ic": pairs,
        "common_keys_seed42": common_cells,
        "common_models_seed42": {
            model: summarize(group)
            for (model,), group in common.group_by("model", maintain_order=True)
        },
        "common_score_rank_correlations": correlations,
        "source_files": evidence,
    }
    (output / "diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--finance-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.old_run, args.finance_run, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cells": len(result["cells"]),
                "paired_daily_ic": result["paired_daily_ic"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
