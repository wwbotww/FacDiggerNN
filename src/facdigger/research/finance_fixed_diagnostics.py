"""Complete the approved F observations and loss geometry without training or S/V scoring."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.datasets.finance_statistics import FinanceStatisticsDataset
from facdigger.datasets.window import FinanceTransformerInferenceWindowDataset
from facdigger.environment import collect_environment
from facdigger.experiments.manifest import collect_git_state
from facdigger.models.finance_patch_transformer import build_finance_transformer_model
from facdigger.models.finance_statistics import FinanceStatisticsRanker
from facdigger.research.finance_diagnostics import (
    diagnostic_inputs,
    fixed_dates,
    score_panel,
    statistics_style_exposures,
    window_datasets,
)
from facdigger.research.finance_fixed_gradients import loss_gradient_diagnostics
from facdigger.training.e1_engine import select_device
from facdigger.training.progress import append_progress
from facdigger.training.runtime import (
    TrainingControl,
    TrainingRuntimeConfig,
    _sync_directory,
    run_lock,
    write_json,
)

CHUNK_DATES = 16
ALLOCATION_CAPS = {"finance": 7200, "statistics_linear": 600, "statistics_mlp": 1200}
STYLES = ["momentum_scaled_mean_20", "reversal_scaled_mean_5", "volatility_latest_vol20"]
KEYS = ["sample_id", "security_id", "asof_date"]


def fit_inputs(snapshot: Path, checksums: Path, config, expected_protocol: dict):
    """Reconstruct the existing plan using metadata; collect target values for F only."""
    manifest, protocol, labelled, pools = diagnostic_inputs(
        snapshot, checksums, config, phases=("F",)
    )
    if protocol != expected_protocol:
        raise DataContractError("fixed diagnostic data protocol differs from C")
    return manifest, labelled["F"], pools["F"]


def subset_dates(dataset, dates):
    if isinstance(dataset, FinanceStatisticsDataset):
        return dataset.subset(dates)
    return FinanceTransformerInferenceWindowDataset(
        feature_store=dataset.feature_store,
        market_store=dataset.market_store,
        inference_index=dataset.sample_rows.filter(pl.col("asof_date").is_in(dates)),
        channels=dataset.channels,
        market_channels=list(dataset.market_channels),
        context_length=dataset.context_length,
        primary_horizon=dataset.primary_horizon,
    )


def validate_observation(tables: dict, pool: pl.DataFrame, fit: pl.DataFrame, config) -> None:
    """Check exact prediction keys and expected daily denominators, never intersect successes."""
    predictions, daily, styles = (tables[key] for key in ("predictions", "daily", "styles"))
    if not predictions.select(KEYS).equals(pool.select(KEYS)):
        raise DataContractError("fixed observation prediction keys differ from complete C")
    for h in config.horizons:
        if (
            predictions[f"score_{h}"].null_count()
            or not predictions[f"score_{h}"].is_finite().all()
        ):
            raise DataContractError("fixed observation has missing/non-finite predictions")
    expected = (
        fit.group_by("asof_date")
        .len()
        .rename({"len": "labelled_rows"})
        .join(
            pool.group_by("asof_date").len().rename({"len": "computational_rows"}),
            on="asof_date",
            validate="1:1",
        )
    )
    keys = ["asof_date", "horizon", "labelled_rows", "computational_rows"]
    wanted = expected.join(pl.DataFrame({"horizon": config.horizons}), how="cross")
    if (
        not daily.select(keys)
        .sort("asof_date", "horizon")
        .equals(wanted.select(keys).sort("asof_date", "horizon"))
    ):
        raise DataContractError("fixed observation daily support differs; no intersection")
    if not daily["full_objective_available"].all() or not (daily["prediction_coverage"] == 1).all():
        raise DataContractError("fixed observation objective or coverage is incomplete")
    style_keys = ["asof_date", "style", "labelled_rows"]
    wanted_styles = expected.select("asof_date", "labelled_rows").join(
        pl.DataFrame({"style": STYLES}),
        how="cross",
    )
    if (
        not styles.select(style_keys)
        .sort("asof_date", "style")
        .equals(wanted_styles.select(style_keys).sort("asof_date", "style"))
    ):
        raise DataContractError("fixed observation style support differs")


def _atomic_parquet(path: Path, table: pl.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        table.write_parquet(stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _sync_directory(path.parent)


def _write_chunk(path: Path, tables: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        _atomic_parquet(path / f"{name}.parquet", table)
    # This marker alone commits the entire group of complete dates.
    write_json(
        path / "complete.json", {name: sha256_file(path / f"{name}.parquet") for name in tables}
    )


def _read_chunk(path: Path) -> dict:
    checksums = json.loads((path / "complete.json").read_text())
    if set(checksums) != {"daily", "styles", "predictions"}:
        raise DataContractError("fixed observation chunk manifest differs")
    for name, expected in checksums.items():
        if sha256_file(path / f"{name}.parquet") != expected:
            raise DataContractError("fixed observation committed chunk checksum mismatch")
    return {name: pl.read_parquet(path / f"{name}.parquet") for name in checksums}


def _source_files(source: Path) -> list[Path]:
    return [source / "diagnostic.json", source / "checkpoints/last.pt"] + [
        source / name
        for epoch in (1, 2)
        for name in (
            f"observations/epoch-{epoch}.pt",
            f"epoch-{epoch}-F-daily.parquet",
            f"epoch-{epoch}-F-predictions.parquet",
            f"epoch-{epoch}-F-styles.parquet",
        )
    ]


def _linear_chunk_source(path: Path, identity: dict) -> dict:
    """Bind an explicitly reused, paused Linear observation; never rewrite its identity.

    A new output can use already committed scores after a runtime-code change.
    Ordinary in-place continuation still requires the exact code commit. All
    scientific fields, source weight hashes, precision and dependencies must match.
    """
    previous = json.loads((path / "diagnostic.json").read_text())
    previous_identity = previous["identity"]
    current_science = {k: v for k, v in identity.items() if k != "code_commit"}
    previous_science = {k: v for k, v in previous_identity.items() if k != "code_commit"}
    if (
        identity["source"]["candidate"] != "statistics_linear"
        or previous_science != current_science
        or previous.get("status") != "paused_budget_or_signal"
        or previous.get("optimizer_updates") != 0
        or previous.get("holdout_used") is not False
    ):
        raise DataContractError("reused Linear chunks differ from the fixed scientific protocol")
    elapsed = 0.0
    for attempt in previous["attempts"]:
        seconds = attempt.get("elapsed_seconds")
        if seconds is None or not math.isfinite(seconds) or seconds < 0:
            raise DataContractError("reused Linear attempt has no closed resource accounting")
        elapsed += seconds
    manifests, dates = {}, {"1": 0, "2": 0}
    for marker in sorted((path / "chunks").glob("*/complete.json")):
        match = re.fullmatch(r"epoch-([12])-\d{4}", marker.parent.name)
        if match is None:
            raise DataContractError("reused Linear chunk name differs")
        tables = _read_chunk(marker.parent)
        manifests[marker.parent.name] = sha256_file(marker)
        dates[match[1]] += tables["daily"]["asof_date"].n_unique()
    if not manifests:
        raise DataContractError("reused Linear run has no committed chunks")
    return {
        "path": str(path.resolve()),
        "audit_sha256": sha256_file(path / "diagnostic.json"),
        "identity": previous_identity,
        "chunk_manifests": manifests,
        "dates_per_epoch": dates,
        "elapsed_seconds": elapsed,
    }


def _summary(output: Path, dates: list, gradients: list) -> dict:
    bins = {
        day: i + 1 for i, part in enumerate(np.array_split(np.asarray(dates), 5)) for day in part
    }
    result = {"optimizer_updates": 0, "F_dates": len(dates), "epochs": {}, "gradients": gradients}
    tables = {}
    for epoch in (1, 2):
        table = pl.read_parquet(output / f"epoch-{epoch}-F-daily.parquet")
        table = table.with_columns(pl.Series("time_bin", [bins[d] for d in table["asof_date"]]))
        tables[epoch] = table
        aggregate = [
            pl.len().alias("expected_dates"),
            pl.col("rank_ic").count().alias("finite_ic_dates"),
            *[
                pl.col(col).mean()
                for col in (
                    "rank_ic",
                    "raw_ic",
                    "surrogate_correlation",
                    "score_std",
                    "weighted_available_loss",
                    "weighted_available_scale_penalty",
                )
            ],
        ]
        result["epochs"][epoch] = {
            "all_F": table.group_by("horizon").agg(*aggregate).sort("horizon").to_dicts(),
            "five_time_bins": table.group_by("time_bin", "horizon")
            .agg(*aggregate)
            .sort("time_bin", "horizon")
            .to_dicts(),
        }
    left, right = [tables[e].filter(pl.col("horizon") == 5).sort("asof_date") for e in (2, 1)]
    result["primary_epoch2_minus_epoch1"] = {
        "mean": (left["rank_ic"] - right["rank_ic"]).mean(),
        "finite_dates": (left["rank_ic"] - right["rank_ic"]).count(),
        "interpretation": "in-sample descriptive change; no generalization inference",
    }
    result["gradient_summary"] = []
    for epoch in (1, 2):
        rows = [row for row in gradients if row["epoch"] == epoch]
        if not rows:
            continue
        for group in rows[0]["geometry"]:
            summary = {"epoch": epoch, "group": group, "expected_dates": len(rows)}
            for metric in ("weighted_norms", "norm_fractions", "cosines"):
                summary[metric] = {}
                for name in rows[0]["geometry"][group][metric]:
                    values = [row["geometry"][group][metric][name] for row in rows]
                    finite = [value for value in values if value is not None]
                    summary[metric][name] = {
                        "finite_dates": len(finite),
                        "mean": float(np.mean(finite)) if finite else None,
                        "median": float(np.median(finite)) if finite else None,
                        "negative_dates": sum(value < 0 for value in finite),
                    }
            result["gradient_summary"].append(summary)
    return result


def run_fixed_diagnostics(
    snapshot: Path,
    checksums: Path,
    config,
    cache: Path,
    source: Path,
    output: Path,
    *,
    budget_seconds: float,
    repository_root: Path,
    cumulative_budget_seconds: float | None = None,
    shutdown_margin_seconds: float = 120,
    reuse_linear_chunks: Path | None = None,
) -> dict:
    started = time.monotonic()
    protected_inputs = [snapshot, cache, source]
    if reuse_linear_chunks is not None:
        protected_inputs.append(reuse_linear_chunks)
    for protected in protected_inputs:
        if output.resolve().is_relative_to(
            protected.resolve()
        ) or protected.resolve().is_relative_to(output.resolve()):
            raise ValueError("fixed diagnostic output must be disjoint from immutable inputs")
    original = json.loads((source / "diagnostic.json").read_text())
    candidate = original["identity"]["candidate"]
    if candidate not in ALLOCATION_CAPS:
        raise ValueError("unknown fixed diagnostic candidate")
    cumulative_budget = (
        ALLOCATION_CAPS[candidate]
        if cumulative_budget_seconds is None
        else cumulative_budget_seconds
    )
    runtime = TrainingRuntimeConfig(
        max_walltime_seconds=budget_seconds,
        shutdown_margin_seconds=shutdown_margin_seconds,
        handle_signals=hasattr(signal, "SIGUSR1"),
    )
    if not math.isfinite(cumulative_budget) or not 0 < budget_seconds <= cumulative_budget:
        raise ValueError("fixed diagnostic candidate/allocation budget is outside approved bounds")
    if (
        original.get("status") != "observations_complete"
        or original.get("training_status") != "paused"
        or original.get("holdout_used") is not False
        or original.get("committed_epoch") != 2
        or original.get("committed_phase") != "epoch_complete"
        or original["identity"]["config"] != config.model_dump(mode="json")
    ):
        raise DataContractError("fixed diagnostics require the completed, unchanged C prefix")
    git = collect_git_state(repository_root)
    if git["dirty"]:
        raise DataContractError("fixed diagnostics require committed clean code")
    source_hashes = {str(p.relative_to(source)): sha256_file(p) for p in _source_files(source)}
    if source_hashes["checkpoints/last.pt"] != original["checkpoint_sha256"]:
        raise DataContractError("original C checkpoint checksum differs")
    environment = collect_environment()
    source_environment = original["attempts"][-1]["environment"]
    critical_dependencies = ("numpy", "torch", "transformers")
    dependency_versions = {
        row["name"]: row["installed_version"]
        for row in environment.get("dependencies", [])
        if row["name"] in critical_dependencies
    }
    source_versions = {
        row["name"]: row["installed_version"]
        for row in source_environment.get("dependencies", [])
        if row["name"] in critical_dependencies
    }
    if dependency_versions != source_versions:
        raise DataContractError("fixed scoring dependencies differ from reused C observations")
    device = select_device(config.training.device)
    source_cuda = source_environment.get("torch", {}).get("cuda_available")
    source_device = (
        ("cuda" if source_cuda else "cpu")
        if config.training.device == "auto"
        else (config.training.device)
    )
    if device != source_device:
        raise DataContractError("fixed scoring device type differs from reused C observations")
    # Existing C observations predate the indexed-CUDA AMP fix and used FP32.
    # New C records the effective precision explicitly. Never mix the two in F.
    forward_precision = original.get("observation_precision", "fp32")
    if forward_precision not in {"fp32", "fp16"}:
        raise DataContractError("unknown source observation precision")
    if forward_precision == "fp16" and device != "cuda":
        raise DataContractError("cannot reuse FP16 observations with CPU forward scoring")
    identity = {
        "source": original["identity"],
        "source_sha256": source_hashes,
        "code_commit": git["commit"],
        "chunk_dates": CHUNK_DATES,
        "gradient_dates_policy": "four_F_time_bins_two_equidistant_dates_each",
        "gradient_mode": "eval",
        "device": device,
        "dependency_versions": dependency_versions,
        "fixed_forward_precision": forward_precision,
        "gradient_sum_tolerance": 5e-3
        if device == "cuda" and config.training.precision == "fp16"
        else 2e-5,
    }
    scientific_identity = identity.copy()
    if reuse_linear_chunks is not None:
        identity["reused_linear_chunks"] = _linear_chunk_source(reuse_linear_chunks, identity)
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "diagnostic.json"
    with run_lock(output / ".diagnostic.lock"):
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            if audit["identity"] != identity:
                raise DataContractError("fixed diagnostic continuation identity differs")
        else:
            if any(p.name != ".diagnostic.lock" for p in output.iterdir()):
                raise DataContractError("fixed diagnostic output has no bound identity")
            audit = {
                "identity": identity,
                "attempts": [],
                "optimizer_updates": 0,
                "holdout_used": False,
            }
        # An unclosed attempt conservatively consumes its reserved budget.
        used = identity.get("reused_linear_chunks", {}).get("elapsed_seconds", 0.0)
        used += sum(a.get("elapsed_seconds", a["budget_seconds"]) for a in audit["attempts"])
        if used + budget_seconds > cumulative_budget:
            raise ValueError("retry budget must debit previous fixed diagnostic attempts")
        audit["attempts"].append(
            {
                "environment": environment,
                "budget_seconds": budget_seconds,
                "cumulative_budget_seconds": cumulative_budget,
                "shutdown_margin_seconds": shutdown_margin_seconds,
                "prior_elapsed_seconds": used,
            }
        )
        audit["status"] = "running"
        audit.pop("reason", None)
        audit.pop("error", None)
        write_json(audit_path, audit)
        control = TrainingControl(runtime)
        control.started = started

        def check_stop():
            if reason := control.stop_reason():
                raise TimeoutError(reason)

        try:
            with control:
                _observe_fixed(
                    snapshot,
                    checksums,
                    config,
                    cache,
                    source,
                    output,
                    original,
                    candidate,
                    device,
                    audit,
                    check_stop,
                    started,
                    budget_seconds,
                    shutdown_margin_seconds,
                )
                if source_hashes != {
                    str(p.relative_to(source)): sha256_file(p) for p in _source_files(source)
                }:
                    raise DataContractError("original C inputs changed during fixed diagnostics")
                if reuse_linear_chunks is not None and identity["reused_linear_chunks"] != (
                    _linear_chunk_source(reuse_linear_chunks, scientific_identity)
                ):
                    raise DataContractError("reused Linear inputs changed during fixed diagnostics")
                audit["status"] = "fixed_diagnostics_complete"
        except TimeoutError as exc:
            audit.update(status="paused_budget_or_signal", reason=str(exc))
        except Exception as exc:
            audit.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            audit["attempts"][-1]["elapsed_seconds"] = time.monotonic() - started
            write_json(audit_path, audit)
        return audit


def _observe_fixed(
    snapshot,
    checksums,
    config,
    cache,
    source,
    output,
    original,
    candidate,
    device,
    audit,
    check_stop,
    started,
    budget_seconds,
    shutdown_margin_seconds,
):
    manifest, fit, pool = fit_inputs(
        snapshot, checksums, config, original["identity"]["data_protocol"]
    )
    dates = fit["asof_date"].unique().sort().to_list()
    panel = fixed_dates(fit)
    if original["F_panel"] != [str(day) for day in panel]:
        raise DataContractError("original F panel differs from preregistered dates")
    expected_steps = 2 * math.ceil(len(dates) / config.training.dates_per_optimizer_step)
    if original["global_step"] != expected_steps:
        raise DataContractError("C prefix has incomplete update exposure")
    gradient_dates = [] if candidate == "statistics_linear" else fixed_dates(fit, bins=4, per_bin=2)
    remaining = [day for day in dates if day not in set(panel)]
    chunks = [remaining[i : i + CHUNK_DATES] for i in range(0, len(remaining), CHUNK_DATES)]
    reusable = audit["identity"].get("reused_linear_chunks", {})
    expected_chunks = {f"epoch-{epoch}-{i:04d}" for epoch in (1, 2) for i in range(len(chunks))}
    if not set(reusable.get("chunk_manifests", {})) <= expected_chunks:
        raise DataContractError("reused Linear chunks are outside the fixed F plan")
    audit.update(
        F_dates=len(dates),
        reused_dates_per_epoch=len(panel),
        new_dates_per_epoch=len(remaining),
        gradient_dates=[str(d) for d in gradient_dates],
    )
    cached = FinanceStatisticsDataset(
        cache / "F",
        identity=original["identity"]["data_protocol"],
        windows=tuple(config.model.statistics_windows),
        expected_rows=pool,
    )
    dataset = (
        window_datasets(snapshot, manifest, config, {"F": pool})["F"]
        if (candidate == "finance")
        else cached
    )
    model = (
        build_finance_transformer_model(config, context_length=dataset.context_length)
        if candidate == "finance"
        else FinanceStatisticsRanker(
            candidate,
            input_dim=(len(config.channels) + len(config.market_channels))
            * (5 * len(config.model.statistics_windows) + 1),
            horizons=tuple(config.horizons),
        )
    ).to(device)
    all_gradients = []
    timings = []
    for epoch in (1, 2):
        check_stop()
        state = torch.load(
            source / f"observations/epoch-{epoch}.pt", map_location="cpu", weights_only=False
        )
        if state["identity"] != original["identity"] or state["epoch"] != epoch:
            raise DataContractError("fixed observation weight identity differs")
        model.load_state_dict(state["model_state"], strict=True)
        originals = {
            name: pl.read_parquet(source / f"epoch-{epoch}-F-{name}.parquet")
            for name in ("daily", "predictions", "styles")
        }
        validate_observation(
            originals,
            pool.filter(pl.col("asof_date").is_in(panel)),
            fit.filter(pl.col("asof_date").is_in(panel)),
            config,
        )
        # Check the new gradient path before spending the allocation on full-F scoring.
        for day in gradient_dates:
            check_stop()
            path = output / f"epoch-{epoch}-gradient-{day}.json"
            if path.exists():
                gradient = json.loads(path.read_text())
                if (
                    gradient["epoch"] != epoch
                    or gradient["asof_date"] != str(day)
                    or gradient["optimizer_updates"] != 0
                    or gradient["mode"] != "eval"
                    or gradient["computational_rows"]
                    != pool.filter(pl.col("asof_date") == day).height
                    or gradient["labelled_rows"] != fit.filter(pl.col("asof_date") == day).height
                    or gradient["additivity"]["passed"] is not True
                ):
                    raise DataContractError("committed gradient identity differs")
            else:
                gradient = loss_gradient_diagnostics(
                    model,
                    subset_dates(dataset, [day]),
                    fit.filter(pl.col("asof_date") == day),
                    config,
                    check_stop=check_stop,
                )
                gradient["epoch"] = epoch
                write_json(path, gradient)
                append_progress(
                    output / "progress.jsonl",
                    {
                        "event": "gradient_date_committed",
                        "epoch": epoch,
                        "asof_date": str(day),
                        "additivity": gradient["additivity"],
                        "optimizer_updates": 0,
                    },
                    attempt=len(audit["attempts"]),
                )
            all_gradients.append(gradient)
        parts = [originals]
        for i, days in enumerate(chunks):
            check_stop()
            chunk = output / "chunks" / f"epoch-{epoch}-{i:04d}"
            day_fit = fit.filter(pl.col("asof_date").is_in(days))
            day_pool = pool.filter(pl.col("asof_date").is_in(days))
            if (chunk / "complete.json").exists():
                tables = _read_chunk(chunk)
            elif chunk.name in reusable.get("chunk_manifests", {}):
                previous_chunk = Path(reusable["path"]) / "chunks" / chunk.name
                if (
                    sha256_file(previous_chunk / "complete.json")
                    != (reusable["chunk_manifests"][chunk.name])
                ):
                    raise DataContractError("reused Linear chunk manifest changed")
                tables = _read_chunk(previous_chunk)
                validate_observation(tables, day_pool, day_fit, config)
                _write_chunk(chunk, tables)
                append_progress(
                    output / "progress.jsonl",
                    {
                        "event": "F_chunk_reused",
                        "epoch": epoch,
                        "chunk": i,
                        "dates": len(days),
                        "source": str(previous_chunk),
                        "optimizer_updates": 0,
                    },
                    attempt=len(audit["attempts"]),
                )
            else:
                chunk_started = time.monotonic()
                daily, predictions = score_panel(
                    model,
                    subset_dates(dataset, days),
                    day_fit,
                    config,
                    check_stop=check_stop,
                    precision=audit["identity"]["fixed_forward_precision"],
                )
                tables = {
                    "daily": daily,
                    "predictions": predictions,
                    "styles": statistics_style_exposures(predictions, cached.subset(days), day_fit),
                }
                validate_observation(tables, day_pool, day_fit, config)
                _write_chunk(chunk, tables)
                duration = time.monotonic() - chunk_started
                append_progress(
                    output / "progress.jsonl",
                    {
                        "event": "F_chunk_committed",
                        "epoch": epoch,
                        "chunk": i,
                        "dates": len(days),
                        "last_date": str(days[-1]),
                        "seconds": duration,
                        "optimizer_updates": 0,
                    },
                    attempt=len(audit["attempts"]),
                )
                timings.append((duration, len(days)))
                if len(timings) == 4:
                    # The first chunk also pays CUDA/kernel/cache startup. Use the
                    # next three predetermined chunks, not a favourable timing sample.
                    seconds_per_date = sum(t[0] for t in timings[1:]) / sum(
                        t[1] for t in timings[1:]
                    )
                    unfinished = sum(
                        len(cs)
                        for e in (1, 2)
                        for j, cs in enumerate(chunks)
                        if not (output / "chunks" / f"epoch-{e}-{j:04d}" / "complete.json").exists()
                    )
                    lower_bound = time.monotonic() - started + seconds_per_date * unfinished
                    audit["throughput_measurement"] = {
                        "warmup_chunks": 1,
                        "measured_chunks": 3,
                        "seconds_per_date": seconds_per_date,
                    }
                    audit["forward_only_projected_seconds"] = lower_bound
                    if lower_bound >= budget_seconds - shutdown_margin_seconds:
                        raise TimeoutError("fixed forward projection exceeds remaining budget")
            validate_observation(tables, day_pool, day_fit, config)
            parts.append(tables)
        merged = {
            name: pl.concat([part[name] for part in parts]).sort(
                "asof_date",
                "security_id"
                if name == "predictions"
                else "horizon"
                if name == "daily"
                else "style",
            )
            for name in ("daily", "predictions", "styles")
        }
        validate_observation(merged, pool, fit, config)
        for name, table in merged.items():
            _atomic_parquet(output / f"epoch-{epoch}-F-{name}.parquet", table)
        del merged, parts
        if any(
            not torch.equal(state["model_state"][key], value.detach().cpu())
            for key, value in model.state_dict().items()
        ):
            raise DataContractError("fixed scoring changed model weights/buffers")
    write_json(output / "full-fit-summary.json", _summary(output, dates, all_gradients))
    audit["model_state_unchanged"] = True
