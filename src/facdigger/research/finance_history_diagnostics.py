"""Read-only R experiment, admitted only by strict historical prediction replay."""

from __future__ import annotations

import json
import time
from pathlib import Path, PureWindowsPath
from typing import Any

import polars as pl
import torch

from facdigger.data.contracts import DataContractError
from facdigger.data.paths import artifact_path
from facdigger.data.snapshots import sha256_file
from facdigger.datasets.splits import split_supervised_training_index
from facdigger.environment import collect_environment
from facdigger.inference.backends import load_checkpoint_backend
from facdigger.inference.runner import _validate_dataset, _verify_replay
from facdigger.inference.source import _load_source_run
from facdigger.research.finance_diagnostics import (
    computation_rows,
    diagnostic_daily,
    evaluate_diagnostic_validation,
    fixed_dates,
    score_panel,
    window_datasets,
)
from facdigger.training.common import load_snapshot_inference_rows
from facdigger.training.finance_transformer_config import FinanceTransformerExperimentConfig
from facdigger.training.runtime import write_json


def run_history_diagnostics(
    matrix_dir: Path,
    snapshots: Path,
    output: Path,
    *,
    device: str = "cuda",
    budget_seconds: float = 4 * 3600,
) -> dict[str, Any]:
    if not 0 < budget_seconds <= 4 * 3600:
        raise ValueError("R budget must be positive and at most four allocation hours")
    for source in (matrix_dir, snapshots):
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError("diagnostic output must be outside historical inputs")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    audit: dict[str, Any] = {
        "experiment": "R",
        "status": "running",
        "budget_seconds": budget_seconds,
        "protocol": "historical; target-filtered C, contaminated selection, old calendar",
        "environment": collect_environment(),
        "sources": {},
        "replay": {},
        "holdout_used": False,
        "source_mutations": False,
    }

    def flush() -> None:
        audit["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "audit.json", audit)

    def check_stop() -> None:
        if time.monotonic() - started >= budget_seconds - 60:
            raise TimeoutError("R allocation budget reached; no automatic extension")

    try:
        cells = json.loads((matrix_dir / "matrix.json").read_text())["stages"]
        resolved = []
        for stage in ("scratch", "finance_pretrained"):
            matches = [c for c in cells if c["fold_id"] == "wf3" and c["stage"] == stage]
            if len(matches) != 1 or matches[0]["status"] != "complete":
                raise DataContractError("R requires one matrix-authoritative complete wf3 run")
            run = matrix_dir / "runs" / "wf3" / stage / PureWindowsPath(matches[0]["run_dir"]).name
            manifest, payload, best = _load_source_run(run)
            if "data_protocol" in manifest or manifest["evaluation_split"] != "valid":
                raise DataContractError("R expects the original legacy validation run")
            snapshot = snapshots / manifest["dataset_id"]
            sm, frames = _validate_dataset(manifest, snapshot)
            if sha256_file(run / "predictions.parquet") != manifest["predictions_sha256"]:
                raise DataContractError("source prediction hash differs from manifest")
            config = FinanceTransformerExperimentConfig.model_validate(payload)
            protocol_rows, split_audit = split_supervised_training_index(
                frames["sample_index"],
                selection_fraction=config.selection_fraction,
            )
            saved_split = manifest["supervised_selection_audit"]
            if json.loads(json.dumps(split_audit, default=str)) != saved_split:
                raise DataContractError("reconstructed historical selection differs")
            labelled = {
                "F": protocol_rows.filter(pl.col("split") == "train_fit"),
                "S": protocol_rows.filter(pl.col("split") == "inner_selection"),
                "V": frames["sample_index"].filter(pl.col("split") == "valid"),
            }
            panels = {
                "F": fixed_dates(labelled["F"]),
                "replay": fixed_dates(labelled["V"], bins=1, per_bin=20),
            }
            labelled["F"] = labelled["F"].filter(pl.col("asof_date").is_in(panels["F"]))
            labelled["replay"] = labelled["V"].filter(pl.col("asof_date").is_in(panels["replay"]))
            pools = {k: computation_rows(v) for k, v in labelled.items()}
            datasets = window_datasets(snapshot, sm, config, {"replay": pools["replay"]})
            paths = {
                "best": best,
                "last": artifact_path(
                    run,
                    manifest["checkpoint"]["last_file"],
                    "last checkpoint",
                ),
            }
            if sha256_file(paths["last"]) != manifest["checkpoint"]["last_sha256"]:
                raise DataContractError("source last checkpoint hash differs from manifest")
            states = {}
            for name, path in paths.items():
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                if (
                    checkpoint["dataset_id"] != manifest["dataset_id"]
                    or checkpoint["config"] != payload
                ):
                    raise DataContractError("historical checkpoint config/data identity differs")
                states[name] = checkpoint["model_state"]
            audit["sources"][stage] = {
                "run": str(run),
                "manifest_sha256": sha256_file(run / "manifest.json"),
                "checkpoints": {k: sha256_file(v) for k, v in paths.items()},
                "panels": panels,
                "split": saved_split,
            }
            backend = load_checkpoint_backend(
                "finance_patch_transformer",
                config_payload=payload,
                checkpoint_path=best,
                training_dataset_id=manifest["dataset_id"],
                context_length=datasets["replay"].context_length,
                device=device,
            )
            model, _, _ = backend.model_runtime()
            t0 = time.monotonic()
            _, replay = score_panel(
                model, datasets["replay"], labelled["replay"], config, check_stop=check_stop
            )
            replay = replay.join(
                labelled["replay"].select("security_id", "asof_date", "target"),
                on=["security_id", "asof_date"],
                validate="1:1",
            ).rename({f"score_{config.primary_horizon}": "score_raw"})
            replay_audit = _verify_replay(
                run,
                manifest,
                "valid",
                replay,
                require_match=False,
                asof_dates=panels["replay"],
            )
            replay_audit["seconds"] = time.monotonic() - t0
            audit["replay"][stage] = replay_audit
            flush()
            if not replay_audit["matched"]:
                audit.update(
                    status="blocked_strict_replay",
                    reason=(
                        "Original scores not reproduced at rtol=1e-6/atol=1e-8; "
                        "no dependent rescoring, tolerance relaxation or legacy retraining. "
                        "C may proceed independently."
                    ),
                )
                flush()
                return audit
            resolved.append((stage, run, config, model, pools, labelled, snapshot, sm, states))

        # Both original best checkpoints must pass before any interpretive rescoring.
        predicted_seconds = sum(
            a["seconds"] / 20 * (2 * (60 + 374) + 462 + 40) for a in audit["replay"].values()
        )
        audit["estimated_rescoring_seconds"] = predicted_seconds
        if predicted_seconds + time.monotonic() - started > budget_seconds - 60:
            raise TimeoutError("replay throughput projects beyond R budget; scope unchanged")
        for stage, run, config, model, pools, labelled, snapshot, sm, states in resolved:
            datasets = window_datasets(snapshot, sm, config, pools)
            target = output / stage
            target.mkdir()
            original = pl.read_parquet(run / "predictions.parquet").sort("asof_date", "security_id")
            support = labelled["V"].sort("asof_date", "security_id")
            if not original.select("security_id", "asof_date", "target").equals(
                support.select("security_id", "asof_date", "target")
            ):
                raise DataContractError("saved V predictions have different support/targets")
            diagnostic_daily(
                original["score_raw"].to_numpy()[:, None],
                computation_rows(support),
                support,
                config,
                horizons=(config.primary_horizon,),
            ).write_parquet(target / "best-V-daily.parquet")
            same = all(torch.equal(v, states["last"][k]) for k, v in states["best"].items())
            for name in ("best", "last"):
                if name == "last" and same:
                    audit["sources"][stage]["last_alias"] = "best; identical state tensors"
                    continue
                model.load_state_dict(states[name], strict=True)
                for phase in ("F", "S") if name == "best" else ("F", "S", "V"):
                    daily, predictions = score_panel(
                        model,
                        datasets[phase],
                        labelled[phase],
                        config,
                        check_stop=check_stop,
                    )
                    daily.write_parquet(target / f"{name}-{phase}-daily.parquet")
                    predictions.write_parquet(target / f"{name}-{phase}-predictions.parquet")
                    if phase == "V":
                        check_stop()
                        evaluate_diagnostic_validation(
                            snapshot,
                            sm,
                            predictions,
                            labelled["V"],
                            config,
                            model_id=f"historical-diagnostic-{stage}-{name}",
                            checkpoint_hash=audit["sources"][stage]["checkpoints"][name],
                            destination=target / f"{name}-V-evaluation",
                        )

            # Predetermined by support differences alone; fixed weights and same L/targets.
            full = load_snapshot_inference_rows(
                snapshot, sm, asof_dates=labelled["V"]["asof_date"].unique().sort().to_list()
            )
            counts = (
                full.group_by("asof_date")
                .len()
                .join(
                    labelled["V"].group_by("asof_date").len(),
                    on="asof_date",
                    suffix="_L",
                )
                .filter(pl.col("len") != pl.col("len_L"))
            )
            if counts.height:
                dates = fixed_dates(counts, bins=1, per_bin=min(20, counts.height))
                cf_label = labelled["V"].filter(pl.col("asof_date").is_in(dates))
                cf = window_datasets(
                    snapshot,
                    sm,
                    config,
                    {
                        "legacy": computation_rows(cf_label),
                        "full": full.filter(pl.col("asof_date").is_in(dates)),
                    },
                )
                audit["sources"][stage]["counterfactual_dates"] = dates
                for name in ("best",) if same else ("best", "last"):
                    model.load_state_dict(states[name], strict=True)
                    for pool, dataset in cf.items():
                        daily, predictions = score_panel(
                            model, dataset, cf_label, config, check_stop=check_stop
                        )
                        daily.write_parquet(target / f"{name}-sensitivity-{pool}-daily.parquet")
                        predictions.write_parquet(
                            target / f"{name}-sensitivity-{pool}-predictions.parquet"
                        )
            flush()
        audit["status"] = "complete"
    except TimeoutError as exc:
        audit.update(status="paused_budget", reason=str(exc))
    except Exception as exc:
        audit.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        flush()
    return audit
