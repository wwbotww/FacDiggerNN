"""Resumable, F/S-only preparation by default; no provider calls or model updates."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from facdigger.data.contracts import DataContractError
from facdigger.datasets.finance_statistics import FinanceStatisticsDataset, cache_statistics
from facdigger.research.finance_diagnostics import window_datasets
from facdigger.research.finance_experiment_config import cell_config
from facdigger.research.finance_experiment_plan import (
    clean_git,
    load_fold_inputs,
    read_experiment_plan,
)
from facdigger.training.runtime import (
    TrainingControl,
    TrainingPaused,
    TrainingRuntimeConfig,
    run_lock,
    write_json,
)


def prepare_experiment_cache(
    output: Path, fold_id: str, *, repository_root: Path,
    runtime: TrainingRuntimeConfig,
) -> dict:
    started = time.monotonic()
    plan = read_experiment_plan(output)
    clean_git(repository_root, plan["identity"]["code_commit"])
    phases = ("F", "S")
    if fold_id not in plan["identity"]["folds"]:
        raise DataContractError("fold is not in the frozen experiment plan")
    root = output / "cache" / fold_id
    if not root.resolve().is_relative_to(output.resolve()):
        raise DataContractError("cache path escaped experiment output")
    control = TrainingControl(runtime)
    control.started = started

    def check_stop():
        if control.stop_reason():
            raise TrainingPaused(control.stop_reason(), root / "preparation.json")

    with run_lock(root / ".prepare.lock"):
        binding = {
            "plan_hash": plan["plan_hash"],
            "data_protocol": plan["identity"]["folds"][fold_id]["data_protocol"],
            "windows": plan["identity"]["supervised_config"]["model"]["statistics_windows"],
        }
        audit_path = root / "preparation.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            if audit["identity"] != binding:
                raise DataContractError("cache preparation identity differs")
        else:
            if set(p.name for p in root.iterdir()) - {".prepare.lock"}:
                raise DataContractError("unowned cache output is not empty")
            audit = {"identity": binding, "completed_phases": []}
        audit["status"] = "running"
        audit.pop("error", None)
        audit.pop("stop_reason", None)
        write_json(audit_path, audit)
        try:
            control.__enter__()
            check_stop()
            seed = plan["identity"]["research"]["seeds"][0]
            config = cell_config(plan["identity"]["supervised_config"], seed)
            manifest, protocol, _, pools = load_fold_inputs(
                plan, fold_id, seed=seed, phases=phases,
            )
            check_stop()
            datasets = None
            for phase in phases:
                path = root / phase
                if not path.exists():
                    pending = root / f".{phase}.pending"
                    # Only our bound, unfinished staging directory can be discarded.
                    if pending.exists():
                        if pending.is_symlink():
                            raise DataContractError("cache staging must not be a symlink")
                        shutil.rmtree(pending)
                    if datasets is None:
                        datasets = window_datasets(
                            Path(plan["identity"]["folds"][fold_id]["path"]),
                            manifest, config, pools,
                        )
                    cache_statistics(
                        datasets[phase], pending, windows=tuple(binding["windows"]),
                        identity=protocol, check_stop=check_stop,
                    )
                    check_stop()
                    pending.replace(path)
                FinanceStatisticsDataset(
                    path, identity=protocol, windows=tuple(binding["windows"]),
                    expected_rows=pools[phase],
                )
                if phase not in audit["completed_phases"]:
                    audit["completed_phases"].append(phase)
                write_json(audit_path, audit)
            audit["status"] = "complete"
        except TrainingPaused as exc:
            audit.update(status="paused", stop_reason=exc.reason)
        except Exception as exc:
            audit.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            control.__exit__(None, None, None)
            write_json(audit_path, audit)
        return audit
