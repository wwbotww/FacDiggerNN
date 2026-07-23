"""Resumable M6 matrix runner with a two-step final-holdout gate."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from facdigger.data.snapshots import sha256_file
from facdigger.environment import collect_environment
from facdigger.experiments.manifest import collect_git_state, sha256_json
from facdigger.research.aggregate import (
    MODEL_KEYS,
    aggregate_research_runs,
    write_research_report,
)
from facdigger.research.config import M6ResearchConfig
from facdigger.research.folds import (
    build_walk_forward_snapshots,
    validate_model_config_paths,
)

SnapshotBuilder = Callable[[M6ResearchConfig], list[dict[str, Any]]]
CellExecutor = Callable[..., Path]


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_cell_executor(
    *,
    model_key: str,
    config_path: Path,
    dataset_path: Path,
    seed: int,
    output_root: Path,
    evaluation_split: str,
    unlock_test: bool,
    resume_from: Path | None,
    repository_root: Path,
) -> Path:
    common = {
        "seed": seed,
        "output_root": output_root,
        "evaluation_split": evaluation_split,
        "unlock_test": unlock_test,
    }
    if model_key == "e0":
        from facdigger.training.e0 import run_e0
        from facdigger.training.e0_config import load_e0_config

        config = load_e0_config(config_path).model_copy(update=common)
        run_dir, _ = run_e0(config, dataset_path, repository_root=repository_root)
    elif model_key == "e1":
        from facdigger.training.e1 import run_e1
        from facdigger.training.e1_config import load_e1_config

        config = load_e1_config(config_path).model_copy(update=common)
        run_dir, _ = run_e1(
            config,
            dataset_path,
            repository_root=repository_root,
            resume_from=resume_from,
        )
    elif model_key == "e2":
        from facdigger.training.e2 import run_e2
        from facdigger.training.e2_config import load_e2_config

        config = load_e2_config(config_path).model_copy(update=common)
        run_dir, _ = run_e2(
            config,
            dataset_path,
            repository_root=repository_root,
            resume_from=resume_from,
        )
    elif model_key == "e3":
        from facdigger.training.e3 import run_e3
        from facdigger.training.e3_config import load_e3_config

        config = load_e3_config(config_path).model_copy(update=common)
        run_dir, _ = run_e3(
            config,
            dataset_path,
            repository_root=repository_root,
            resume_from=resume_from,
        )
    else:
        raise ValueError(f"unknown research model key: {model_key}")
    return run_dir.resolve()


def _recoverable_checkpoint(output_root: Path) -> Path | None:
    manifests = sorted(
        output_root.glob("*/manifest.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in manifests:
        manifest = _load_json(path)
        recoverable = manifest.get("recoverable_checkpoint")
        if manifest.get("status") != "failed" or not recoverable:
            continue
        checkpoint = path.parent / str(recoverable)
        if checkpoint.is_file():
            return checkpoint.resolve()
    return None


def _new_research_run(
    config: M6ResearchConfig, repository_root: Path
) -> tuple[Path, dict[str, Any]]:
    created_at = datetime.now(timezone.utc)
    config_payload = config.model_dump(mode="json")
    config_hash = sha256_json(config_payload)
    identity = {
        "config_hash": config_hash,
        "created_at": created_at.isoformat(),
        "nonce": uuid.uuid4().hex,
    }
    run_id = (
        f"{config.research_id}-{created_at.strftime('%Y%m%dT%H%M%SZ')}-"
        f"{sha256_json(identity)[:8]}"
    )
    run_dir = config.output_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "status": "building_snapshots",
        "phase": "validation",
        "run_id": run_id,
        "research_id": config.research_id,
        "created_at": created_at.isoformat(),
        "updated_at": created_at.isoformat(),
        "config_hash": config_hash,
        "repository_root": str(repository_root),
        "holdout_unlocked": False,
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config_payload, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(
        run_dir / "matrix.json",
        {"schema_version": 1, "validation": [], "holdout": []},
    )
    return run_dir, manifest


def _resume_research_run(
    run_dir: Path, config: M6ResearchConfig
) -> tuple[Path, dict[str, Any]]:
    root = run_dir.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"research manifest missing: {manifest_path}")
    manifest = _load_json(manifest_path)
    if manifest["config_hash"] != sha256_json(config.model_dump(mode="json")):
        raise ValueError("research resume configuration does not match")
    if manifest.get("status") == "complete":
        raise ValueError("cannot resume a complete research run")
    return root, manifest


def _cell_identity(
    *, fold_id: str, seed: int, model_key: str, evaluation_split: str
) -> tuple[str, int, str, str]:
    return fold_id, seed, model_key, evaluation_split


def _run_matrix_phase(
    *,
    phase: str,
    fold_plans: list[dict[str, Any]],
    config: M6ResearchConfig,
    model_paths: dict[str, Path],
    run_dir: Path,
    matrix: dict[str, Any],
    executor: CellExecutor,
    repository_root: Path,
    evaluation_split: str,
    unlock_test: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = matrix[phase]
    indexed = {
        _cell_identity(
            fold_id=cell["fold_id"],
            seed=int(cell["seed"]),
            model_key=cell["model_key"],
            evaluation_split=cell["evaluation_split"],
        ): cell
        for cell in records
    }
    for fold in fold_plans:
        for seed in config.seeds:
            for model_key in MODEL_KEYS:
                identity = _cell_identity(
                    fold_id=fold["fold_id"],
                    seed=seed,
                    model_key=model_key,
                    evaluation_split=evaluation_split,
                )
                existing = indexed.get(identity)
                if existing is not None and existing.get("status") == "complete":
                    continue
                output_root = (
                    run_dir
                    / "runs"
                    / phase
                    / fold["fold_id"]
                    / model_key
                    / f"seed-{seed}"
                )
                output_root.mkdir(parents=True, exist_ok=True)
                cell = existing or {
                    "fold_id": fold["fold_id"],
                    "seed": seed,
                    "model_key": model_key,
                    "dataset_id": fold["dataset_id"],
                    "dataset_path": fold["dataset_path"],
                    "evaluation_split": evaluation_split,
                    "status": "pending",
                }
                if existing is None:
                    records.append(cell)
                    indexed[identity] = cell
                resume_from = _recoverable_checkpoint(output_root) if model_key != "e0" else None
                cell.update(
                    {
                        "status": "running",
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "resume_from": str(resume_from) if resume_from else None,
                    }
                )
                _write_json(run_dir / "matrix.json", matrix)
                try:
                    completed_run = executor(
                        model_key=model_key,
                        config_path=model_paths[model_key],
                        dataset_path=Path(fold["dataset_path"]),
                        seed=seed,
                        output_root=output_root,
                        evaluation_split=evaluation_split,
                        unlock_test=unlock_test,
                        resume_from=resume_from,
                        repository_root=repository_root,
                    )
                    cell.update(
                        {
                            "status": "complete",
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                            "run_dir": str(completed_run.resolve()),
                            "metrics_sha256": sha256_file(completed_run / "metrics.json"),
                            "predictions_sha256": sha256_file(
                                completed_run / "predictions.parquet"
                            ),
                            "error": None,
                        }
                    )
                except Exception as exc:
                    cell.update(
                        {
                            "status": "failed",
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                        }
                    )
                    _write_json(run_dir / "matrix.json", matrix)
                    raise
                _write_json(run_dir / "matrix.json", matrix)
    return records


def run_m6_research(
    config: M6ResearchConfig,
    *,
    repository_root: str | Path,
    resume_run: str | Path | None = None,
    unlock_final_holdout: bool = False,
    snapshot_builder: SnapshotBuilder | None = None,
    cell_executor: CellExecutor | None = None,
) -> tuple[Path, dict[str, Any]]:
    repository = Path(repository_root).resolve()
    if unlock_final_holdout and resume_run is None:
        raise ValueError("final holdout requires --resume-run after validation freeze")
    model_paths = validate_model_config_paths(config)
    builder = snapshot_builder or build_walk_forward_snapshots
    executor = cell_executor or _default_cell_executor
    if resume_run is None:
        run_dir, manifest = _new_research_run(config, repository)
        try:
            fold_plans = builder(config)
            _write_json(run_dir / "folds.json", fold_plans)
        except Exception as exc:
            manifest.update(
                {
                    "status": "failed",
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            _write_json(run_dir / "manifest.json", manifest)
            raise
    else:
        run_dir, manifest = _resume_research_run(Path(resume_run), config)
        fold_plans = _load_json(run_dir / "folds.json")
    matrix = _load_json(run_dir / "matrix.json")
    try:
        if manifest["status"] not in {"validation_complete", "holdout_failed"}:
            manifest.update(
                {
                    "status": "running",
                    "phase": "validation",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            _write_json(run_dir / "manifest.json", manifest)
            validation_cells = _run_matrix_phase(
                phase="validation",
                fold_plans=fold_plans,
                config=config,
                model_paths=model_paths,
                run_dir=run_dir,
                matrix=matrix,
                executor=executor,
                repository_root=repository,
                evaluation_split="valid",
                unlock_test=False,
            )
            validation_result = aggregate_research_runs(
                validation_cells, config, evaluation_split="valid"
            )
            validation_dir = run_dir / "validation"
            write_research_report(validation_result, validation_dir)
            validation_matrix_hash = sha256_json(validation_cells)
            freeze = {
                "schema_version": 1,
                "frozen_at": datetime.now(timezone.utc).isoformat(),
                "config_hash": manifest["config_hash"],
                "folds_sha256": sha256_file(run_dir / "folds.json"),
                "validation_matrix_hash": validation_matrix_hash,
                "validation_research_sha256": sha256_file(
                    validation_dir / "research.json"
                ),
                "final_holdout_fold": config.folds[-1].fold_id,
                "holdout_has_been_read": False,
            }
            _write_json(run_dir / "freeze.json", freeze)
            manifest.update(
                {
                    "status": "validation_complete",
                    "phase": "frozen",
                    "validation_completed_at": datetime.now(timezone.utc).isoformat(),
                    "holdout_unlocked": False,
                    "freeze_sha256": sha256_file(run_dir / "freeze.json"),
                }
            )
            _write_json(run_dir / "manifest.json", manifest)

        if not unlock_final_holdout:
            return run_dir, manifest
        freeze = _load_json(run_dir / "freeze.json")
        if freeze["config_hash"] != manifest["config_hash"]:
            raise ValueError("frozen research config hash does not match")
        if freeze["validation_matrix_hash"] != sha256_json(matrix["validation"]):
            raise ValueError("validation matrix changed after research freeze")
        if freeze["validation_research_sha256"] != sha256_file(
            run_dir / "validation" / "research.json"
        ):
            raise ValueError("validation research report changed after freeze")
        final_fold = fold_plans[-1]
        if final_fold["fold_id"] != freeze["final_holdout_fold"]:
            raise ValueError("final fold differs from frozen holdout fold")
        manifest.update(
            {
                "status": "running",
                "phase": "final_holdout",
                "holdout_unlocked": True,
                "holdout_unlocked_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_json(run_dir / "manifest.json", manifest)
        holdout_cells = _run_matrix_phase(
            phase="holdout",
            fold_plans=[final_fold],
            config=config,
            model_paths=model_paths,
            run_dir=run_dir,
            matrix=matrix,
            executor=executor,
            repository_root=repository,
            evaluation_split="test",
            unlock_test=True,
        )
        holdout_result = aggregate_research_runs(
            holdout_cells,
            config,
            evaluation_split="test",
            fold_ids=[final_fold["fold_id"]],
        )
        write_research_report(holdout_result, run_dir / "holdout")
        freeze.update(
            {
                "holdout_has_been_read": True,
                "holdout_read_at": datetime.now(timezone.utc).isoformat(),
                "holdout_matrix_hash": sha256_json(holdout_cells),
                "holdout_research_sha256": sha256_file(
                    run_dir / "holdout" / "research.json"
                ),
            }
        )
        _write_json(run_dir / "freeze.json", freeze)
        manifest.update(
            {
                "status": "complete",
                "phase": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "freeze_sha256": sha256_file(run_dir / "freeze.json"),
                "git": collect_git_state(repository),
                "environment": collect_environment(include_model_dependencies=True),
            }
        )
        _write_json(run_dir / "manifest.json", manifest)
    except Exception as exc:
        phase = str(manifest.get("phase", "validation"))
        manifest.update(
            {
                "status": "holdout_failed" if phase == "final_holdout" else "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        _write_json(run_dir / "manifest.json", manifest)
        raise
    return run_dir, manifest
