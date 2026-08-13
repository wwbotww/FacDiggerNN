"""FacDiggerNN command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from facdigger.config import load_project_config
from facdigger.environment import collect_environment, environment_is_healthy
from facdigger.models.patchtst_probe import PatchTSTProbeError, run_patchtst_probe

app = typer.Typer(
    name="facdigger",
    help="Point-in-time machine-learning factor research for US equities.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Validate or ingest point-in-time market data.")
dataset_app = typer.Typer(help="Build immutable feature/label dataset snapshots.")
train_app = typer.Typer(help="Train factor-model experiments.")
research_app = typer.Typer(help="Run and freeze walk-forward E0-E3 research.")
release_app = typer.Typer(help="Create and inspect immutable model releases.")
factor_batch_app = typer.Typer(help="Publish and verify cross-project factor deliveries.")
production_app = typer.Typer(help="Run the fail-closed daily EODHD production service.")
app.add_typer(data_app, name="data")
app.add_typer(dataset_app, name="dataset")
app.add_typer(train_app, name="train")
app.add_typer(research_app, name="research")
app.add_typer(release_app, name="release")
app.add_typer(factor_batch_app, name="factor-batch")
app.add_typer(production_app, name="production")


@app.command()
def doctor(
    require_model: Annotated[
        bool,
        typer.Option(help="Require torch, Transformers and Hugging Face Hub to import."),
    ] = True,
) -> None:
    """Inspect Python packages and the available compute device."""

    report = collect_environment(include_model_dependencies=True)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not environment_is_healthy(report, require_model=require_model):
        raise typer.Exit(code=1)


@app.command("probe-patchtst")
def probe_patchtst_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option(help="Directory for the compatibility report")],
    local_files_only: Annotated[
        bool,
        typer.Option(help="Do not access the network; use the local Hugging Face cache only."),
    ] = False,
) -> None:
    """Audit the pinned PatchTST checkpoint and run a train/resume smoke test."""

    project_config = load_project_config(config)
    try:
        report = run_patchtst_probe(project_config, output, local_files_only=local_files_only)
    except PatchTSTProbeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


@data_app.command("validate")
def data_validate_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Validate standardized US-equity Parquet inputs without creating a snapshot."""

    from facdigger.data.adapters import StandardParquetAdapter
    from facdigger.data.config import load_dataset_build_config

    dataset_config = load_dataset_build_config(config)
    adapter = StandardParquetAdapter(dataset_config.sources)
    typer.echo(json.dumps(adapter.audit(), ensure_ascii=False, indent=2, sort_keys=True))


@data_app.command("probe")
def data_probe_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Probe a configured provider and report its live response shape."""

    from facdigger.data.providers.registry import provider_from_config

    try:
        report = provider_from_config(config).probe()
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str))


@data_app.command("ingest")
def data_ingest_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Convert one configured provider into the standard Parquet boundary."""

    from facdigger.data.providers.registry import provider_from_config

    try:
        result = provider_from_config(config).ingest()
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "provider": result.provider,
                "output_dir": str(result.output_dir),
                "files": {name: str(path) for name, path in result.files.items()},
                "warnings": result.manifest["warnings"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@dataset_app.command("build")
def dataset_build_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Build or reuse a content-addressed immutable dataset snapshot."""

    from facdigger.data.config import load_dataset_build_config
    from facdigger.data.snapshots import build_dataset_snapshot

    dataset_config = load_dataset_build_config(config)
    snapshot_dir, manifest = build_dataset_snapshot(dataset_config)
    typer.echo(
        json.dumps(
            {"dataset_id": manifest["dataset_id"], "path": str(snapshot_dir)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@dataset_app.command("build-inference")
def dataset_build_inference_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    release: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            readable=True,
            help="Verified ModelRelease providing the frozen feature contract and scaler.",
        ),
    ],
) -> None:
    """Build a target-free snapshot with the ModelRelease's frozen scaler."""

    from facdigger.data.config import load_inference_snapshot_config
    from facdigger.data.inference_snapshots import build_inference_snapshot

    try:
        snapshot_dir, manifest = build_inference_snapshot(
            load_inference_snapshot_config(config), release
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "snapshot_id": manifest["snapshot_id"],
                "release_id": manifest["feature_contract"]["release_id"],
                "path": str(snapshot_dir),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@train_app.command("e0")
def train_e0_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    dataset: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
) -> None:
    """Train and evaluate an E0 MLP or LightGBM baseline."""

    from facdigger.training.e0 import run_e0
    from facdigger.training.e0_config import load_e0_config

    try:
        run_dir, metrics = run_e0(load_e0_config(config), dataset, repository_root=Path.cwd())
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    raw = metrics["metrics"]["raw"]
    typer.echo(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "dataset_id": metrics["dataset_id"],
                "evaluation_split": metrics["evaluation_split"],
                "coverage": metrics["coverage"]["coverage"],
                "mean_rank_ic": raw["rank_ic"]["mean"],
                "rank_icir": raw["rank_ic"]["ir"],
                "gross_q_high_minus_low": raw["portfolio"].get("gross_q_high_minus_low"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@train_app.command("e1")
def train_e1_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    dataset: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    resume: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, readable=True, help="Resume from last.pt."),
    ] = None,
) -> None:
    """Train and evaluate a randomly initialized PatchTST alpha model."""

    from facdigger.training.e1 import run_e1
    from facdigger.training.e1_config import load_e1_config

    try:
        run_dir, metrics = run_e1(
            load_e1_config(config),
            dataset,
            repository_root=Path.cwd(),
            resume_from=resume,
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    raw = metrics["metrics"]["raw"]
    typer.echo(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "dataset_id": metrics["dataset_id"],
                "evaluation_split": metrics["evaluation_split"],
                "coverage": metrics["coverage"]["coverage"],
                "mean_rank_ic": raw["rank_ic"]["mean"],
                "rank_icir": raw["rank_ic"]["ir"],
                "gross_q_high_minus_low": raw["portfolio"].get("gross_q_high_minus_low"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@train_app.command("e2")
def train_e2_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    dataset: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    resume: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, readable=True, help="Resume from last.pt."),
    ] = None,
) -> None:
    """Train and evaluate an ETTh1-initialized PatchTST alpha model."""

    from facdigger.training.e2 import run_e2
    from facdigger.training.e2_config import load_e2_config

    try:
        run_dir, metrics = run_e2(
            load_e2_config(config),
            dataset,
            repository_root=Path.cwd(),
            resume_from=resume,
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    raw = metrics["metrics"]["raw"]
    typer.echo(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "dataset_id": metrics["dataset_id"],
                "evaluation_split": metrics["evaluation_split"],
                "coverage": metrics["coverage"]["coverage"],
                "mean_rank_ic": raw["rank_ic"]["mean"],
                "rank_icir": raw["rank_ic"]["ir"],
                "gross_q_high_minus_low": raw["portfolio"].get("gross_q_high_minus_low"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@train_app.command("e3")
def train_e3_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    dataset: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    resume: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Resume from pretraining or fine-tuning last.pt.",
        ),
    ] = None,
) -> None:
    """Train E3 with train-only financial masked pretraining."""

    from facdigger.training.e3 import run_e3
    from facdigger.training.e3_config import load_e3_config

    try:
        run_dir, metrics = run_e3(
            load_e3_config(config),
            dataset,
            repository_root=Path.cwd(),
            resume_from=resume,
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    raw = metrics["metrics"]["raw"]
    typer.echo(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "dataset_id": metrics["dataset_id"],
                "evaluation_split": metrics["evaluation_split"],
                "coverage": metrics["coverage"]["coverage"],
                "mean_rank_ic": raw["rank_ic"]["mean"],
                "rank_icir": raw["rank_ic"]["ir"],
                "gross_q_high_minus_low": raw["portfolio"].get("gross_q_high_minus_low"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@app.command("predict")
def predict_command(
    run: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True, help="Completed source run."),
    ],
    split: Annotated[
        str | None,
        typer.Option(help="train, valid or test; defaults to the source run evaluation split."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="New output directory; defaults to <run>/replays/<replay_id>."),
    ] = None,
    dataset: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=False,
            readable=True,
            help="Optional relocated copy of the exact source dataset snapshot.",
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(help="Inference device: cpu, cuda or auto."),
    ] = "cpu",
    unlock_test: Annotated[
        bool,
        typer.Option(help="Explicitly allow reading the test split."),
    ] = False,
    verify_replay: Annotated[
        bool,
        typer.Option(help="Require original-split scores to match source predictions."),
    ] = True,
) -> None:
    """Reload an E0-E3 checkpoint and reproduce research predictions."""

    from facdigger.inference.runner import run_inference

    if split is not None and split not in {"train", "valid", "test"}:
        typer.echo("split must be train, valid or test", err=True)
        raise typer.Exit(code=2)
    if device not in {"cpu", "cuda", "auto"}:
        typer.echo("device must be cpu, cuda or auto", err=True)
        raise typer.Exit(code=2)
    try:
        destination, manifest = run_inference(
            run,
            split=split,
            output_dir=output,
            dataset_dir=dataset,
            device=device,
            unlock_test=unlock_test,
            require_replay_match=verify_replay,
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "output_dir": str(destination),
                "source_run_id": manifest["source_run_id"],
                "split": manifest["split"],
                "rows": manifest["row_count"],
                "coverage": manifest["coverage"]["coverage"],
                "replay_matched": manifest["replay_verification"].get("matched"),
                "predictions": str(destination / "predictions.parquet"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@release_app.command("create")
def release_create_command(
    run: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True, help="Completed source run."),
    ],
    output_root: Annotated[
        Path,
        typer.Option(help="Root for content-addressed immutable model releases."),
    ] = Path("artifacts/releases"),
) -> None:
    """Freeze a clean completed run into an immutable inference release."""

    from facdigger.inference.releases import create_model_release

    try:
        destination, manifest = create_model_release(
            run,
            output_root,
            repository_root=Path.cwd(),
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "release_id": manifest.release_id,
                "model_id": manifest.model_id,
                "model_type": manifest.model_type,
                "path": str(destination),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@release_app.command("verify")
def release_verify_command(
    release: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True, help="Model release directory."),
    ],
) -> None:
    """Verify a release identity and every bound artifact hash."""

    from facdigger.inference.releases import load_model_release

    try:
        manifest = load_model_release(release)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "release_id": manifest.release_id,
                "model_id": manifest.model_id,
                "model_type": manifest.model_type,
                "status": manifest.status,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@factor_batch_app.command("from-predictions")
def factor_batch_from_predictions_command(
    predictions: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ],
    release: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True),
    ],
    output_root: Annotated[
        Path,
        typer.Option(help="Root for content-addressed FactorBatch deliveries."),
    ] = Path("artifacts/factor_batches"),
) -> None:
    """Publish verified E3 evaluation predictions as an isolated replay batch."""

    from facdigger.inference.factor_batch import publish_evaluation_factor_batch

    try:
        destination, manifest = publish_evaluation_factor_batch(
            predictions,
            release,
            output_root,
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "delivery_id": manifest.delivery_id,
                "source_kind": manifest.source.kind,
                "rows": manifest.artifact.row_count,
                "path": str(destination),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@factor_batch_app.command("verify")
def factor_batch_verify_command(
    bundle: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True),
    ],
) -> None:
    """Verify one FactorBatch's semantic identity, files, hashes and coverage."""

    from facdigger.inference.factor_batch import load_factor_batch

    try:
        manifest = load_factor_batch(bundle)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "delivery_id": manifest.delivery_id,
                "source_kind": manifest.source.kind,
                "model_type": manifest.model.model_type,
                "rows": manifest.artifact.row_count,
                "status": manifest.status,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@app.command("signal")
def signal_command(
    release: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True, help="Verified E3 ModelRelease."),
    ],
    dataset: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            readable=True,
            help="Target-free inference snapshot built from this release.",
        ),
    ],
    output_root: Annotated[
        Path,
        typer.Option(help="Root for content-addressed FactorBatch deliveries."),
    ] = Path("artifacts/factor_batches"),
    asof: Annotated[
        str,
        typer.Option(help="latest or one YYYY-MM-DD trading date."),
    ] = "latest",
    device: Annotated[str, typer.Option(help="Inference device: cpu, cuda or auto.")] = "cpu",
) -> None:
    """Generate one strict FactorBatch from an E3 release and inference snapshot."""

    from facdigger.inference.runner import run_signal_inference

    if device not in {"cpu", "cuda", "auto"}:
        typer.echo("device must be cpu, cuda or auto", err=True)
        raise typer.Exit(code=2)
    try:
        destination, signal_manifest = run_signal_inference(
            release,
            output_root=output_root,
            dataset_dir=dataset,
            asof=asof,
            device=device,
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "output_dir": str(destination),
                "delivery_id": signal_manifest["delivery_id"],
                "source_run_id": signal_manifest["source"]["run_id"],
                "rows": signal_manifest["artifact"]["row_count"],
                "minimum_asof_date": signal_manifest["time"]["minimum_asof_date"],
                "maximum_asof_date": signal_manifest["time"]["maximum_asof_date"],
                "factors": str(destination / "factors.parquet"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


@production_app.command("plan")
def production_plan_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Validate configuration and show the current New-York-time target window."""

    from datetime import datetime, timezone

    from facdigger.production.calendar import production_window
    from facdigger.production.config import load_production_config

    try:
        production = load_production_config(config)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    window = production_window(datetime.now(timezone.utc), production.schedule)
    typer.echo(
        json.dumps(
            {
                "target_date": window.target_date.isoformat(),
                "first_attempt_at": window.first_attempt_at.isoformat(),
                "cutoff_at": window.cutoff_at.isoformat(),
                "phase": window.phase,
                "fixed_release_id": production.model.release_id,
                "training_snapshots": "never_modified",
                "inference_retention_sessions": (
                    production.inference.retention_sessions
                ),
                "factor_batch_retention": production.factor_batch.retention,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@production_app.command("bootstrap")
def production_bootstrap_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Create the bounded production store from accepted historical bronze."""

    from facdigger.production.config import load_production_config
    from facdigger.production.lock import ProductionLock
    from facdigger.production.runner import bootstrap_store

    production = load_production_config(config)
    try:
        with ProductionLock(production.state_database):
            current = bootstrap_store(production)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "revision_id": current.revision_id,
                "path": str(current.root),
                "resolved_start": current.manifest["resolved_start"],
                "resolved_end": current.manifest["resolved_end"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@production_app.command("tick")
def production_tick_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Run one idempotent scheduling decision and production transaction."""

    from facdigger.production.config import load_production_config
    from facdigger.production.lock import ProductionLock
    from facdigger.production.runner import run_production_tick

    production = load_production_config(config)
    try:
        with ProductionLock(production.state_database):
            result = run_production_tick(production)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if result.action in {"blocked", "expired"}:
        raise typer.Exit(code=1)


@production_app.command("status")
def production_status_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Inspect persisted target state without touching data or inference outputs."""

    from facdigger.production.config import load_production_config
    from facdigger.production.runner import status_json

    try:
        typer.echo(status_json(load_production_config(config)))
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@production_app.command("health")
def production_health_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Check that the long-running service heartbeat remains fresh."""

    from facdigger.production.config import load_production_config
    from facdigger.production.service import production_health

    try:
        result = production_health(load_production_config(config))
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["healthy"]:
        raise typer.Exit(code=1)


@production_app.command("serve")
def production_serve_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Run the portable container-owned scheduler until SIGTERM or SIGINT."""

    from facdigger.production.config import load_production_config
    from facdigger.production.service import serve_production

    try:
        serve_production(load_production_config(config))
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command("evaluate")
def evaluate_command(
    predictions: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True)
    ],
    dataset: Annotated[
        Path, typer.Option(exists=True, file_okay=False, readable=True)
    ],
    output: Annotated[Path, typer.Option(help="New independent evaluation directory.")],
    costs_bps: Annotated[
        str, typer.Option(help="Comma-separated one-way cost assumptions in basis points.")
    ] = "0,10,20,50",
    minimum_coverage: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="Minimum accepted sample coverage.")
    ] = 1.0,
) -> None:
    """Evaluate an existing prediction table without loading its model."""

    from facdigger.evaluation.runner import evaluate_prediction_file

    try:
        costs = [float(value.strip()) for value in costs_bps.split(",") if value.strip()]
        destination, evaluation_manifest = evaluate_prediction_file(
            predictions,
            dataset,
            output,
            costs_bps=costs,
            minimum_coverage=minimum_coverage,
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "output_dir": str(destination),
                "dataset_id": evaluation_manifest["dataset_id"],
                "evaluation_split": evaluation_manifest["evaluation_split"],
                "rows": evaluation_manifest["row_count"],
                "coverage": evaluation_manifest["coverage"]["coverage"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@app.command("compare")
def compare_command(
    runs: Annotated[
        str,
        typer.Option(help="Comma-separated run directories evaluated on identical samples."),
    ],
    output: Annotated[Path, typer.Option(help="Directory for comparison.json/html")],
) -> None:
    """Compare model runs only after enforcing identical dataset and prediction keys."""

    from facdigger.evaluation.compare import compare_runs

    run_paths = [Path(value.strip()) for value in runs.split(",") if value.strip()]
    try:
        destination, comparison = compare_runs(run_paths, output)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "output_dir": str(destination),
                "dataset_id": comparison["dataset_id"],
                "evaluation_split": comparison["evaluation_split"],
                "runs": comparison["runs"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@research_app.command("plan")
def research_plan_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Validate and display an M6 matrix without building snapshots or training."""

    from facdigger.experiments.manifest import sha256_json
    from facdigger.research.config import load_m6_config
    from facdigger.research.folds import validate_model_config_paths

    research = load_m6_config(config)
    paths = validate_model_config_paths(research)
    typer.echo(
        json.dumps(
            {
                "research_id": research.research_id,
                "config_hash": sha256_json(research.model_dump(mode="json")),
                "folds": [fold.model_dump(mode="json") for fold in research.folds],
                "seeds": research.seeds,
                "models": {key: str(value) for key, value in paths.items()},
                "validation_cells": len(research.folds) * len(research.seeds) * 4,
                "final_holdout_cells": len(research.seeds) * 4,
                "holdout_policy": "separate resume command with explicit unlock",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


@research_app.command("preflight")
def research_preflight_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Run the non-training source and protocol gate for a final M6 experiment."""

    from facdigger.research.config import load_m6_config
    from facdigger.research.preflight import research_preflight

    try:
        report = research_preflight(load_m6_config(config))
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    if not report["ready"]:
        raise typer.Exit(code=1)


@research_app.command("run")
def research_run_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    resume_run: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, readable=True, help="Resume an M6 run."),
    ] = None,
    unlock_final_holdout: Annotated[
        bool,
        typer.Option(
            help="Read the frozen final test split; requires --resume-run after validation."
        ),
    ] = False,
) -> None:
    """Execute or resume the frozen walk-forward research matrix."""

    from facdigger.research.config import load_m6_config
    from facdigger.research.runner import run_m6_research

    try:
        run_dir, manifest = run_m6_research(
            load_m6_config(config),
            repository_root=Path.cwd(),
            resume_run=resume_run,
            unlock_final_holdout=unlock_final_holdout,
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "status": manifest["status"],
                "phase": manifest["phase"],
                "holdout_unlocked": manifest["holdout_unlocked"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
