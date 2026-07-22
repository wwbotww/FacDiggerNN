"""FacDiggerNN command-line interface."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Annotated

import typer

from facdigger.config import load_project_config
from facdigger.environment import collect_environment, environment_is_healthy
from facdigger.experiments.manifest import create_run_manifest
from facdigger.models.patchtst_probe import PatchTSTProbeError, run_patchtst_probe

app = typer.Typer(
    name="facdigger",
    help="Point-in-time machine-learning factor research for US equities.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Validate or ingest point-in-time market data.")
dataset_app = typer.Typer(help="Build immutable feature/label dataset snapshots.")
train_app = typer.Typer(help="Train factor-model experiments.")
app.add_typer(data_app, name="data")
app.add_typer(dataset_app, name="dataset")
app.add_typer(train_app, name="train")


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


@app.command("manifest")
def manifest_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option(help="Root directory for generated run artifacts")],
) -> None:
    """Resolve configuration and create a reproducible run manifest."""

    project_config = load_project_config(config)
    command = shlex.join(
        ["facdigger", "manifest", "--config", str(config), "--output", str(output)]
    )
    run_dir, _ = create_run_manifest(
        config=project_config,
        output_root=output,
        repository_root=Path.cwd(),
        command=command,
    )
    typer.echo(str(run_dir))


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
