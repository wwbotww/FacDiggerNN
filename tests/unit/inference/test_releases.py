from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest
import yaml

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.experiments.manifest import sha256_json
from facdigger.inference.factor_batch import (
    factor_batch_metadata,
    load_factor_batch,
    publish_evaluation_factor_batch,
)
from facdigger.inference.releases import create_model_release, load_model_release

torch = pytest.importorskip("torch")


def _source_run(tmp_path: Path, *, commit: str = "1" * 40) -> tuple[Path, Path]:
    dataset = tmp_path / "snapshot"
    dataset.mkdir()
    scaler = {"channels": {"r_close": {"median": 0.0, "scale": 1.0}}}
    (dataset / "scaler.json").write_text(json.dumps(scaler), encoding="utf-8")
    dataset_manifest = {
        "schema_version": 3,
        "dataset_id": "dataset-1",
        "config": {
            "features": {
                "name": "price_volume_v1",
                "channels": ["r_close"],
                "context_length": 20,
                "scaler": "train_global_robust",
            },
            "label": {"horizon": 5},
        },
    }
    (dataset / "manifest.json").write_text(
        json.dumps(dataset_manifest), encoding="utf-8"
    )

    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints" / "best.pt"
    torch.save(
        {
            "schema_version": 3,
            "experiment_family": "e2",
            "objective": "cross_sectional_rank_correlation_surrogate_v2_full_date",
            "target_transform": "full_date_average_rank_percentile_minus_one_to_one",
            "optimization_protocol": {
                "unit": "complete_date",
                "method": "exact_two_pass_full_date_pearson",
            },
            "dataset_id": "dataset-1",
        },
        checkpoint,
    )
    config = {"experiment_id": "e3-release-test", "channels": ["r_close"]}
    (run / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    predictions_path = run / "predictions.parquet"
    pl.DataFrame(
        {
            "security_id": ["security-1", "security-2"],
            "symbol": ["AAA", "BBB"],
            "asof_date": [datetime(2026, 8, 10).date(), datetime(2026, 8, 10).date()],
            "score_raw": [0.4, -0.2],
            "score_neutralized": [None, None],
            "target": [0.01, -0.01],
            "split": ["valid", "valid"],
            "model_id": ["e3-release-test", "e3-release-test"],
            "checkpoint_hash": [sha256_file(checkpoint), sha256_file(checkpoint)],
            "dataset_id": ["dataset-1", "dataset-1"],
            "eligible": [True, True],
            "industry_code": [None, None],
            "log_float_market_cap": [None, None],
        },
        schema_overrides={
            "asof_date": pl.Date,
            "score_raw": pl.Float64,
            "score_neutralized": pl.Float64,
            "target": pl.Float64,
            "industry_code": pl.String,
            "log_float_market_cap": pl.Float64,
        },
    ).write_parquet(predictions_path)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "run_id": "e3-release-test-run",
        "model_id": "e3-release-test",
        "model_type": "financial_pretrained_patchtst",
        "dataset_id": "dataset-1",
        "dataset_path": str(dataset),
        "dataset_manifest_hash": sha256_file(dataset / "manifest.json"),
        "config_hash": sha256_json(config),
        "checkpoint": {
            "file": "checkpoints/best.pt",
            "sha256": sha256_file(checkpoint),
        },
        "input": {
            "feature_set": "price_volume_v1",
            "context_length": 20,
            "channels": ["r_close"],
            "feature_scaler": "train_global_robust",
            "feature_scaler_sha256": sha256_file(dataset / "scaler.json"),
        },
        "finetuning": {
            "objective": "cross_sectional_rank_correlation_surrogate_v2_full_date",
            "target_transform": "full_date_average_rank_percentile_minus_one_to_one",
        },
        "predictions_sha256": sha256_file(predictions_path),
        "artifacts": {"predictions": "predictions.parquet"},
        "git": {"commit": commit, "branch": "test", "dirty": False},
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run, dataset


def _write_source_provenance(dataset: Path, *, warnings: list[str]) -> None:
    (dataset / "source_manifest.json").write_text(
        json.dumps({"provider": "eodhd", "warnings": warnings}), encoding="utf-8"
    )


def test_model_release_binds_checkpoint_config_scaler_and_lineage(
    tmp_path, monkeypatch
) -> None:
    commit = "1" * 40
    run, _ = _source_run(tmp_path, commit=commit)
    monkeypatch.setattr(
        "facdigger.inference.releases.collect_git_state",
        lambda _: {
            "commit": commit,
            "branch": "test",
            "dirty": False,
            "status_porcelain": "",
        },
    )

    release, manifest = create_model_release(
        run,
        tmp_path / "releases",
        repository_root=tmp_path,
        created_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
    )
    repeated, repeated_manifest = create_model_release(
        run,
        tmp_path / "releases",
        repository_root=tmp_path,
        created_at=datetime(2026, 8, 13, tzinfo=timezone.utc),
    )

    assert release == repeated
    assert release.name == manifest.release_id == repeated_manifest.release_id
    assert "schema_version" not in json.loads(
        (release / "manifest.json").read_text(encoding="utf-8")
    )
    assert "checkpoint_protocol" not in json.loads(
        (release / "manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest.artifacts) == {
        "checkpoint",
        "resolved_config",
        "scaler",
        "training_dataset_manifest",
        "source_run_manifest",
        "checkpoint_protocol",
    }
    assert manifest.training_data.dataset_id == "dataset-1"
    assert manifest.feature_contract.scaler_contract == "train_global_robust"
    assert load_model_release(release).release_id == manifest.release_id
    source, model = factor_batch_metadata(manifest, source_kind="signal_inference")
    assert source.run_manifest_sha256 == manifest.source.run_manifest_sha256
    assert model.release_id == manifest.release_id
    assert model.checkpoint_sha256 == manifest.artifacts["checkpoint"].sha256


def test_verified_e3_predictions_publish_through_the_factor_batch_contract(
    tmp_path, monkeypatch
) -> None:
    commit = "1" * 40
    run, _ = _source_run(tmp_path, commit=commit)
    monkeypatch.setattr(
        "facdigger.inference.releases.collect_git_state",
        lambda _: {
            "commit": commit,
            "branch": "test",
            "dirty": False,
            "status_porcelain": "",
        },
    )
    release_dir, release = create_model_release(
        run, tmp_path / "releases", repository_root=tmp_path
    )
    predictions_path = tmp_path / "predictions.parquet"
    predictions_path.write_bytes((run / "predictions.parquet").read_bytes())

    bundle, manifest = publish_evaluation_factor_batch(
        predictions_path, release_dir, tmp_path / "factor-batches"
    )

    factors = pl.read_parquet(bundle / "factors.parquet")
    assert factors.columns == [
        "security_id",
        "symbol",
        "asof_date",
        "score",
        "eligible",
    ]
    assert "target" not in factors.columns
    assert manifest.source.kind == "evaluation_predictions"
    assert manifest.input.universe_semantics == "eligible_scored_cross_section"
    assert manifest.model.release_id == release.release_id
    assert load_factor_batch(bundle).delivery_id == manifest.delivery_id

    predictions = pl.read_parquet(predictions_path).with_columns(
        pl.lit("0" * 64).alias("checkpoint_hash")
    )
    predictions.write_parquet(predictions_path)
    with pytest.raises(DataContractError, match="differs from the artifact"):
        publish_evaluation_factor_batch(
            predictions_path, release_dir, tmp_path / "rejected"
        )


def test_model_release_rejects_uncommitted_source_run_but_allows_later_clean_commit(
    tmp_path, monkeypatch
) -> None:
    run, _ = _source_run(tmp_path, commit="1" * 40)
    monkeypatch.setattr(
        "facdigger.inference.releases.collect_git_state",
        lambda _: {
            "commit": "2" * 40,
            "branch": "test",
            "dirty": False,
            "status_porcelain": "",
        },
    )
    _, release = create_model_release(
        run, tmp_path / "releases", repository_root=tmp_path
    )
    assert release.source.commit == "1" * 40

    monkeypatch.setattr(
        "facdigger.inference.releases.collect_git_state",
        lambda _: {
            "commit": "2" * 40,
            "branch": "test",
            "dirty": True,
            "status_porcelain": "publisher change",
        },
    )
    with pytest.raises(DataContractError, match="publisher requires a clean"):
        create_model_release(run, tmp_path / "dirty-publisher", repository_root=tmp_path)

    monkeypatch.setattr(
        "facdigger.inference.releases.collect_git_state",
        lambda _: {
            "commit": "2" * 40,
            "branch": "test",
            "dirty": False,
            "status_porcelain": "",
        },
    )
    run_manifest_path = run / "manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["git"]["dirty"] = True
    run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
    with pytest.raises(DataContractError, match="source run.*clean Git worktree"):
        create_model_release(run, tmp_path / "rejected", repository_root=tmp_path)


def test_model_release_detects_artifact_tampering(tmp_path, monkeypatch) -> None:
    commit = "1" * 40
    run, _ = _source_run(tmp_path, commit=commit)
    monkeypatch.setattr(
        "facdigger.inference.releases.collect_git_state",
        lambda _: {
            "commit": commit,
            "branch": "test",
            "dirty": False,
            "status_porcelain": "",
        },
    )
    release, _ = create_model_release(
        run, tmp_path / "releases", repository_root=tmp_path
    )
    (release / "scaler.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DataContractError, match="artifact integrity failure: scaler"):
        load_model_release(release)


def test_model_release_rejects_undeclared_directory(tmp_path, monkeypatch) -> None:
    commit = "1" * 40
    run, _ = _source_run(tmp_path, commit=commit)
    monkeypatch.setattr(
        "facdigger.inference.releases.collect_git_state",
        lambda _: {
            "commit": commit,
            "branch": "test",
            "dirty": False,
            "status_porcelain": "",
        },
    )
    release, _ = create_model_release(
        run, tmp_path / "releases", repository_root=tmp_path
    )
    (release / "undeclared").mkdir()

    with pytest.raises(DataContractError, match="undeclared or missing entries"):
        load_model_release(release)


def test_model_release_rejects_pre_full_date_patchtst_checkpoint(
    tmp_path, monkeypatch
) -> None:
    commit = "1" * 40
    run, _ = _source_run(tmp_path, commit=commit)
    checkpoint = run / "checkpoints" / "best.pt"
    torch.save(
        {
            "schema_version": 2,
            "experiment_family": "e2",
            "objective": "cross_sectional_rank_correlation_surrogate_v1",
            "dataset_id": "dataset-1",
        },
        checkpoint,
    )
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint"]["sha256"] = sha256_file(checkpoint)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "facdigger.inference.releases.collect_git_state",
        lambda _: {
            "commit": commit,
            "branch": "test",
            "dirty": False,
            "status_porcelain": "",
        },
    )

    with pytest.raises(DataContractError, match="checkpoint schema 3"):
        create_model_release(run, tmp_path / "releases", repository_root=tmp_path)


def test_model_release_rejects_unbound_or_modified_scaler(tmp_path, monkeypatch) -> None:
    commit = "1" * 40
    run, dataset = _source_run(tmp_path, commit=commit)
    monkeypatch.setattr(
        "facdigger.inference.releases.collect_git_state",
        lambda _: {
            "commit": commit,
            "branch": "test",
            "dirty": False,
            "status_porcelain": "",
        },
    )
    (dataset / "scaler.json").write_text('{"changed": true}', encoding="utf-8")

    with pytest.raises(DataContractError, match="does not bind the current training scaler"):
        create_model_release(run, tmp_path / "releases", repository_root=tmp_path)


def test_model_release_rejects_unbound_or_modified_predictions(
    tmp_path, monkeypatch
) -> None:
    commit = "1" * 40
    run, _ = _source_run(tmp_path, commit=commit)
    monkeypatch.setattr(
        "facdigger.inference.releases.collect_git_state",
        lambda _: {
            "commit": commit,
            "branch": "test",
            "dirty": False,
            "status_porcelain": "",
        },
    )
    predictions_path = run / "predictions.parquet"
    pl.read_parquet(predictions_path).with_columns(
        (pl.col("score_raw") + 1.0).alias("score_raw")
    ).write_parquet(predictions_path)

    with pytest.raises(DataContractError, match="does not bind its predictions"):
        create_model_release(run, tmp_path / "releases", repository_root=tmp_path)


def test_model_release_rejects_unstable_eodhd_identity(tmp_path, monkeypatch) -> None:
    commit = "1" * 40
    run, dataset = _source_run(tmp_path, commit=commit)
    _write_source_provenance(
        dataset,
        warnings=["security_id uses provider-symbol fallback for: UNKNOWN.US"],
    )
    monkeypatch.setattr(
        "facdigger.inference.releases.collect_git_state",
        lambda _: {
            "commit": commit,
            "branch": "test",
            "dirty": False,
            "status_porcelain": "",
        },
    )

    with pytest.raises(DataContractError, match="provider-symbol fallback"):
        create_model_release(run, tmp_path / "releases", repository_root=tmp_path)


@pytest.mark.parametrize(
    ("security_id", "expected_policy", "error"),
    [
        ("eodhd:isin:US0000000001", "eodhd_isin_only", None),
        ("eodhd:symbol:UNKNOWN.US", None, "provider-symbol fallback"),
    ],
)
def test_model_release_verifies_eodhd_identity_in_training_features(
    tmp_path, monkeypatch, security_id, expected_policy, error
) -> None:
    commit = "1" * 40
    run, dataset = _source_run(tmp_path, commit=commit)
    _write_source_provenance(dataset, warnings=[])
    pl.DataFrame(
        {
            "security_id": [security_id],
            "asof_date": [datetime(2026, 8, 12).date()],
            "r_close": [0.0],
        }
    ).write_parquet(dataset / "features.parquet")
    dataset_manifest_path = dataset / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    dataset_manifest["artifacts"] = {"features": "features.parquet"}
    dataset_manifest_path.write_text(json.dumps(dataset_manifest), encoding="utf-8")
    run_manifest_path = run / "manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["dataset_manifest_hash"] = sha256_file(dataset_manifest_path)
    run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
    monkeypatch.setattr(
        "facdigger.inference.releases.collect_git_state",
        lambda _: {
            "commit": commit,
            "branch": "test",
            "dirty": False,
            "status_porcelain": "",
        },
    )

    if error is not None:
        with pytest.raises(DataContractError, match=error):
            create_model_release(run, tmp_path / "releases", repository_root=tmp_path)
        return
    _, manifest = create_model_release(
        run, tmp_path / "releases", repository_root=tmp_path
    )
    assert manifest.feature_contract.identity_policy == expected_policy
