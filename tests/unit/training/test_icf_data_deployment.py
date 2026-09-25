from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from facdigger.data.config import load_dataset_build_config
from facdigger.data.providers.eodhd.config import load_eodhd_config
from facdigger.research.transformer_config import load_transformer_comparison_config
from facdigger.training.runtime import run_lock

REPOSITORY = Path(__file__).resolve().parents[3]


def module(name):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / "scripts/icf" / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


job = module("data_job")
setup = module("configure_data")


def test_configuration_is_isolated_and_never_overwrites_edits(tmp_path):
    paths = setup.configure(tmp_path, REPOSITORY)
    before = {path: path.read_bytes() for path in paths}
    assert setup.configure(tmp_path, REPOSITORY) == paths
    assert before == {path: path.read_bytes() for path in paths}
    provider = load_eodhd_config(paths[0])
    base = load_dataset_build_config(paths[1])
    research = load_transformer_comparison_config(paths[2])
    assert base.sources.bars == provider.output_dir / "bars_daily.parquet"
    assert research.base_dataset_config == paths[1]
    assert all(path.is_absolute() for path in research.experiments.model_dump().values())
    assert research.snapshot_output_root == base.output_root
    paths[0].write_text(paths[0].read_text().replace("refresh: false", "refresh: true"))
    with pytest.raises(ValueError, match="without overwriting"):
        setup.configure(tmp_path, REPOSITORY)


def credential(tmp_path):
    private = tmp_path / "secrets"
    private.mkdir(mode=0o700)
    path = private / "eodhd.env"
    path.write_text("export EODHD_API_TOKEN='fixture-secret'\n")
    path.chmod(0o600)
    return path


def test_private_credentials_are_parsed_without_shell_execution(tmp_path):
    path = credential(tmp_path)
    assert job.load_credential(path) == "fixture-secret"
    path.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        job.load_credential(path)
    path.chmod(0o600)
    path.parent.chmod(0o755)
    with pytest.raises(ValueError, match="0700"):
        job.load_credential(path)
    path.parent.chmod(0o700)
    link = path.parent / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        job.load_credential(link)
    marker = tmp_path / "must-not-exist"
    path.write_text(f"EODHD_API_TOKEN='fixture'; touch {marker}\n")
    with pytest.raises(ValueError, match="assignment"):
        job.load_credential(path)
    assert not marker.exists()


def test_cpu_job_serializes_downloads_and_preparation(tmp_path, monkeypatch):
    monkeypatch.setenv("FD_ROOT", str(tmp_path))
    monkeypatch.setenv("FD_DATA_MODE", "ingest")
    with run_lock(tmp_path / ".data.lock"), pytest.raises(RuntimeError, match="already"):
        job.run_job()


def test_cpu_plan_loads_credential_only_in_job_and_removes_it_on_failure(tmp_path, monkeypatch):
    paths = setup.configure(tmp_path, REPOSITORY)
    for key, value in {
        "FD_ROOT": str(tmp_path),
        "FD_DATA_MODE": "plan",
        "FD_DATA_CONFIG": str(paths[0]),
        "FD_CREDENTIAL_FILE": str(credential(tmp_path)),
        "FD_RESERVE_API_CALLS": "10000",
        "FD_REQUESTS_PER_MINUTE": "300",
    }.items():
        monkeypatch.setenv(key, value)

    class Provider:
        def __init__(self, config):
            assert os.environ["EODHD_API_TOKEN"] == "fixture-secret"

        def configure_download(self, **kwargs):
            assert kwargs == {"reserve_calls": 10000, "requests_per_minute": 300}

        def plan_historical_download(self, **kwargs):
            raise RuntimeError("fake network unavailable")

    monkeypatch.setattr(job, "EODHDProvider", Provider)
    with pytest.raises(RuntimeError, match="fake network"):
        job.run_job()
    assert "EODHD_API_TOKEN" not in os.environ


def test_submission_has_no_gpu_or_credentials_and_defaults_to_plan(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    sbatch = fake / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\nimport os, sys, json\n"
        "print(json.dumps({'args': sys.argv[1:], 'env': dict(os.environ)}))\n"
    )
    sbatch.chmod(0o755)
    config = tmp_path / "data.env"
    text = (REPOSITORY / "configs/deployment/icf/data.example.env").read_text()
    text = text.replace('FD_ROOT="/home/${USER}/facdigger"', f'FD_ROOT="{tmp_path}"')
    text = text.replace(
        'FD_PYTHON="$FD_CODE_ROOT/.venv/bin/python"', f'FD_PYTHON="{sys.executable}"'
    )
    config.write_text(text)
    setup.configure(tmp_path, REPOSITORY)
    env = {
        **os.environ,
        "PATH": f"{fake}:{os.environ['PATH']}",
        "EODHD_API_TOKEN": "must-not-export",
        "UNRELATED_SECRET": "also-private",
    }
    result = subprocess.run(
        ["bash", str(REPOSITORY / "scripts/icf/submit_data.sh"), str(config), "--test-only"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    submitted = json.loads(result.stdout)
    assert submitted["env"]["FD_DATA_MODE"] == "plan"
    assert "--test-only" in submitted["args"]
    assert "--no-requeue" in submitted["args"]
    assert not any(arg.startswith("--gres") for arg in submitted["args"])
    assert "must-not-export" not in result.stdout and "also-private" not in result.stdout
