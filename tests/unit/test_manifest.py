import json
from datetime import datetime, timezone

import yaml

from facdigger.config import ProjectConfig
from facdigger.experiments import manifest as manifest_module


def test_create_run_manifest_writes_resolved_config_and_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        manifest_module,
        "collect_environment",
        lambda include_model_dependencies: {"python": "test"},
    )
    monkeypatch.setattr(
        manifest_module,
        "collect_git_state",
        lambda cwd: {"commit": "abc123", "branch": "main", "dirty": False},
    )
    config = ProjectConfig()
    now = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)

    run_dir, manifest = manifest_module.create_run_manifest(
        config=config,
        output_root=tmp_path,
        repository_root=tmp_path,
        command="facdigger manifest",
        now=now,
    )

    assert run_dir.name.startswith("20260722T120000Z-")
    assert manifest["git"]["commit"] == "abc123"
    assert manifest["source_checkpoint"]["revision"] == config.model.source.revision
    persisted_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    persisted_config = yaml.safe_load(
        (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    assert persisted_manifest["config_hash"] == manifest["config_hash"]
    assert persisted_config["market"]["name"] == "us_equities_daily"


def test_config_hash_is_order_independent() -> None:
    left = manifest_module.sha256_json({"b": 2, "a": 1})
    right = manifest_module.sha256_json({"a": 1, "b": 2})
    assert left == right
