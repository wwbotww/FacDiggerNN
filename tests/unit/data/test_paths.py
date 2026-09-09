from pathlib import Path

import pytest

from facdigger.data.contracts import DataContractError
from facdigger.data.paths import artifact_path
from facdigger.inference.source import resolve_training_snapshot


def test_windows_artifact_separator_is_portable(tmp_path):
    expected = tmp_path / "checkpoints" / "best.pt"
    expected.parent.mkdir()
    expected.write_bytes(b"checkpoint")
    assert artifact_path(tmp_path, r"checkpoints\best.pt", "checkpoint") == expected
    assert artifact_path(tmp_path, "checkpoints/best.pt", "checkpoint") == expected


@pytest.mark.parametrize("relative", [
    "../outside.pt", r"..\outside.pt", "/tmp/outside.pt", r"D:\weights\best.pt",
    r"D:best.pt", r"\\server\weights\best.pt", "", ".",
])
def test_foreign_drives_and_path_escapes_are_never_relative_artifacts(tmp_path, relative):
    with pytest.raises(DataContractError, match="relative artifact|escapes"):
        artifact_path(tmp_path, relative, "checkpoint")


def test_artifact_symlinks_cannot_escape_root(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    (root / "best.pt").symlink_to(outside)
    with pytest.raises(DataContractError, match="escapes"):
        artifact_path(root, "best.pt", "checkpoint")


def test_snapshot_relocation_is_explicit_and_does_not_rewrite_source(tmp_path):
    manifest = {"dataset_path": r"D:\missing\snapshots\dataset"}
    with pytest.raises(FileNotFoundError, match="specify --dataset"):
        resolve_training_snapshot(manifest)
    assert resolve_training_snapshot(manifest, tmp_path) == Path(tmp_path)
    assert manifest["dataset_path"] == r"D:\missing\snapshots\dataset"


def test_identity_decoupling_does_not_drop_required_source_provenance(tmp_path):
    from types import SimpleNamespace

    from facdigger.data.config import InferenceSnapshotConfig
    from facdigger.data.inference_snapshots import _require_source_provenance

    (tmp_path / "training.json").write_text(
        '{"artifacts": {"source_manifest": "source_manifest.json"}}'
    )
    release = SimpleNamespace(artifacts={
        "training_dataset_manifest": SimpleNamespace(file="training.json"),
    })
    config = InferenceSnapshotConfig.model_validate({
        "sources": {"bars": "bars.parquet", "universe": "universe.parquet"},
    })
    with pytest.raises(DataContractError, match="requires source provenance"):
        _require_source_provenance(config, tmp_path, release)
