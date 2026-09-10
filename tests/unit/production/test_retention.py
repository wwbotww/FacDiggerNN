from facdigger.production.retention import prune_inference_snapshots


def test_retention_removes_only_old_dated_inference_directories(tmp_path) -> None:
    for day in range(1, 13):
        (tmp_path / f"2026-08-{day:02d}").mkdir()
    unrelated = tmp_path / "manual"
    unrelated.mkdir()

    removed = prune_inference_snapshots(tmp_path, keep_sessions=10)

    assert [path.name for path in removed] == ["2026-08-01", "2026-08-02"]
    assert unrelated.is_dir()


def test_retry_snapshots_are_bounded_without_touching_unknown_directories(tmp_path):
    import json

    root = tmp_path / "2026-08-17"
    root.mkdir()
    for index in range(4):
        child = root / str(index)
        child.mkdir()
        (child / "manifest.json").write_text(json.dumps({
            "contract": "facdigger.inference_snapshot", "status": "complete",
            "snapshot_id": str(index), "config": {"asof_date": "2026-08-17"},
            "created_at": f"2026-08-18T0{index}:00:00+00:00",
        }))
    unknown = root / "manual"
    unknown.mkdir()
    for name, content in (("unreadable", "[]"), ("bad-config", '{"config": []}')):
        directory = root / name
        directory.mkdir()
        (directory / "manifest.json").write_text(content)
    removed = prune_inference_snapshots(tmp_path, keep_sessions=10)
    assert [path.name for path in removed] == ["0", "1"]
    assert unknown.exists() and (root / "2").exists() and (root / "3").exists()
    assert (root / "unreadable").exists() and (root / "bad-config").exists()
