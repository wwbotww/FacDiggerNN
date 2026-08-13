from facdigger.production.retention import prune_inference_snapshots


def test_retention_removes_only_old_dated_inference_directories(tmp_path) -> None:
    for day in range(1, 13):
        (tmp_path / f"2026-08-{day:02d}").mkdir()
    unrelated = tmp_path / "manual"
    unrelated.mkdir()

    removed = prune_inference_snapshots(tmp_path, keep_sessions=10)

    assert [path.name for path in removed] == ["2026-08-01", "2026-08-02"]
    assert unrelated.is_dir()
