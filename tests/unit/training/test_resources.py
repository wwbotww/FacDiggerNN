from facdigger.training.resources import TrainingResourceBudget, effective_resource_limits


def test_resource_limits_use_visible_slice_and_process_memory(monkeypatch):
    monkeypatch.setattr("facdigger.training.resources.cgroup_memory_limit", lambda: 8 * 1024**3)
    budget = TrainingResourceBudget(
        cuda_peak_reserved_bytes=80 * 1024**3,
        cuda_memory_fraction=0.85,
        host_peak_rss_bytes=20 * 1024**3,
    )
    limits = effective_resource_limits(budget, {"device_memory_bytes": 16 * 1024**3})
    assert limits["cuda_peak_reserved_bytes"] == int(16 * 1024**3 * 0.85)
    assert limits["host_peak_rss_bytes"] == 8 * 1024**3


def test_default_resource_budget_preserves_rtx_admission():
    budget = TrainingResourceBudget()
    assert budget.cuda_peak_reserved_bytes == int(7.2 * 1024**3)
    assert budget.host_peak_rss_bytes == 13 * 1024**3
    assert budget.projected_days == 14
