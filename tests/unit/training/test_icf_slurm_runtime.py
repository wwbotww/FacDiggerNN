from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/icf/slurm_runtime.py"
spec = importlib.util.spec_from_file_location("icf_slurm_runtime", SCRIPT)
slurm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slurm)


@pytest.fixture(autouse=True)
def clean_slurm_environment(monkeypatch):
    for name in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(slurm.time, "monotonic", lambda: 100.0)


@pytest.mark.parametrize(
    "value,seconds",
    [("04:00:00", 14400), ("2-00:00:00", 172800), ("05:30", 330), ("00:00", 0)],
)
def test_slurm_remaining_time(value, seconds):
    assert slurm.parse_time_left(value) == seconds


@pytest.mark.parametrize(
    "value", ["UNLIMITED", "NOT_SET", "", "INVALID", "0:90", "1:00\n2:00", "-00:01"]
)
def test_unknown_time_is_not_an_unlimited_training_budget(value):
    with pytest.raises(ValueError):
        slurm.parse_time_left(value)


@pytest.mark.parametrize(
    "raw,array,index,expected",
    [("42", None, None, "42"), ("43", "42", "0", "42_0"),
     ("44", "42", "1", "42_1"), ("42", "42", "58", "42_58")],
)
def test_current_task_query_checks_both_raw_and_array_identity(
    monkeypatch, raw, array, index, expected,
):
    monkeypatch.setenv("SLURM_JOB_ID", raw)
    if array is not None:
        monkeypatch.setenv("SLURM_ARRAY_JOB_ID", array)
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", index)
    calls = []

    def query(args, **kwargs):
        calls.append(args)
        assert kwargs["check"] and kwargs["capture_output"] and kwargs["text"]
        assert kwargs["timeout"] == 15
        # The placeholder's bare raw ID selects siblings as well. This fixture
        # reproduces the original failure unless the caller selects its element.
        if array is not None and args[args.index("-j") + 1] == array:
            return SimpleNamespace(stdout=f"43|{array}_0|01:00\n{raw}|{expected}|02:00\n")
        return SimpleNamespace(stdout=f"{raw}|{expected}|02:00\n")

    monkeypatch.setattr(slurm.subprocess, "run", query)
    assert slurm.current_task_ref() == expected
    assert slurm.remaining_seconds(raw) == 120
    assert calls == [["squeue", "-h", "-r", "-j", expected, "-o", "%A|%i|%L"]]


@pytest.mark.parametrize(
    "raw,array,index",
    [(None, None, None), ("0", None, None), ("42_0", None, None),
     ("42", "42", None), ("42", None, "0"), ("42", "", "0"),
     ("42", "42", ""), ("42", "42", "-1"), ("42", "42", "0-3"),
     ("42", "42,43", "1")],
)
def test_incomplete_or_ambiguous_environment_never_queries_slurm(monkeypatch, raw, array, index):
    for name, value in zip(
        ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"),
        (raw, array, index), strict=True,
    ):
        if value is not None:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(slurm.subprocess, "run", lambda *a, **k: pytest.fail("must not query"))
    with pytest.raises(ValueError):
        slurm.remaining_seconds(raw or "42")


@pytest.mark.parametrize(
    "response",
    ["", "\n", "42|42|01:00\n42|42|02:00\n", "43|42|01:00\n",
     "42|42_0|01:00\n", "01:00\n", "42|42|01:00|extra\n",
     "42|42|UNLIMITED\n", "42|42|NOT_SET\n", "42|42|-01:00\n"],
)
def test_wrong_missing_duplicate_or_nonfinite_records_fail_closed(monkeypatch, response):
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setattr(slurm.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=response))
    with pytest.raises(ValueError):
        slurm.remaining_seconds("42")


def test_cannot_borrow_another_job_budget(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setattr(slurm.subprocess, "run", lambda *a, **k: pytest.fail("must not query"))
    with pytest.raises(ValueError, match="current SLURM_JOB_ID"):
        slurm.remaining_seconds("43")


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired("squeue", 15),
                                  subprocess.CalledProcessError(1, "squeue")])
def test_query_failure_never_falls_back_to_a_default_budget(monkeypatch, error):
    monkeypatch.setenv("SLURM_JOB_ID", "42")

    def query(*args, **kwargs):
        raise error

    monkeypatch.setattr(slurm.subprocess, "run", query)
    with pytest.raises(type(error)):
        slurm.remaining_seconds("42")


@pytest.mark.parametrize("delay,expected", [(2.1, 117), (125.0, 0)])
def test_query_latency_is_charged_and_each_call_resamples(monkeypatch, delay, expected):
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    ticks = iter([100.0, 100.0 + delay, 300.0, 300.0])
    responses = iter(["42|42|02:00\n", "42|42|00:30\n"])
    monkeypatch.setattr(slurm.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        slurm.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=next(responses)),
    )
    assert slurm.remaining_seconds("42") == expected
    assert slurm.remaining_seconds("42") == 30
