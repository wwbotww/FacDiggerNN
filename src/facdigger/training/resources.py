"""Explicit benchmark budgets; independent of any scheduler or deployment site."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from facdigger.data.config import StrictModel


class TrainingResourceBudget(StrictModel):
    cuda_peak_reserved_bytes: int = Field(default=int(7.2 * 1024**3), gt=0)
    host_peak_rss_bytes: int = Field(default=13 * 1024**3, gt=0)
    projected_days: float = Field(default=14.0, gt=0, allow_inf_nan=False)
    cuda_memory_fraction: float = Field(default=1.0, gt=0, le=1)


def load_resource_budget(path: str | Path | None) -> TrainingResourceBudget | None:
    if path is None:
        return None
    return TrainingResourceBudget.model_validate(
        yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    )


def cgroup_memory_limit() -> int | None:
    """Read finite cgroup-v2 constraints, including parents; unknown stays unknown."""
    if sys.platform != "linux":
        return None
    try:
        entries = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
        relative = next(line[3:] for line in entries if line.startswith("0::"))
        root = Path("/sys/fs/cgroup")
        directory = root / relative.lstrip("/")
        limits = []
        for parent in [directory, *directory.parents]:
            if not parent.is_relative_to(root):
                break
            path = parent / "memory.max"
            if path.is_file():
                value = path.read_text(encoding="utf-8").strip()
                if value != "max":
                    limits.append(int(value))
        return min(limits) if limits else None
    except (OSError, ValueError, StopIteration):
        return None


def training_hardware() -> dict[str, Any]:
    import torch

    cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_properties(0) if cuda else None
    return {
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": "cuda" if cuda else "cpu",
        "device_name": device.name if device else None,
        "device_memory_bytes": int(device.total_memory) if device else None,
        "compute_capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
    }


def effective_resource_limits(
    budget: TrainingResourceBudget,
    hardware: dict[str, Any],
) -> dict[str, int | float]:
    gpu_limit = budget.cuda_peak_reserved_bytes
    if hardware["device_memory_bytes"] is not None:
        gpu_limit = min(
            gpu_limit, int(hardware["device_memory_bytes"] * budget.cuda_memory_fraction)
        )
    memory_limit = cgroup_memory_limit()
    return {
        "cuda_peak_reserved_bytes": gpu_limit,
        "host_peak_rss_bytes": min(budget.host_peak_rss_bytes, memory_limit)
        if memory_limit is not None
        else budget.host_peak_rss_bytes,
        "projected_days": budget.projected_days,
    }
