"""Runtime environment diagnostics that never crash on a failed optional import."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    installed_version: str | None
    importable: bool
    import_error: str | None = None


def inspect_dependency(distribution: str, module: str | None = None) -> DependencyStatus:
    module_name = module or distribution.replace("-", "_")
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None

    try:
        importlib.import_module(module_name)
        return DependencyStatus(distribution, version, True)
    except Exception as exc:  # optional binary dependencies can fail with non-ImportError errors
        return DependencyStatus(
            distribution,
            version,
            False,
            f"{type(exc).__name__}: {exc}",
        )


def collect_environment(include_model_dependencies: bool = True) -> dict[str, Any]:
    dependencies = [
        ("pydantic", "pydantic"),
        ("PyYAML", "yaml"),
        ("typer", "typer"),
    ]
    if include_model_dependencies:
        dependencies.extend(
            [
                ("numpy", "numpy"),
                ("torch", "torch"),
                ("transformers", "transformers"),
                ("huggingface-hub", "huggingface_hub"),
            ]
        )

    result: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": [asdict(inspect_dependency(*item)) for item in dependencies],
    }

    torch_status = next((item for item in result["dependencies"] if item["name"] == "torch"), None)
    if torch_status and torch_status["importable"]:
        torch = importlib.import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
        result["torch"] = {
            "cuda_available": cuda_available,
            "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        }
    return result


def environment_is_healthy(report: dict[str, Any], require_model: bool = False) -> bool:
    required = {"pydantic", "PyYAML", "typer"}
    if require_model:
        required.update({"numpy", "torch", "transformers", "huggingface-hub"})
    return all(
        item["importable"] for item in report["dependencies"] if item["name"] in required
    ) and required.issubset({item["name"] for item in report["dependencies"]})
