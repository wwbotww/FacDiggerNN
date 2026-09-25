"""ICF CPU preparation: private credentials, persistent cache, one writer, manual retry."""

from __future__ import annotations

import json
import os
import shlex
import stat
from pathlib import Path

from facdigger.data.config import load_dataset_build_config
from facdigger.data.providers.eodhd.config import load_eodhd_config
from facdigger.data.providers.eodhd.provider import EODHDProvider
from facdigger.research.transformer_config import load_transformer_comparison_config
from facdigger.research.transformer_snapshots import prepare_transformer_snapshots
from facdigger.training.runtime import load_training_runtime, run_lock, write_json


def load_credential(path: Path) -> str:
    """Read the dedicated 0700 directory / 0600 file without executing shell code."""
    parent = path.parent.stat()
    if (
        not path.is_absolute()
        or parent.st_uid != os.getuid()
        or (stat.S_IMODE(parent.st_mode) != 0o700)
    ):
        raise ValueError("credential directory must be private (0700) and owned by this user")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or (stat.S_IMODE(info.st_mode) != 0o600)
        ):
            raise ValueError("credential must be a user-owned regular file with mode 0600")
        try:
            parts = shlex.split(stream.read())
        except ValueError:
            raise ValueError("invalid credential assignment format") from None
    if parts and parts[0] == "export":
        parts = parts[1:]
    prefix = "EODHD_API_TOKEN="
    if len(parts) != 1 or not parts[0].startswith(prefix) or not parts[0][len(prefix) :]:
        raise ValueError("credential file must contain only one EODHD_API_TOKEN assignment")
    return parts[0][len(prefix) :]


def _persistent(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if (
        not path.is_absolute()
        or not resolved.is_relative_to(root)
        or str(resolved).startswith("/afs/")
    ):
        raise ValueError("ICF data paths must be absolute, under FD_ROOT, and outside AFS")
    return resolved


def run_job() -> dict:
    requested_root = Path(os.environ["FD_ROOT"])
    root = _persistent(requested_root, requested_root.resolve())
    mode = os.environ["FD_DATA_MODE"]
    if mode not in {"plan", "ingest", "prepare"}:
        raise ValueError("FD_DATA_MODE must be plan, ingest or prepare")
    # All three modes share the lock: no bronze writer may race feature building.
    with run_lock(root / ".data.lock"):
        if mode == "prepare":
            config = load_transformer_comparison_config(os.environ["FD_RESEARCH_CONFIG"])
            base = load_dataset_build_config(config.base_dataset_config)
            for path in base.sources.model_dump().values():
                if path is not None:
                    _persistent(Path(path), root)
            _persistent(config.snapshot_output_root, root)
            output = _persistent(Path(os.environ["FD_PREPARE_OUTPUT"]), root)
            result = prepare_transformer_snapshots(
                config, output, load_training_runtime(os.environ["FD_RUNTIME"])
            )
        else:
            config = load_eodhd_config(os.environ["FD_DATA_CONFIG"])
            if config.universe.mode != "historical_liquid" or config.api_token_env != (
                "EODHD_API_TOKEN"
            ):
                raise ValueError("ICF download requires historical_liquid and EODHD_API_TOKEN")
            for path in (config.output_dir, config.cache_dir, config.state_dir):
                _persistent(path, root)
            os.environ["EODHD_API_TOKEN"] = load_credential(Path(os.environ["FD_CREDENTIAL_FILE"]))
            try:
                provider = EODHDProvider(config)
                reserve = int(os.environ["FD_RESERVE_API_CALLS"])
                provider.configure_download(
                    reserve_calls=reserve,
                    requests_per_minute=int(os.environ["FD_REQUESTS_PER_MINUTE"]),
                )
                if mode == "plan":
                    result = provider.plan_historical_download(reserve_calls=reserve)
                    write_json(root / "inputs" / "download_plan.json", result)
                else:
                    ingestion = provider.ingest()
                    result = {
                        "status": "complete",
                        "output_dir": str(ingestion.output_dir),
                        "warnings": ingestion.manifest["warnings"],
                    }
            finally:
                os.environ.pop("EODHD_API_TOKEN", None)
        return result


if __name__ == "__main__":
    try:
        print(json.dumps(run_job(), ensure_ascii=False, indent=2, default=str))
    except Exception as exc:
        # Avoid tracebacks containing provider responses or local credential contents.
        print(f"CPU preparation stopped: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1) from None
