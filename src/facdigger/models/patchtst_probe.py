"""Executable compatibility probe for the pinned IBM PatchTST checkpoint."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from facdigger.config import ProjectConfig
from facdigger.models.patchtst_adapter import (
    CanonicalPatchTSTConfig,
    canonical_state_key,
    load_matching_encoder_weights,
)


class PatchTSTProbeError(RuntimeError):
    """Raised after a failed probe has persisted its diagnostic report."""


def _select_device(torch: Any, preferred: str) -> str:
    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise PatchTSTProbeError("runtime.preferred_device=cuda but CUDA is unavailable")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_patchtst_probe(
    project_config: ProjectConfig,
    output_dir: str | Path,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Run the pinned-checkpoint load, transfer, train-step and resume smoke test."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "compatibility_report.json"
    source = project_config.model.source
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "source_model": source.model_id,
        "source_revision": source.revision,
        "minimum_loaded_numel_ratio": source.minimum_loaded_numel_ratio,
        "local_files_only": local_files_only,
    }

    try:
        import torch
        import transformers
        from huggingface_hub import hf_hub_download
        from transformers import PatchTSTConfig, PatchTSTForPretraining, PatchTSTModel

        report["versions"] = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }

        random.seed(project_config.seed)
        torch.manual_seed(project_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(project_config.seed)

        config_path = hf_hub_download(
            repo_id=source.model_id,
            filename="config.json",
            revision=source.revision,
            local_files_only=local_files_only,
        )
        raw_checkpoint_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        canonical = CanonicalPatchTSTConfig.from_checkpoint_dict(raw_checkpoint_config)
        report["raw_checkpoint_config"] = raw_checkpoint_config
        report["canonical_checkpoint_config"] = canonical.model_dump(mode="json")

        expected = project_config.model
        compatibility_checks = {
            "context_length": (canonical.context_length, expected.context_length),
            "num_input_channels": (canonical.num_input_channels, expected.input_channels),
            "patch_length": (canonical.patch_length, expected.patch_length),
            "patch_stride": (canonical.patch_stride, expected.patch_stride),
        }
        report["project_shape_compatibility"] = {
            key: {"source": pair[0], "project": pair[1], "matches": pair[0] == pair[1]}
            for key, pair in compatibility_checks.items()
        }
        incompatible = [key for key, pair in compatibility_checks.items() if pair[0] != pair[1]]
        if incompatible:
            raise PatchTSTProbeError(
                "Project model shape differs from the pinned source config: "
                + ", ".join(incompatible)
            )

        weights_path = hf_hub_download(
            repo_id=source.model_id,
            filename="pytorch_model.bin",
            revision=source.revision,
            local_files_only=local_files_only,
        )
        raw_state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(raw_state, dict) and "state_dict" in raw_state:
            raw_state = raw_state["state_dict"]
        if not isinstance(raw_state, dict):
            raise PatchTSTProbeError("Downloaded checkpoint does not contain a state dict")

        source_hf_config = PatchTSTConfig(**canonical.to_transformers_kwargs())
        source_model = PatchTSTForPretraining(source_hf_config)
        source_backbone = source_model.base_model
        raw_encoder_state = {
            key: value
            for key, value in raw_state.items()
            if canonical_state_key(key).startswith("encoder.")
        }
        target_parameter_names = {
            canonical_state_key(key) for key, _ in source_backbone.named_parameters()
        }
        raw_parameter_names = {
            key for key in raw_encoder_state if canonical_state_key(key) in target_parameter_names
        }
        checkpoint_load_report = load_matching_encoder_weights(
            target_backbone=source_backbone,
            source_encoder_state=raw_encoder_state,
            minimum_loaded_numel_ratio=source.minimum_loaded_numel_ratio,
            source_parameter_names=raw_parameter_names,
        )
        report["checkpoint_to_library"] = checkpoint_load_report.to_dict()

        target_kwargs = canonical.to_transformers_kwargs()
        target_kwargs["do_mask_input"] = False
        target_hf_config = PatchTSTConfig(**target_kwargs)
        target_backbone = PatchTSTModel(target_hf_config)
        transfer_report = load_matching_encoder_weights(
            target_backbone=target_backbone,
            source_encoder_state=source_backbone.state_dict(),
            minimum_loaded_numel_ratio=source.minimum_loaded_numel_ratio,
            source_parameter_names={name for name, _ in source_backbone.named_parameters()},
        )
        report["source_backbone_to_target"] = transfer_report.to_dict()

        device = _select_device(torch, project_config.runtime.preferred_device)
        target_backbone.to(device)
        target_backbone.train()
        optimizer = torch.optim.AdamW(target_backbone.parameters(), lr=1e-5)
        use_amp = device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        past_values = torch.randn(
            2,
            expected.context_length,
            expected.input_channels,
            device=device,
        )
        observed_mask = torch.ones_like(past_values, dtype=torch.bool)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            outputs = target_backbone(
                past_values=past_values,
                past_observed_mask=observed_mask,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state
            loss = hidden.float().square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        checkpoint_path = output / "smoke_checkpoint.pt"
        torch.save(
            {
                "model": target_backbone.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "global_step": 1,
                "seed": project_config.seed,
            },
            checkpoint_path,
        )
        resumed_model = PatchTSTModel(target_hf_config).to(device)
        resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-5)
        resumed = torch.load(checkpoint_path, map_location=device, weights_only=False)
        resumed_model.load_state_dict(resumed["model"], strict=True)
        resumed_optimizer.load_state_dict(resumed["optimizer"])
        if resumed.get("global_step") != 1:
            raise PatchTSTProbeError("Checkpoint global_step was not restored")

        report["smoke_test"] = {
            "device": device,
            "amp_enabled": use_amp,
            "input_shape": list(past_values.shape),
            "hidden_shape": list(hidden.shape),
            "loss": float(loss.detach().cpu()),
            "forward_backward": True,
            "checkpoint_resume": True,
            "checkpoint_path": str(checkpoint_path),
            "cuda_peak_memory_mb": (
                float(torch.cuda.max_memory_allocated() / 1024**2) if use_amp else None
            ),
        }
        report["status"] = "passed"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_report(report_path, report)
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_report(report_path, report)
        raise PatchTSTProbeError(
            f"PatchTST compatibility probe failed; diagnostics: {report_path}"
        ) from exc
