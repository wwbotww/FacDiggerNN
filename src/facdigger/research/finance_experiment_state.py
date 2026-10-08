"""Immutable observations and checked, read-only imports of the two-epoch prefix."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.experiments.manifest import sha256_json
from facdigger.models.finance_patch_transformer import build_finance_transformer_model
from facdigger.models.finance_statistics import FinanceStatisticsRanker, statistics_dropout_identity
from facdigger.research.finance_experiment_config import candidate_settings
from facdigger.research.finance_experiment_plan import disjoint_output
from facdigger.research.finance_prefix_diagnostics import _observe_chunks
from facdigger.training.runtime import _sync_directory, save_checkpoint, write_json


def file_inventory(root: Path, files: list[Path]) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(set(files))}


def verify_inventory(root: Path, inventory: dict) -> None:
    if not isinstance(inventory, dict) or not inventory:
        raise DataContractError("committed artifact inventory is empty")
    for name, expected in inventory.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise DataContractError("committed artifact is missing or outside the run")
        if sha256_file(path) != expected:
            raise DataContractError(f"committed artifact checksum differs: {name}")


def execution_identity(config, environment: dict) -> dict:
    available = environment.get("torch", {}).get("cuda_available", False)
    device = config.training.device
    if device == "auto":
        device = "cuda" if available else "cpu"
    if device == "cuda" and not available:
        raise DataContractError("bound CUDA execution environment is unavailable")
    dependencies = {
        item["name"]: item["installed_version"]
        for item in environment.get("dependencies", [])
        if item["name"] in {"numpy", "torch", "transformers"}
    }
    if set(dependencies) != {"numpy", "torch", "transformers"} or not all(dependencies.values()):
        raise DataContractError("numerical dependency versions must be recorded")
    return {
        "device": device,
        "precision": config.training.precision if device == "cuda" else "fp32",
        "python": environment["python"].split()[0], "machine": environment["machine"],
        "dependencies": dependencies,
        "cuda_device_name": (
            environment["torch"].get("cuda_device_name") if device == "cuda" else None
        ),
    }


def build_cell_model(config, candidate: str, context_length: int, device: str):
    kind, dropout = candidate_settings(candidate)
    model = (
        build_finance_transformer_model(config, context_length=context_length)
        if kind is None else FinanceStatisticsRanker(
            kind, input_dim=(len(config.channels) + len(config.market_channels))
            * (5 * len(config.model.statistics_windows) + 1),
            horizons=tuple(config.horizons), dropout=dropout,
        )
    )
    return model.to(device)


def protocol_hash(config, protocol: dict, candidate: str) -> str:
    kind, dropout = candidate_settings(candidate)
    identity = {
        "config": config.model_dump(mode="json"), "data_protocol": protocol,
        **statistics_dropout_identity(kind, dropout),
    }
    if kind is not None:
        identity["diagnostic_model"] = kind
    return sha256_json(identity)


def validate_checkpoint(state: dict, config, protocol: dict, candidate: str) -> None:
    kind, _ = candidate_settings(candidate)
    if (
        state.get("config") != config.model_dump(mode="json")
        or state.get("protocol_hash") != protocol_hash(config, protocol, candidate)
        or state.get("data_protocol") != protocol
        or state.get("dataset_id") != protocol["dataset_id"]
        or state.get("model_type") != (kind or "finance_patch_transformer")
        or state.get("contract") != "finance_patch_transformer_training_resume"
    ):
        raise DataContractError("complete cell checkpoint protocol differs")


def _equal_weights(left: dict, right: dict) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[key].detach().cpu(), right[key].detach().cpu()) for key in left
    )


def _copy_verified(source: Path, destination: Path, expected: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != expected:
            raise DataContractError("imported prefix artifact changed")
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer)
        writer.flush()
        os.fsync(writer.fileno())
    if sha256_file(temporary) != expected or sha256_file(source) != expected:
        raise DataContractError("prefix source changed during the read-only copy")
    temporary.replace(destination)
    _sync_directory(destination.parent)


def import_prefix(source: Path, root: Path, identity: dict, config) -> dict:
    """Copy committed training/RNG state; never write to the source, even its lock."""
    disjoint_output(root, source)
    destination = root / "parent"
    marker = destination / "import.json"
    if marker.is_file():
        record = json.loads(marker.read_text())
        if record["source"] != str(source.resolve()) or record["cell_identity"] != identity:
            raise DataContractError("prefix import binding differs")
        verify_inventory(destination, record["files"])
    else:
        audit_path = source / "diagnostic.json"
        audit_hash = sha256_file(audit_path)
        audit = json.loads(audit_path.read_text())
        origin = audit["identity"]
        kind, dropout = candidate_settings(identity["cell"]["candidate"])
        if (
            origin.get("candidate") != (kind or "finance")
            or origin.get("config") != config.model_dump(mode="json")
            or origin.get("data_protocol") != identity["data_protocol"]
            or origin.get("prefix_epochs") != 2
            or origin.get("statistics_dropout", 0.1) != dropout
            or audit.get("status") != "observations_complete"
            or audit.get("holdout_used") is not False
            or audit.get("training_status") != "paused"
        ):
            raise DataContractError("source is not the matching completed two-epoch prefix")
        if not audit.get("attempts") or any(
            execution_identity(config, attempt["environment"]) != identity["execution"]
            for attempt in audit["attempts"]
        ):
            raise DataContractError("prefix numerical execution environment differs")
        paths = ["diagnostic.json", "checkpoints/last.pt"] + [
            f"observations/epoch-{epoch}.pt" for epoch in (0, 1, 2)
        ]
        files = {name: sha256_file(source / name) for name in paths}
        if files["diagnostic.json"] != audit_hash or (
            files["checkpoints/last.pt"] != audit["checkpoint_sha256"]
        ):
            raise DataContractError("prefix committed checkpoint hash differs")
        state = torch.load(source / "checkpoints/last.pt", map_location="cpu", weights_only=False)
        validate_checkpoint(state, config, identity["data_protocol"], identity["cell"]["candidate"])
        if (
            state["epoch"] != 2 or state["progress"]["phase"] != "epoch_complete"
            or state["progress"]["finished"] or [h["epoch"] for h in state["history"]] != [1, 2]
        ):
            raise DataContractError("prefix is not a committed, unfinished second epoch")
        for epoch in (0, 1, 2):
            name = f"observations/epoch-{epoch}.pt"
            observed = torch.load(source / name, map_location="cpu", weights_only=False)
            if observed.get("identity") != origin or observed.get("epoch") != epoch:
                raise DataContractError("prefix observation identity differs")
            known = audit.get("observation_source_sha256", {}).get(name)
            if known is not None and known != files[name]:
                raise DataContractError("prefix observation checksum differs")
            if epoch == 2 and not _equal_weights(observed["model_state"], state["model_state"]):
                raise DataContractError("prefix epoch2 weights differ from the resume state")
            if epoch == state["best_epoch"] and not _equal_weights(
                observed["model_state"], state["best_checkpoint"]["model_state"],
            ):
                raise DataContractError("prefix best weights differ from its observation")
        for name, expected in files.items():
            _copy_verified(source / name, destination / name, expected)
        verify_inventory(source, files)
        record = {
            "source": str(source.resolve()), "cell_identity": identity, "files": files,
            "source_code": audit["attempts"][-1].get("git"),
        }
        write_json(marker, record)
    target = root / "checkpoints/last.pt"
    if not target.exists():
        _copy_verified(
            destination / "checkpoints/last.pt", target, record["files"]["checkpoints/last.pt"],
        )
    return record


class CompleteStateObserver:
    """Observe all F/S at an epoch boundary, with durable complete-date chunk commits."""

    def __init__(self, *, root, identity, config, datasets, labelled, cached, check_stop, attempt):
        self.root, self.identity, self.config = root, identity, config
        self.datasets, self.labelled, self.cached = datasets, labelled, cached
        self.check_stop, self.attempt = check_stop, attempt

    def marker(self, epoch):
        return self.root / "observations" / f"epoch-{epoch}.json"

    def completed(self, epoch) -> bool:
        path = self.marker(epoch)
        if not path.is_file():
            return False
        audit = json.loads(path.read_text())
        if audit["identity"] != self.identity or audit["epoch"] != epoch:
            raise DataContractError("observation identity differs")
        verify_inventory(self.root, audit["files"])
        return True

    def __call__(self, model, epoch):
        path = self.root / "observations" / f"epoch-{epoch}.pt"
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved["identity"] != self.identity or saved["epoch"] != epoch:
                raise DataContractError("saved observation identity differs")
            if not _equal_weights(saved["model_state"], model.state_dict()):
                raise DataContractError("observation weights changed at the same epoch")
        else:
            save_checkpoint(path, {
                "identity": self.identity, "epoch": epoch,
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            })
        # Commit the weight binding before the first chunk. A final observation marker
        # alone cannot protect partially scored epochs from changed source weights.
        source_path = path.with_suffix(".source.json")
        source = {
            "identity": self.identity, "epoch": epoch, "weights_sha256": sha256_file(path),
        }
        if source_path.is_file():
            if json.loads(source_path.read_text()) != source:
                raise DataContractError("partial observation source weights changed")
        else:
            if any((self.root / "chunks").glob(f"epoch-{epoch}-[FS]-*")):
                raise DataContractError("partial observation has no committed source binding")
            write_json(source_path, source)
        if self.completed(epoch):
            return
        weight_hash = sha256_file(path)
        for phase in ("F", "S"):
            self.check_stop()
            _observe_chunks(
                model, self.datasets[phase], self.labelled[phase], self.cached[phase],
                self.config, self.root, epoch=epoch, phase=phase, precision="fp32",
                check_stop=self.check_stop, attempt=self.attempt,
            )
        original = torch.load(path, map_location="cpu", weights_only=False)
        if sha256_file(path) != weight_hash or not _equal_weights(
            original["model_state"], model.state_dict(),
        ):
            raise DataContractError("fixed observation changed model state")
        files = [path, source_path, *self.root.glob(f"epoch-{epoch}-[FS]-*.parquet")]
        for folder in (self.root / "chunks").glob(f"epoch-{epoch}-[FS]-*"):
            files.extend(p for p in folder.iterdir() if not p.name.endswith(".tmp") and p.is_file())
        write_json(self.marker(epoch), {
            "identity": self.identity, "epoch": epoch, "optimizer_updates": 0,
            "files": file_inventory(self.root, files),
        })

    def repair(self, state: dict, parent: bool) -> None:
        epochs = [0, *[row["epoch"] for row in state["history"]]]
        model = None
        for epoch in epochs:
            self.check_stop()
            if self.completed(epoch):
                continue
            path = self.root / "observations" / f"epoch-{epoch}.pt"
            if path.is_file():
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if saved["identity"] != self.identity or saved["epoch"] != epoch:
                    raise DataContractError("repair observation identity differs")
                weights = saved["model_state"]
            elif parent and epoch <= 2:
                weights = torch.load(
                    self.root / "parent/observations" / f"epoch-{epoch}.pt",
                    map_location="cpu", weights_only=False,
                )["model_state"]
            elif epoch == state["epoch"] and state["progress"]["phase"] == "epoch_complete":
                weights = state["model_state"]
            else:
                raise DataContractError("earlier epoch observation weights are missing")
            if (
                epoch == state["epoch"] and state["progress"]["phase"] == "epoch_complete"
                and not _equal_weights(weights, state["model_state"])
            ):
                raise DataContractError("observation weights differ from the committed epoch")
            if model is None:
                model = build_cell_model(
                    self.config, self.identity["cell"]["candidate"],
                    self.datasets["F"].context_length, self.identity["execution"]["device"],
                )
            model.load_state_dict(weights, strict=True)
            self(model, epoch)
