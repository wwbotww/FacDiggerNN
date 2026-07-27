"""Provider-neutral provenance contract for standardized Parquet sources."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from facdigger.data.contracts import DataContractError

STANDARDIZATION_CONTRACT_NAME = "facdigger.standard_parquet"
STANDARDIZATION_CONTRACT_VERSION = 1
STANDARD_TABLES = {"bars", "universe", "corporate_actions", "delistings"}
REQUIRED_STANDARD_TABLES = {"bars", "universe"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_table_evidence(tables: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(tables, dict):
        raise DataContractError("source standardization contract tables must be a mapping")
    unknown = sorted(set(tables) - STANDARD_TABLES)
    if unknown:
        raise DataContractError(
            f"source standardization contract contains unknown tables: {unknown}"
        )
    missing = sorted(REQUIRED_STANDARD_TABLES - set(tables))
    if missing:
        raise DataContractError(
            f"source standardization contract is missing required tables: {missing}"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for name, evidence in tables.items():
        if not isinstance(evidence, dict):
            raise DataContractError(
                f"source standardization evidence for {name} must be a mapping"
            )
        digest = evidence.get("sha256")
        filename = evidence.get("file")
        if not isinstance(filename, str) or not filename:
            raise DataContractError(
                f"source standardization evidence for {name} has no file name"
            )
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise DataContractError(
                f"source standardization evidence for {name} has no valid SHA-256"
            )
        normalized[name] = dict(evidence)
    return normalized


def build_standardization_contract(
    tables: dict[str, dict[str, Any]],
    *,
    research_ready: bool,
) -> dict[str, Any]:
    """Create the vendor-neutral proof consumed beyond the provider boundary."""

    return {
        "name": STANDARDIZATION_CONTRACT_NAME,
        "version": STANDARDIZATION_CONTRACT_VERSION,
        "status": "passed",
        "research_ready": research_ready,
        "tables": _validate_table_evidence(tables),
    }


def read_source_provenance_manifest(path: Path) -> dict[str, Any]:
    """Read and normalize one versioned standardization proof."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise DataContractError(f"Cannot read source manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataContractError("source manifest root must be a mapping")
    contract = payload.get("standardization")
    if not isinstance(contract, dict):
        raise DataContractError(
            "source manifest has no standardization contract; re-run provider ingestion"
        )
    if contract.get("name") != STANDARDIZATION_CONTRACT_NAME:
        raise DataContractError("source manifest standardization contract name is unsupported")
    if contract.get("version") != STANDARDIZATION_CONTRACT_VERSION:
        raise DataContractError("source manifest standardization contract version is unsupported")
    status = contract.get("status")
    if status not in {"passed", "failed"}:
        raise DataContractError("source standardization status must be passed or failed")
    research_ready = contract.get("research_ready")
    if not isinstance(research_ready, bool):
        raise DataContractError("source standardization research_ready must be boolean")
    warnings = payload.get("warnings") or []
    if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        raise DataContractError("source manifest warnings must be a list of strings")
    return {
        "available": True,
        "provider": payload.get("provider"),
        "source_revision": payload.get("source_revision"),
        "standardization": {
            "name": contract["name"],
            "version": contract["version"],
            "status": status,
        },
        "research_ready": research_ready,
        "tables": _validate_table_evidence(contract.get("tables")),
        "warnings": warnings,
    }


def require_accepted_source(provenance: dict[str, Any]) -> None:
    if provenance["standardization"]["status"] != "passed":
        raise DataContractError("source standardization contract did not pass")


def validate_source_table_bindings(
    provenance: dict[str, Any],
    configured: dict[str, Path | None],
) -> None:
    """Bind the provider-neutral proof to the exact configured Parquet files."""

    tables = provenance["tables"]
    for name, table_path in configured.items():
        if table_path is None:
            continue
        evidence = tables.get(name)
        if evidence is None:
            raise DataContractError(
                f"source standardization contract has no evidence for configured {name}"
            )
        if not table_path.is_file():
            raise DataContractError(f"Source Parquet does not exist: {table_path}")
        actual = _sha256_file(table_path)
        expected = evidence["sha256"]
        if actual != expected:
            raise DataContractError(
                f"source table hash mismatch for {name}: expected={expected}, actual={actual}"
            )
