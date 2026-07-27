from __future__ import annotations

import json

import pytest

from facdigger.data.adapters import StandardParquetAdapter
from facdigger.data.config import ParquetSourceConfig
from facdigger.data.contracts import DataContractError


def test_source_manifest_requires_provider_neutral_standardization_contract(tmp_path) -> None:
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(
            {
                "provider": "legacy-provider",
            }
        ),
        encoding="utf-8",
    )
    adapter = StandardParquetAdapter(
        ParquetSourceConfig(
            bars=tmp_path / "bars.parquet",
            universe=tmp_path / "universe.parquet",
            source_manifest=manifest,
        )
    )

    with pytest.raises(DataContractError, match="no standardization contract"):
        adapter.load()


def test_standardization_contract_hash_binds_quality_proof_to_files(tmp_path) -> None:
    bars = tmp_path / "bars.parquet"
    universe = tmp_path / "universe.parquet"
    bars.write_bytes(b"changed-bars")
    universe.write_bytes(b"changed-universe")
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(
            {
                "provider": "test-provider",
                "standardization": {
                    "name": "facdigger.standard_parquet",
                    "version": 1,
                    "status": "passed",
                    "research_ready": False,
                    "tables": {
                        "bars": {"file": "bars.parquet", "sha256": "0" * 64},
                        "universe": {
                            "file": "universe.parquet",
                            "sha256": "0" * 64,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    adapter = StandardParquetAdapter(
        ParquetSourceConfig(
            bars=bars,
            universe=universe,
            source_manifest=manifest,
        )
    )

    with pytest.raises(DataContractError, match="hash mismatch for bars"):
        adapter.load()
