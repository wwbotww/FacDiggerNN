from __future__ import annotations

import json

import pytest

from facdigger.data.adapters import StandardParquetAdapter
from facdigger.data.config import ParquetSourceConfig
from facdigger.data.contracts import DataContractError


def test_historical_eodhd_manifest_requires_passed_quality_gate(tmp_path) -> None:
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(
            {
                "provider": "eodhd",
                "selection": {"mode": "historical_liquid"},
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

    with pytest.raises(DataContractError, match="passed quality gate"):
        adapter.load()


def test_historical_eodhd_manifest_hash_binds_quality_proof_to_files(tmp_path) -> None:
    bars = tmp_path / "bars.parquet"
    universe = tmp_path / "universe.parquet"
    bars.write_bytes(b"changed-bars")
    universe.write_bytes(b"changed-universe")
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(
            {
                "provider": "eodhd",
                "selection": {"mode": "historical_liquid"},
                "quality": {"gate": {"status": "passed"}},
                "tables": {
                    "bars": {"sha256": "0" * 64},
                    "universe": {"sha256": "0" * 64},
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
