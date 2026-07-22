"""Provider boundary for converting vendor payloads into standard tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ProviderIngestResult:
    """Files produced by one provider ingestion run."""

    provider: str
    output_dir: Path
    files: dict[str, Path]
    manifest: dict[str, Any]


@runtime_checkable
class MarketDataProvider(Protocol):
    """Minimal boundary that keeps downstream code vendor-neutral."""

    name: str

    def probe(self) -> dict[str, Any]: ...

    def ingest(self) -> ProviderIngestResult: ...
