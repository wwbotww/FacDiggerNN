"""Configuration-driven provider factory."""

from __future__ import annotations

from pathlib import Path

import yaml

from facdigger.data.providers.base import MarketDataProvider
from facdigger.data.providers.eodhd import EODHDProvider, load_eodhd_config


def provider_from_config(path: str | Path) -> MarketDataProvider:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    provider = raw.get("provider")
    if provider == "eodhd":
        return EODHDProvider(load_eodhd_config(config_path))
    raise ValueError(f"Unsupported data provider: {provider!r}")
