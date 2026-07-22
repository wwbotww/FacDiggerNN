"""Strict configuration for the EODHD provider adapter."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AnyHttpUrl, Field, model_validator

from facdigger.data.config import StrictModel


class SecurityMetadataOverride(StrictModel):
    """Optional metadata supplied without spending an API request."""

    provider_symbol: str
    isin: str | None = None
    name: str | None = None
    exchange: str | None = None
    currency: str = "USD"
    security_type: str = "Common Stock"


class EODHDConfig(StrictModel):
    provider: Literal["eodhd"] = "eodhd"
    base_url: AnyHttpUrl = "https://eodhd.com/api"
    api_token_env: str = "EODHD_API_TOKEN"
    allow_demo_token: bool = True
    symbols: list[str] = Field(min_length=1)
    exchange_code: str = "US"
    start: date | None = None
    end: date | None = None
    lookback_days: int = Field(default=365, ge=1, le=3650)
    output_dir: Path = Path("data/bronze")
    cache_dir: Path = Path("data/cache/eodhd")
    state_dir: Path = Path("data/state/eodhd")
    cache_ttl_hours: int = Field(default=24, ge=0)
    refresh: bool = False
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    max_retries: int = Field(default=2, ge=0, le=5)
    max_calls_per_day: int = Field(default=20, ge=1)
    fetch_symbol_metadata: bool = False
    metadata_failure_policy: Literal["warn", "fail"] = "warn"
    include_corporate_actions: bool = False
    metadata_overrides: list[SecurityMetadataOverride] = Field(default_factory=list)
    min_listed_sessions: int = Field(default=20, ge=1)
    min_price: float = Field(default=1.0, ge=0)
    min_adv20_usd: float = Field(default=1_000_000.0, ge=0)

    @model_validator(mode="after")
    def validate_dates_and_symbols(self) -> EODHDConfig:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("start must not be after end")
        normalized = [symbol.strip().upper() for symbol in self.symbols]
        if any(not symbol or "." not in symbol for symbol in normalized):
            raise ValueError("EODHD symbols must include an exchange suffix, e.g. AAPL.US")
        if len(set(normalized)) != len(normalized):
            raise ValueError("symbols must be unique")
        self.symbols = normalized
        return self

    def resolved_dates(self, today: date | None = None) -> tuple[date, date]:
        anchor = today or date.today()
        end = self.end or anchor
        start = self.start or (end - timedelta(days=self.lookback_days))
        return start, end


def load_eodhd_config(path: str | Path) -> EODHDConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"EODHD configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return EODHDConfig.model_validate(raw)
