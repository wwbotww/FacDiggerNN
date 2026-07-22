"""EODHD data-provider adapter."""

from facdigger.data.providers.eodhd.config import EODHDConfig, load_eodhd_config
from facdigger.data.providers.eodhd.provider import EODHDProvider

__all__ = ["EODHDConfig", "EODHDProvider", "load_eodhd_config"]
