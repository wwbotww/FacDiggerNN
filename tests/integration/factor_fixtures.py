"""Small standard-table fixtures shared by model delivery integration tests."""

from datetime import date, timedelta

import polars as pl

from facdigger.data.config import DatasetBuildConfig
from facdigger.data.snapshots import build_dataset_snapshot


def sessions(count: int) -> list[date]:
    sessions = []
    current = date(2022, 1, 3)
    while len(sessions) < count:
        if current.weekday() < 5:
            sessions.append(current)
        current += timedelta(days=1)
    return sessions


def build_price_volume_snapshot(tmp_path, *, count: int = 85):
    calendar = sessions(count)
    bars = []
    universe = []
    for security_index in range(6):
        security_id = f"sec-{security_index}"
        for index, trade_date in enumerate(calendar):
            close = 20.0 + security_index * 3 + index * (0.02 + security_index * 0.002)
            volume = 1_000_000.0 + security_index * 100_000 + index
            bars.append(
                {
                    "security_id": security_id,
                    "symbol": f"S{security_index}",
                    "trade_date": trade_date,
                    "open": close - 0.05,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": volume,
                    "dollar_volume": close * volume,
                    "adj_factor": 1.0,
                    "source_revision": "synthetic-e3",
                }
            )
            universe.append(
                {
                    "security_id": security_id,
                    "symbol": f"S{security_index}",
                    "trade_date": trade_date,
                    "listed_days": 300 + index,
                    "exchange": "XNAS",
                    "security_type": "common_stock",
                    "is_primary_listing": True,
                    "is_listed": True,
                    "is_delisted": False,
                    "is_halted": False,
                    "industry_code": "TECH" if security_index < 3 else "FIN",
                    "float_market_cap": float(1_000_000_000 * (security_index + 1)),
                    "close": close,
                    "adv20_usd": close * volume,
                    "eligible": True,
                }
            )
    bronze = tmp_path / "bronze"
    bronze.mkdir()
    bars_path = bronze / "bars.parquet"
    universe_path = bronze / "universe.parquet"
    pl.DataFrame(bars).write_parquet(bars_path)
    pl.DataFrame(universe).write_parquet(universe_path)
    config = DatasetBuildConfig.model_validate(
        {
            "sources": {"bars": bars_path, "universe": universe_path},
            "output_root": tmp_path / "snapshots",
            "features": {"context_length": 20},
            "split": {
                "train_end": calendar[42],
                "valid_end": calendar[62],
                "test_end": calendar[80],
                "embargo_sessions": 2,
            },
        }
    )
    return build_dataset_snapshot(config)[0]
