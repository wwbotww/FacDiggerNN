from datetime import datetime, timedelta, timezone

from facdigger.production.config import ProductionServiceConfig
from facdigger.production.service import production_health
from facdigger.production.state import ProductionState


def _config(tmp_path) -> ProductionServiceConfig:
    return ProductionServiceConfig.model_validate(
        {
            "data": {
                "provider_config": tmp_path / "provider.yaml",
                "store_root": tmp_path / "store",
                "bootstrap_source": tmp_path / "bronze",
            },
            "model": {
                "release_root": tmp_path / "releases",
                "release_id": "1" * 64,
            },
            "inference": {"output_root": tmp_path / "inference"},
            "factor_batch": {"output_root": tmp_path / "factors"},
            "state_database": tmp_path / "production.sqlite3",
            "poll_seconds": 60,
        }
    )


def test_health_requires_recent_heartbeat(tmp_path) -> None:
    config = _config(tmp_path)
    now = datetime(2026, 8, 13, 0, 0, tzinfo=timezone.utc)

    assert production_health(config, now=now)["healthy"] is False
    with ProductionState(config.state_database) as state:
        state.heartbeat(at=now - timedelta(seconds=120), phase="sleeping")
    assert production_health(config, now=now)["healthy"] is True
    with ProductionState(config.state_database) as state:
        state.heartbeat(at=now - timedelta(seconds=181), phase="sleeping")
    assert production_health(config, now=now)["healthy"] is False
