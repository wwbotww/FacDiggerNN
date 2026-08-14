from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from pydantic import ValidationError

pytest.importorskip("torch")
pytest.importorskip("transformers")

from facdigger.inference.history import (  # noqa: E402
    HistoricalReplayConfig,
    _partition_rows,
)


def _config(**updates) -> dict:
    values = {
        "history_id": "fixed-e3-history",
        "release_dir": "artifacts/releases/release",
        "inference_snapshot_dir": "data/inference_snapshots/snapshot",
        "acknowledge_non_oos": True,
    }
    values.update(updates)
    return values


def test_historical_replay_requires_explicit_non_oos_acknowledgement() -> None:
    with pytest.raises(ValidationError, match="acknowledge_non_oos"):
        HistoricalReplayConfig.model_validate(
            _config(acknowledge_non_oos=False)
        )


def test_historical_replay_canonicalizes_security_scope() -> None:
    config = HistoricalReplayConfig.model_validate(
        _config(security_ids=["sec-b", "sec-a"])
    )

    assert config.security_ids == ["sec-a", "sec-b"]


@pytest.mark.parametrize(
    "updates, message",
    [
        (
            {"start_date": date(2024, 1, 2), "end_date": date(2024, 1, 1)},
            "start_date",
        ),
        ({"security_ids": ["sec-a", "sec-a"]}, "unique"),
        ({"security_ids": [" "]}, "empty"),
    ],
)
def test_historical_replay_rejects_ambiguous_scope(updates, message) -> None:
    with pytest.raises(ValidationError, match=message):
        HistoricalReplayConfig.model_validate(_config(**updates))


def test_historical_replay_partitions_rows_by_calendar_year() -> None:
    rows = pl.DataFrame(
        {
            "security_id": ["sec-b", "sec-a", "sec-a", "sec-b"],
            "asof_date": [
                date(2023, 1, 3),
                date(2022, 12, 30),
                date(2023, 1, 3),
                date(2022, 12, 30),
            ],
        },
        schema={"security_id": pl.String, "asof_date": pl.Date},
    )

    partitions = _partition_rows(rows)

    assert [year for year, _ in partitions] == [2022, 2023]
    assert [partition.height for _, partition in partitions] == [2, 2]
    assert partitions[0][1]["security_id"].to_list() == ["sec-a", "sec-b"]
    assert partitions[1][1]["security_id"].to_list() == ["sec-a", "sec-b"]
