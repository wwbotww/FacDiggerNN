from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from facdigger.datasets.sampler import SequenceBatchSampler
from facdigger.training.e3_engine import split_pretraining_index


def test_pretraining_split_uses_only_official_train_dates() -> None:
    start = date(2024, 1, 1)
    rows = []
    for index in range(14):
        split = "train" if index < 10 else "valid" if index < 12 else "test"
        rows.append(
            {
                "sample_id": str(index),
                "security_id": "sec",
                "asof_date": start + timedelta(days=index),
                "split": split,
            }
        )
    sample_index = pl.DataFrame(rows)

    pretrain, selection, audit = split_pretraining_index(
        sample_index, validation_fraction=0.2
    )

    assert set(pretrain["split"]) == {"pretrain_train"}
    assert set(selection["split"]) == {"pretrain_selection"}
    assert len(pretrain) == 8
    assert len(selection) == 2
    assert selection["asof_date"].max() < date(2024, 1, 11)
    assert audit["formal_validation_rows_used"] == 0
    assert audit["formal_test_rows_used"] == 0


def test_sequence_sampler_is_epoch_deterministic_and_resumable() -> None:
    sampler = SequenceBatchSampler(10, batch_size=3, shuffle=True, seed=17)
    sampler.set_epoch(4)
    expected = list(sampler)
    restored = SequenceBatchSampler(10, batch_size=3, shuffle=True, seed=17)
    restored.load_state_dict(sampler.state_dict())

    assert list(restored) == expected
    assert sorted(index for batch in expected for index in batch) == list(range(10))
