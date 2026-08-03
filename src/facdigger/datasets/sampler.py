"""Deterministic batching that keeps each cross-sectional date group intact."""

from __future__ import annotations

import random
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from typing import Any


class DateGroupedBatchSampler:
    def __init__(
        self,
        asof_dates: Sequence[Any],
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        minimum_group_size: int = 1,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if minimum_group_size < 1:
            raise ValueError("minimum_group_size must be positive")
        if minimum_group_size > (batch_size + 1) // 2:
            raise ValueError(
                "minimum_group_size cannot exceed half of batch_size for balanced chunks"
            )
        if drop_last:
            raise ValueError("DateGroupedBatchSampler cannot drop cross-sectional rows")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.minimum_group_size = minimum_group_size
        self.drop_last = drop_last
        self.epoch = 0
        grouped: OrderedDict[Any, list[int]] = OrderedDict()
        for index, asof_date in enumerate(asof_dates):
            grouped.setdefault(asof_date, []).append(index)
        self.groups = list(grouped.values())
        undersized = [len(group) for group in self.groups if len(group) < minimum_group_size]
        if undersized:
            raise ValueError(
                "date group is smaller than minimum_group_size: "
                f"{min(undersized)} < {minimum_group_size}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = int(state["epoch"])

    def _batches(self) -> list[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        groups = [list(group) for group in self.groups]
        if self.shuffle:
            generator.shuffle(groups)
            for group in groups:
                generator.shuffle(group)
        batches: list[list[int]] = []
        for group in groups:
            chunk_count = (len(group) + self.batch_size - 1) // self.batch_size
            base_size, larger_chunks = divmod(len(group), chunk_count)
            if base_size < self.minimum_group_size:
                raise ValueError(
                    "date group cannot be split without exceeding batch_size or violating "
                    "minimum_group_size"
                )
            start = 0
            for chunk_index in range(chunk_count):
                size = base_size + (1 if chunk_index < larger_chunks else 0)
                chunk = group[start : start + size]
                start += size
                batches.append(chunk)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())


class SequenceBatchSampler:
    """Deterministically shuffle independent sequence windows by epoch."""

    def __init__(
        self,
        size: int,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if size < 1:
            raise ValueError("size must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.size = size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = int(state["epoch"])

    def _batches(self) -> list[list[int]]:
        indices = list(range(self.size))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, self.size, self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        if self.drop_last:
            return self.size // self.batch_size
        return (self.size + self.batch_size - 1) // self.batch_size
