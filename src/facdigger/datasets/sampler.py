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
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        grouped: OrderedDict[Any, list[int]] = OrderedDict()
        for index, asof_date in enumerate(asof_dates):
            grouped.setdefault(asof_date, []).append(index)
        self.groups = list(grouped.values())

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
        current: list[int] = []
        for group in groups:
            chunks = [
                group[start : start + self.batch_size]
                for start in range(0, len(group), self.batch_size)
            ]
            for chunk in chunks:
                if current and len(current) + len(chunk) > self.batch_size:
                    batches.append(current)
                    current = []
                current.extend(chunk)
                if len(current) == self.batch_size:
                    batches.append(current)
                    current = []
        if current and not self.drop_last:
            batches.append(current)
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
