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


class FullDateBatchSampler:
    """Yield every row for one date as a single CPU-side DataLoader batch.

    ``batch_size`` is deliberately absent: callers may split the collated date
    into device microbatches, but the optimization objective must still see the
    complete cross-section. No date is split or combined with another date.
    """

    def __init__(
        self,
        asof_dates: Sequence[Any],
        *,
        shuffle: bool,
        seed: int,
        minimum_group_size: int = 1,
        drop_last: bool = False,
    ) -> None:
        if minimum_group_size < 1:
            raise ValueError("minimum_group_size must be positive")
        if drop_last:
            raise ValueError("FullDateBatchSampler cannot drop cross-sectional rows")
        self.shuffle = shuffle
        self.seed = seed
        self.minimum_group_size = minimum_group_size
        self.drop_last = drop_last
        self.epoch = 0
        self.groups: list[tuple[int, int]] = []
        seen: set[Any] = set()
        group_start = 0
        previous: Any | None = None
        for index, asof_date in enumerate(asof_dates):
            if index == 0:
                previous = asof_date
                seen.add(asof_date)
                continue
            if asof_date != previous:
                if asof_date in seen:
                    raise ValueError("FullDateBatchSampler requires contiguous date groups")
                self.groups.append((group_start, index))
                group_start = index
                previous = asof_date
                seen.add(asof_date)
        if len(asof_dates) > 0:
            self.groups.append((group_start, len(asof_dates)))
        if not self.groups:
            raise ValueError("FullDateBatchSampler requires at least one date group")
        undersized = [
            stop - start
            for start, stop in self.groups
            if stop - start < minimum_group_size
        ]
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

    def _batches(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        groups = list(self.groups)
        if self.shuffle:
            generator.shuffle(groups)
        for start, stop in groups:
            group = list(range(start, stop))
            if self.shuffle:
                generator.shuffle(group)
            yield group

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self.groups)


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


class DateSecurityBalancedBatchSampler:
    """Use every endpoint once while interleaving shuffled date cross-sections."""

    def __init__(
        self,
        asof_dates: Sequence[Any],
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not asof_dates:
            raise ValueError("DateSecurityBalancedBatchSampler requires samples")
        grouped: OrderedDict[Any, list[int]] = OrderedDict()
        for index, asof_date in enumerate(asof_dates):
            grouped.setdefault(asof_date, []).append(index)
        self.groups = list(grouped.values())
        self.size = len(asof_dates)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = int(state["epoch"])

    def _batches(self) -> list[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        groups = [list(group) for group in self.groups]
        generator.shuffle(groups)
        for group in groups:
            generator.shuffle(group)
        batches: list[list[int]] = []
        batch: list[int] = []
        active = groups
        while active:
            next_active: list[list[int]] = []
            for group in active:
                batch.append(group.pop())
                if group:
                    next_active.append(group)
                if len(batch) == self.batch_size:
                    batches.append(batch)
                    batch = []
            active = next_active
            generator.shuffle(active)
        if batch:
            batches.append(batch)
        if sum(len(item) for item in batches) != self.size:
            raise RuntimeError("balanced sampler lost or duplicated samples")
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return (self.size + self.batch_size - 1) // self.batch_size
