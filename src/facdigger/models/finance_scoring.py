"""Shared label-free full-date forward path for training evaluation and inference."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from facdigger.datasets.sampler import FullDateBatchSampler
from facdigger.datasets.window import (
    FinanceTransformerInferenceWindowDataset,
    FinanceTransformerWindowDataset,
)
from facdigger.models.finance_patch_transformer import FinancePatchTransformer

FinanceScoringDataset = FinanceTransformerWindowDataset | FinanceTransformerInferenceWindowDataset


def _full_date_loader(
    dataset: FinanceScoringDataset,
    *,
    shuffle: bool,
    seed: int,
    num_workers: int,
    minimum_group_size: int,
) -> tuple[DataLoader, FullDateBatchSampler]:
    sampler = FullDateBatchSampler(
        dataset.asof_dates,
        shuffle=shuffle,
        seed=seed,
        minimum_group_size=minimum_group_size,
    )
    return (
        DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        ),
        sampler,
    )


def _device_microbatches(
    full_date_batch: dict[str, torch.Tensor], *, batch_size: int
) -> list[dict[str, torch.Tensor]]:
    rows = int(full_date_batch["sample_index"].numel())
    if rows < 1 or batch_size < 1:
        raise ValueError("complete-date batch and physical batch size must be positive")
    chunk_count = (rows + batch_size - 1) // batch_size
    base_size, larger_chunks = divmod(rows, chunk_count)
    result: list[dict[str, torch.Tensor]] = []
    start = 0
    for chunk_index in range(chunk_count):
        size = base_size + (1 if chunk_index < larger_chunks else 0)
        stop = start + size
        result.append(
            {
                "values": full_date_batch["values"][start:stop],
                "observed_mask": full_date_batch["observed_mask"][start:stop],
                "sample_index": full_date_batch["sample_index"][start:stop],
            }
        )
        start = stop
    return result


def _market_tensors(
    dataset: FinanceScoringDataset,
    sample_indices: torch.Tensor,
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    window = dataset.market_window_for_sample_indices(sample_indices.cpu().numpy())
    values = (
        torch.from_numpy(window.values)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32, non_blocking=True)
    )
    observed = (
        torch.from_numpy(window.observed_mask)
        .unsqueeze(0)
        .to(device=device, dtype=torch.bool, non_blocking=True)
    )
    return values, observed


def predict_finance_transformer(
    model: FinancePatchTransformer,
    dataset: FinanceScoringDataset,
    *,
    batch_size: int,
    device: str,
    precision: str,
    num_workers: int,
) -> np.ndarray:
    loader, sampler = _full_date_loader(
        dataset,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
        minimum_group_size=1,
    )
    sampler.set_epoch(0)
    amp_enabled = device == "cuda" and precision == "fp16"
    primary_column = model.horizons.index(dataset.primary_horizon)
    predictions = np.empty(len(dataset), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for full_date_batch in loader:
            local_chunks: list[torch.Tensor] = []
            for microbatch in _device_microbatches(full_date_batch, batch_size=batch_size):
                values = microbatch["values"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                observed = microbatch["observed_mask"].to(
                    device=device, dtype=torch.bool, non_blocking=True
                )
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    local_chunks.append(model.encode_local(values, observed).embedding)
            market_values, market_observed = _market_tensors(
                dataset, full_date_batch["sample_index"], device=device
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                market = model.encode_market(market_values, market_observed)
                scores = model.score_date(torch.cat(local_chunks, dim=0), market).scores[
                    :, primary_column
                ]
            indices = full_date_batch["sample_index"].numpy()
            predictions[indices] = scores.detach().float().cpu().numpy()
    return predictions
