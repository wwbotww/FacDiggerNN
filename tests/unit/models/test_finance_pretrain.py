from __future__ import annotations

import torch

from facdigger.models.finance_patch_transformer import FinancePatchTransformer
from facdigger.models.finance_pretrain import (
    FinanceNativePretrainer,
    contiguous_patch_element_mask,
)


def _pretrainer() -> FinanceNativePretrainer:
    supervised = FinancePatchTransformer(
        context_length=16,
        num_local_channels=4,
        num_market_channels=2,
        num_asset_channels=2,
        horizons=(1, 5),
        patch_length=4,
        patch_stride=2,
        local_d_model=8,
        local_num_attention_heads=2,
        local_num_hidden_layers=1,
        local_ffn_dim=16,
        market_d_model=8,
        market_num_attention_heads=2,
        market_num_hidden_layers=1,
        market_ffn_dim=16,
        embedding_dim=8,
        cross_num_attention_heads=2,
        cross_num_hidden_layers=1,
        cross_ffn_dim=16,
        statistics_output_dim=4,
        statistics_windows=(4, 8),
        dropout=0.0,
    )
    return FinanceNativePretrainer(
        local_encoder=supervised.local_encoder,
        market_encoder=supervised.market_encoder,
        context_length=16,
        num_local_channels=4,
        num_future_local_channels=2,
        num_market_channels=2,
        embedding_dim=8,
        reconstruction_weight=0.4,
        future_summary_weight=0.6,
        huber_delta=1.0,
        mask_ratio=0.3,
        minimum_span_patches=1,
        maximum_span_patches=3,
        shared_channel_fraction=0.5,
    )


def test_contiguous_patch_mask_combines_shared_and_independent_channels() -> None:
    torch.manual_seed(7)
    observed = torch.ones(2, 16, 4, dtype=torch.bool)
    mask, ratio = contiguous_patch_element_mask(
        observed,
        patch_length=4,
        patch_stride=2,
        sequence_start=0,
        mask_ratio=0.3,
        minimum_span_patches=1,
        maximum_span_patches=3,
        shared_channel_fraction=0.5,
    )

    assert mask.shape == observed.shape
    assert 0 < ratio < 1
    assert mask.any() and (~mask).any()
    assert any(
        not torch.equal(mask[batch, :, 0], mask[batch, :, 1])
        for batch in range(mask.shape[0])
    )


def test_finance_pretrainer_local_and_market_objectives_backpropagate() -> None:
    torch.manual_seed(11)
    model = _pretrainer()
    local = model.forward_local(
        torch.randn(3, 16, 4),
        torch.ones(3, 16, 4, dtype=torch.bool),
        torch.randn(3, 5, 2),
        torch.ones(3, 5, 2, dtype=torch.bool),
    )
    market = model.forward_market(
        torch.randn(2, 16, 2),
        torch.ones(2, 16, 2, dtype=torch.bool),
        torch.randn(2, 5, 2),
        torch.ones(2, 5, 2, dtype=torch.bool),
    )
    (local.loss + market.loss).backward()

    assert torch.isfinite(local.loss)
    assert torch.isfinite(market.loss)
    assert local.masked_observed_elements > 0
    assert market.observed_future_summary_elements > 0
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.local_encoder.parameters()
    )
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.market_encoder.parameters()
    )
