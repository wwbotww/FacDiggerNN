from __future__ import annotations

import torch

from facdigger.models.finance_patch_transformer import (
    FinancePatchTransformer,
    MultiScaleStatisticsEncoder,
)


def _model(*, dropout: float = 0.0) -> FinancePatchTransformer:
    return FinancePatchTransformer(
        context_length=32,
        num_local_channels=4,
        num_market_channels=2,
        num_asset_channels=2,
        horizons=(1, 5, 20),
        patch_length=8,
        patch_stride=4,
        local_d_model=16,
        local_num_attention_heads=4,
        local_num_hidden_layers=2,
        local_ffn_dim=32,
        market_d_model=8,
        market_num_attention_heads=2,
        market_num_hidden_layers=1,
        market_ffn_dim=16,
        embedding_dim=16,
        cross_num_attention_heads=4,
        cross_num_hidden_layers=2,
        cross_ffn_dim=32,
        statistics_output_dim=8,
        statistics_windows=(4, 8, 16, 32),
        dropout=dropout,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    values = torch.randn(6, 32, 4, generator=generator)
    observed = torch.ones_like(values, dtype=torch.bool)
    observed[0, :3, 0] = False
    values = values.masked_fill(~observed, 0.0)
    market_values = torch.randn(1, 32, 2, generator=generator)
    market_observed = torch.ones_like(market_values, dtype=torch.bool)
    return values, observed, market_values, market_observed


def test_finance_patch_transformer_outputs_all_horizons_and_backpropagates() -> None:
    model = _model()
    values, observed, market_values, market_observed = _inputs()

    output = model(values, observed, market_values, market_observed)

    assert output.local.embedding.shape == (6, 16)
    assert output.date_scores.scores.shape == (6, 3)
    assert torch.isfinite(output.date_scores.scores).all()
    assert torch.allclose(
        output.date_scores.context_gate,
        torch.full((3,), 0.1),
        atol=1e-6,
    )
    output.date_scores.scores.square().mean().backward()
    assert model.local_encoder.backbone.encoder.layers[0].self_attn.k_proj.weight.grad is not None
    assert model.cross_sectional_encoder.context_head.weight.grad is not None


def test_cross_sectional_scores_are_permutation_equivariant() -> None:
    model = _model().eval()
    values, observed, market_values, market_observed = _inputs()
    with torch.no_grad():
        local = model.encode_local(values, observed).embedding
        market = model.encode_market(market_values, market_observed)
        reference = model.score_date(local, market).scores
        permutation = torch.tensor([3, 0, 5, 1, 4, 2])
        permuted = model.score_date(local[permutation], market).scores

    torch.testing.assert_close(permuted, reference[permutation])


def test_market_context_changes_contextual_but_not_local_scores() -> None:
    model = _model().eval()
    values, observed, market_values, market_observed = _inputs()
    with torch.no_grad():
        local = model.encode_local(values, observed).embedding
        market = model.encode_market(market_values, market_observed)
        changed_market = model.encode_market(market_values + 2.0, market_observed)
        reference = model.score_date(local, market)
        changed = model.score_date(local, changed_market)

    torch.testing.assert_close(reference.local_scores, changed.local_scores)
    assert not torch.allclose(reference.contextual_scores, changed.contextual_scores)


def test_multiscale_statistics_ignore_unobserved_values() -> None:
    encoder = MultiScaleStatisticsEncoder(
        num_asset_channels=2,
        windows=(2, 4),
        output_dim=4,
        dropout=0.0,
    ).eval()
    values = torch.tensor(
        [[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]]
    )
    observed = torch.ones_like(values, dtype=torch.bool)
    observed[:, 1, 0] = False
    altered = values.clone()
    altered[:, 1, 0] = 1_000_000.0

    with torch.no_grad():
        reference = encoder(values, observed)
        changed = encoder(altered, observed)

    torch.testing.assert_close(reference, changed)


def test_local_encoding_is_independent_of_physical_microbatch_in_eval_mode() -> None:
    model = _model().eval()
    values, observed, _, _ = _inputs()
    with torch.no_grad():
        full = model.encode_local(values, observed).embedding
        chunked = torch.cat(
            [
                model.encode_local(value_chunk, observed_chunk).embedding
                for value_chunk, observed_chunk in zip(
                    values.split(2), observed.split(2), strict=True
                )
            ]
        )

    torch.testing.assert_close(full, chunked)
