from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from test_finance_complete_experiments import last, setup_cells
from test_finance_interventions import intervention_plans
from test_finance_transformer_training import _config

from facdigger.data.contracts import DataContractError
from facdigger.models.finance_patch_transformer import (
    CrossSectionalEncoder,
    build_finance_transformer_model,
)
from facdigger.research import finance_comparison as comparison
from facdigger.research.finance_experiment_config import (
    experiment_cells,
    load_finance_experiment_config,
)
from facdigger.research.finance_experiment_state import protocol_hash, validate_checkpoint
from facdigger.research.finance_experiment_summary import summarize_experiments
from facdigger.training.finance_transformer_config import load_finance_transformer_config
from facdigger.training.finance_transformer_engine import _multi_horizon_loss
from facdigger.training.ranking import cross_sectional_rank_correlation_loss

ROOT = Path(__file__).resolve().parents[2]


def ablation_plans():
    plans = intervention_plans()
    for role, name in (('no_aux', 'no_aux'), ('no_stock', 'no_stock_interaction'),
                       ('epochs30', 'epochs30')):
        config = load_finance_experiment_config(ROOT / f"configs/research/finance_{name}.yaml")
        plans[role] = copy.deepcopy(plans['reference'])
        plans[role]['identity'].update(
            research=config.model_dump(mode='json'), cells=experiment_cells(config),
            supervised_config=load_finance_transformer_config(config.supervised_config).model_dump(
                mode='json'),
        )
    return plans


def test_ablation_matrix_preserves_original_comparison_and_v_gate(tmp_path, monkeypatch):
    old, extended = intervention_plans(), ablation_plans()
    comparison.validate_intervention_plans(old)
    comparison.validate_intervention_plans(extended)
    assert sum(len(p['identity']['cells']) for p in extended.values()) == 180
    assert len(comparison.ABLATION_METHOD_BINDINGS) == 12
    assert len(comparison.STATIC_COMPARISONS) == 20
    assert len(comparison.ABLATION_PAIRS) == 8
    monkeypatch.setattr(comparison, 'read_comparison', lambda *a: (
        {'identity': {'methods': comparison.ABLATION_METHOD_BINDINGS}}, extended,
    ))
    with pytest.raises(DataContractError, match='missing original package methods'):
        comparison.require_complete_validation_package(tmp_path)


@pytest.mark.parametrize('change', ['missing', 'scale', 'rank_weights', 'lr', 'short_clock',
                                   'mixed', 'support', 'candidates'])
def test_ablation_binding_rejects_undeclared_changes(change):
    plans = ablation_plans()
    arm = plans['no_aux']['identity']
    if change == 'missing':
        del plans['no_stock']
    elif change == 'scale':
        arm['supervised_config']['training']['objective']['scale_regularization'] = 0
    elif change == 'rank_weights':
        arm['supervised_config']['training']['objective']['horizon_weights'] = {5: 1}
    elif change == 'lr':
        arm['supervised_config']['training']['head_learning_rate'] *= .3
    elif change == 'short_clock':
        plans['epochs30']['identity']['supervised_config']['training']['minimum_epochs'] = 6
    elif change == 'mixed':
        plans['epochs30']['identity']['supervised_config']['research_ablation'] = (
            'no_auxiliary_rank'
        )
    elif change == 'support':
        arm['folds']['wf1']['data_protocol']['fixture'] = 'different'
    else:
        plans['no_stock']['identity']['research']['candidates'].append('statistics_linear')
    with pytest.raises(DataContractError):
        comparison.validate_intervention_plans(plans)


def test_default_payload_and_initialization_stay_identical():
    base = _config()
    payload = base.model_dump(mode='json')
    assert 'research_ablation' not in payload
    assert json.loads(base.model_dump_json()) == payload
    assert type(base).model_validate({**payload, 'research_ablation': None}).model_dump(
        mode='json') == payload
    models, states = [], []
    for variant in (None, 'no_auxiliary_rank', 'no_stock_interaction'):
        config = type(base).model_validate({**payload, 'research_ablation': variant})
        torch.manual_seed(42)
        models.append(build_finance_transformer_model(config, context_length=32))
        states.append(torch.get_rng_state())
        if variant is not None:
            assert protocol_hash(config, {}, 'finance') != protocol_hash(base, {}, 'finance')
    for model, rng in zip(models[1:], states[1:], strict=True):
        assert torch.equal(rng, states[0])
        assert model.state_dict().keys() == models[0].state_dict().keys()
        for key, tensor in model.state_dict().items():
            assert torch.equal(tensor, models[0].state_dict()[key])


def test_no_aux_gradient_is_primary_rank_plus_all_scale_terms():
    torch.manual_seed(24)
    scores = (torch.randn(17, 3) * torch.tensor([.2, 1.4, 3.0])).requires_grad_()
    targets = torch.randn(17, 3)
    weights = {1: .2, 5: .6, 20: .2}
    loss, audit = _multi_horizon_loss(
        scores, targets, horizons=(1, 5, 20), horizon_weights=weights,
        epsilon=1e-6, scale_regularization=.01, rank_horizons=(5,),
    )
    primary = .6 * cross_sectional_rank_correlation_loss(scores[:, 1], targets[:, 1], epsilon=1e-6)
    scale = .01 * sum(weights[h] * torch.log(scores[:, i].std(unbiased=False) + 1e-6).square()
                      for i, h in enumerate((1, 5, 20)))
    torch.testing.assert_close(loss, primary + scale)
    actual = torch.autograd.grad(loss, scores, retain_graph=True)[0]
    expected = torch.autograd.grad(primary + scale, scores)[0]
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[:, [0, 2]]) > 0  # Auxiliary scale is still optimized.
    changed = targets.clone()
    changed[:, [0, 2]] *= -1
    other, _ = _multi_horizon_loss(
        scores, changed, horizons=(1, 5, 20), horizon_weights=weights,
        epsilon=1e-6, scale_regularization=.01, rank_horizons=(5,),
    )
    torch.testing.assert_close(torch.autograd.grad(other, scores)[0], actual)
    assert all(f'rank_loss_{h}' in audit for h in (1, 5, 20))


def test_independent_context_equals_explicit_pairs_and_preserves_market_information():
    torch.manual_seed(17)
    torch.set_num_threads(1)
    model = CrossSectionalEncoder(d_model=8, num_attention_heads=2, num_hidden_layers=2,
                                  ffn_dim=16, dropout=0., num_horizons=3,
                                  stock_interaction=False).eval()
    stocks, market = torch.randn(7, 8, requires_grad=True), torch.randn(8, requires_grad=True)
    output = model(stocks, market)
    oracle = torch.cat([model(stocks[i:i+1], market).scores for i in range(7)])
    torch.testing.assert_close(output.scores, oracle, rtol=1e-5, atol=1e-6)
    expected_grad = torch.autograd.grad(oracle.sum(), (stocks, market), retain_graph=True)
    actual_grad = torch.autograd.grad(output.scores.sum(), (stocks, market), retain_graph=True)
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    single_grad = torch.autograd.grad(output.scores[0].sum(), stocks)[0]
    assert torch.count_nonzero(single_grad[1:]) == 0
    order = torch.tensor([4, 2, 6, 0, 1, 3, 5])
    torch.testing.assert_close(model(stocks[order], market).scores, output.scores[order])
    torch.testing.assert_close(model(stocks[:2], market).scores, output.scores[:2])
    appended = torch.cat([stocks, torch.randn(5, 8) * 10])
    torch.testing.assert_close(model(appended, market).scores[:7], output.scores)
    assert not torch.allclose(model(stocks, -market).scores, output.scores)
    # The explicit reference uses the original encoder on each two-token sequence.
    original = copy.deepcopy(model)
    original.stock_interaction = True
    expected = torch.cat([original(stocks[i:i+1], market).scores for i in range(7)])
    torch.testing.assert_close(expected, output.scores, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('phase', ['update', 'observation', 'final_commit'])
@pytest.mark.parametrize('candidate,ablation,epochs', [
    ('statistics_mlp_dropout03', 'no_auxiliary_rank', 3),
    ('finance', 'no_auxiliary_rank', 3),
    ('finance', 'no_stock_interaction', 3),
    ('statistics_mlp_dropout03', None, 30),
    ('finance', None, 30),
])
def test_ablation_resume(tmp_path, monkeypatch, phase, candidate, ablation, epochs):
    from test_finance_complete_experiments import (
        test_complete_cell_resumes_without_changing_optimizer_clock_or_rng as check_resume,
    )
    check_resume(tmp_path, monkeypatch, phase, candidate, 1, ablation, epochs)
    checkpoint = torch.load(last(tmp_path/'continuous', candidate), weights_only=False)
    assert checkpoint['epoch'] == epochs
    assert checkpoint['config'].get('research_ablation') == ablation
    if ablation:
        original = type(_config()).model_validate(
            {**checkpoint['config'], 'research_ablation': None}
        )
        with pytest.raises(DataContractError, match='protocol differs'):
            validate_checkpoint(checkpoint, original, checkpoint['data_protocol'], candidate)
    if epochs == 30:
        assert checkpoint['scheduler_state']['last_epoch'] == checkpoint['global_step']
        config = copy.deepcopy(checkpoint['config'])
        config['training'].update(max_epochs=10, minimum_epochs=6)
        with pytest.raises(DataContractError, match='protocol differs'):
            validate_checkpoint(checkpoint, type(_config()).model_validate(config),
                                checkpoint['data_protocol'], candidate)


def test_no_aux_observation_keeps_original_full_reference_loss(tmp_path, monkeypatch):
    records = []
    for name, ablation in [('base', None), ('no_aux', 'no_auxiliary_rank')]:
        folder = tmp_path/name
        folder.mkdir()
        setup = setup_cells(folder, monkeypatch, 'statistics_mlp_dropout03', ablation=ablation)
        root = setup['plan']('run')
        setup['run'](root)
        result = summarize_experiments({name: root}, folder/'summary', repository_root=tmp_path)
        assert not result['validation_used'] and not result['holdout_used']
        import polars as pl
        records.append(pl.read_csv(folder/'summary/trajectory.csv').filter(pl.col('epoch') == 0))
    for field in ('rank_ic', 'raw_ic', 'surrogate_correlation', 'weighted_available_loss'):
        assert records[0][field].to_list() == records[1][field].to_list()
