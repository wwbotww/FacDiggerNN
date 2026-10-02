"""Zero-update, fixed-state loss geometry on one original full computational date."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import torch

from facdigger.data.contracts import DataContractError
from facdigger.models.finance_scoring import _full_date_loader
from facdigger.models.finance_statistics import FinanceStatisticsRanker
from facdigger.training.e1_engine import _restore_rng_state, _rng_state
from facdigger.training.finance_transformer_engine import (
    _supervised_target_lookup,
    backward_complete_date_with_embedding_replay,
)


def _shared_parameter(name: str, model) -> bool:
    if isinstance(model, FinanceStatisticsRanker):
        return name.startswith(("head.0.", "head.3."))
    return not name.startswith(
        (
            "cross_sectional_encoder.local_head.",
            "cross_sectional_encoder.context_head.",
            "cross_sectional_encoder.context_gate_logits",
        )
    )


def loss_gradient_diagnostics(model, dataset, labelled, config, *, check_stop=None) -> dict:
    """Differentiate four weighted terms and their original sum; never construct an optimizer.

    AMP uses a fixed scale of 1024 for every term, then unscales into float64 on CPU.
    The additivity residual is relative to the sum of term norms, avoiding division
    by a nearly cancelled total. This new numerical check does not change replay tolerances.
    """
    if dataset.sample_rows["asof_date"].n_unique() != 1:
        raise DataContractError("gradient diagnostic requires exactly one complete date")
    if isinstance(model, FinanceStatisticsRanker) and model.kind != "statistics_mlp":
        raise DataContractError("gradient diagnostic is registered only for Finance and MLP")
    rng = _rng_state()
    modes = [(module, module.training) for module in model.modules()]
    parameters = list(model.named_parameters())
    old_gradients = [None if p.grad is None else p.grad.detach().clone() for _, p in parameters]
    original = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    device = str(next(model.parameters()).device)
    amp = device.startswith("cuda") and config.training.precision == "fp16"
    vectors, losses, audits = {}, {}, {}
    objective = config.training.objective
    try:
        model.eval()
        lookup, projection = _supervised_target_lookup(config, dataset, labelled)
        loader, _ = _full_date_loader(
            dataset,
            shuffle=False,
            seed=0,
            num_workers=0,
            minimum_group_size=1,
        )
        batch = next(iter(loader))
        for component in (*config.horizons, "scale", None):
            if check_stop:
                check_stop()
            _restore_rng_state(rng)
            model.zero_grad(set_to_none=True)
            scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)
            audit = backward_complete_date_with_embedding_replay(
                model,
                dataset,
                batch,
                device=device,
                target_rank_lookup=lookup,
                row_to_label_index=projection,
                horizon_weights=objective.horizon_weights,
                epsilon=objective.epsilon,
                scale_regularization=objective.scale_regularization,
                amp_enabled=amp,
                scaler=scaler,
                physical_microbatch_size=config.training.batch_size,
                dates_in_optimizer_step=1,
                relative_tolerance=config.training.replay_relative_tolerance,
                absolute_tolerance=config.training.replay_absolute_tolerance,
                diagnostic_component=component,
            )
            name = "total" if component is None else str(component)
            vector = torch.cat(
                [
                    (torch.zeros_like(p) if p.grad is None else p.grad.detach())
                    .flatten()
                    .cpu()
                    .double()
                    / scaler.get_scale()
                    for _, p in parameters
                ]
            ).numpy()
            if not np.isfinite(vector).all():
                raise DataContractError(f"non-finite diagnostic gradient: {name}")
            vectors[name] = vector
            losses[name] = (
                audit["loss"]
                if component is None
                else objective.scale_regularization * audit["scale_penalty"]
                if component == "scale"
                else objective.horizon_weights[component] * audit[f"rank_loss_{component}"]
            )
            audits[name] = audit
        names = [str(h) for h in config.horizons] + ["scale"]
        denominator = sum(float(np.linalg.norm(vectors[name])) for name in names)
        residual = float(np.linalg.norm(sum(vectors[name] for name in names) - vectors["total"]))
        relative = residual / max(denominator, np.finfo(float).tiny)
        tolerance = 5e-3 if amp else 2e-5
        if not np.isfinite(relative) or relative > tolerance:
            raise DataContractError(
                f"diagnostic gradient sum differs: relative_l2={relative}, limit={tolerance}"
            )
        groups = {
            "all": np.ones(len(vectors["total"]), dtype=bool),
            "shared": np.concatenate(
                [np.full(p.numel(), _shared_parameter(name, model)) for name, p in parameters]
            ),
        }
        for family in sorted({name.split(".", 1)[0] for name, _ in parameters}):
            groups[f"module:{family}"] = np.concatenate(
                [np.full(p.numel(), name.split(".", 1)[0] == family) for name, p in parameters]
            )
        geometry = {}
        for group, mask in groups.items():
            selected = {key: value[mask] for key, value in vectors.items()}
            norms = {key: float(np.linalg.norm(value)) for key, value in selected.items()}
            total_norms = sum(norms[name] for name in names)
            geometry[group] = {
                "parameters": int(mask.sum()),
                "weighted_norms": norms,
                "norm_fractions": {
                    name: norms[name] / total_norms if total_norms > 0 else None for name in names
                },
                "cosines": {
                    f"{left}:{right}": (
                        float(
                            np.dot(selected[left], selected[right]) / (norms[left] * norms[right])
                        )
                        if norms[left] > 0 and norms[right] > 0
                        else None
                    )
                    for left, right in combinations(names, 2)
                },
            }
        return {
            "asof_date": str(dataset.sample_rows["asof_date"][0]),
            "computational_rows": len(dataset),
            "labelled_rows": labelled.height,
            "mode": "eval",
            "optimizer_updates": 0,
            "amp_scale": 1024.0 if amp else 1.0,
            "effective_precision": "fp16" if amp else "fp32",
            "weighted_component_losses": losses,
            "term_coefficients": {
                **{str(h): objective.horizon_weights[h] for h in config.horizons},
                "scale": objective.scale_regularization,
            },
            "geometry": geometry,
            "additivity": {"relative_l2": relative, "tolerance": tolerance, "passed": True},
            "replay_audits": audits,
        }
    finally:
        changed = any(
            not torch.equal(original[key], value.detach().cpu())
            for key, value in model.state_dict().items()
        )
        if changed:
            model.load_state_dict(original, strict=True)
        for (_, parameter), gradient in zip(parameters, old_gradients, strict=True):
            parameter.grad = gradient
        for module, training in modes:
            module.training = training
        _restore_rng_state(rng)
        if changed:
            raise DataContractError("zero-update diagnostic unexpectedly changed model state")
