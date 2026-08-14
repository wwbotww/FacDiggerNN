from __future__ import annotations

import json
from datetime import date

import polars as pl
import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("transformers")

from facdigger.data.contracts import DataContractError  # noqa: E402
from facdigger.experiments.manifest import sha256_json  # noqa: E402
from facdigger.inference.runner import _load_source_run, _select_signal_inputs  # noqa: E402
from facdigger.inference.scoring import patch_config  # noqa: E402


def _model() -> dict:
    return {
        "patch_length": 4,
        "patch_stride": 4,
        "d_model": 8,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "ffn_dim": 16,
        "dropout": 0.0,
        "norm_type": "layernorm",
        "alpha_hidden_dim": 8,
        "alpha_dropout": 0.0,
    }


def test_patch_config_dispatches_all_e1_e2_e3_manifests() -> None:
    base = {
        "channels": ["x0", "x1"],
        "model": _model(),
    }
    e1, e1_training = patch_config(
        "random_patchtst",
        {
            **base,
            "training": {"max_epochs": 1, "minimum_epochs": 1},
        },
    )
    e2, e2_training = patch_config(
        "etth1_transferred_patchtst",
        {
            **base,
            "training": {
                "max_epochs": 2,
                "minimum_epochs": 2,
                "head_only_epochs": 1,
            },
        },
    )
    e3, e3_training = patch_config(
        "financial_pretrained_patchtst",
        {
            **base,
            "pretraining": {"max_epochs": 1, "minimum_epochs": 1},
            "finetuning": {
                "max_epochs": 2,
                "minimum_epochs": 2,
                "head_only_epochs": 1,
            },
        },
    )

    assert e1.model.d_model == e2.model.d_model == e3.model.d_model == 8
    assert e1_training.max_epochs == 1
    assert e2_training.head_only_epochs == e3_training.head_only_epochs == 1


def test_source_run_loader_rejects_checkpoint_tampering_and_path_escape(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = {"experiment_id": "test"}
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    checkpoint = run_dir / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = {
        "status": "complete",
        "model_type": "mlp",
        "config_hash": sha256_json(config),
        "checkpoint": {"file": "best.pt", "sha256": "not-the-real-hash"},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataContractError, match="checkpoint hash"):
        _load_source_run(run_dir)

    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    manifest["checkpoint"]["file"] = "../outside.pt"
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataContractError, match="escapes"):
        _load_source_run(run_dir)


def test_latest_signal_date_uses_candidate_universe_without_stale_fallback() -> None:
    first = date(2026, 8, 10)
    latest = date(2026, 8, 11)
    index = pl.DataFrame(
        {"security_id": ["sec-a"], "asof_date": [first]},
        schema={"security_id": pl.String, "asof_date": pl.Date},
    )
    universe = pl.DataFrame(
        {
            "security_id": ["sec-a", "sec-a"],
            "symbol": ["AAA", "AAA"],
            "asof_date": [first, latest],
            "eligible": [True, False],
        },
        schema={
            "security_id": pl.String,
            "symbol": pl.String,
            "asof_date": pl.Date,
            "eligible": pl.Boolean,
        },
    )

    rows, candidates = _select_signal_inputs(index, universe, asof="latest")

    assert rows.is_empty()
    assert candidates["asof_date"].to_list() == [latest]
    assert candidates["eligible"].to_list() == [False]
    with pytest.raises(DataContractError, match="absent from the delivery universe"):
        _select_signal_inputs(index, universe, asof="2026-08-12")
