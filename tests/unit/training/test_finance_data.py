from __future__ import annotations

import copy
import json
from pathlib import Path

import polars as pl
import pytest

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import build_dataset_snapshot
from facdigger.training.finance_data import (
    finance_data_protocol,
    load_finance_selection,
    validate_encoder_data_protocol,
)
from facdigger.training.finance_pretrain_config import FinancePretrainingExperimentConfig
from facdigger.training.finance_transformer_config import FinanceTransformerExperimentConfig
from facdigger.training.run_state import training_run
from facdigger.training.runtime import TrainingControl
from tests.integration.test_finance_selection_snapshot import finance_config


@pytest.fixture
def finance_snapshot(tmp_path):
    config = finance_config(tmp_path)
    snapshot, manifest = build_dataset_snapshot(config)
    return snapshot, manifest


def experiment():
    return FinanceTransformerExperimentConfig.model_validate(
        {
            "model": {"statistics_windows": [5, 20]},
        }
    )


@pytest.mark.parametrize(
    "damage", ["plan", "local_scaler", "market_scaler", "ssl_future", "expectation"]
)
def test_actual_finance_inputs_are_validated_before_persisting_run(
    finance_snapshot, tmp_path, damage
):
    snapshot, manifest = finance_snapshot
    config = experiment()
    plan_path = snapshot / "finance_selection_plan.json"
    plan = json.loads(plan_path.read_text())
    if damage == "plan":
        changed = dict(plan, a_end=plan["p_start"])
        plan_path.write_text(json.dumps(changed))
    elif damage.endswith("scaler"):
        path = snapshot / "scaler.json"
        payload = json.loads(path.read_text())
        payload[damage.removesuffix("_scaler")]["fit_end"] = plan["p_start"]
        path.write_text(json.dumps(payload))
    elif damage == "ssl_future":
        path = snapshot / "pretraining_index.parquet"
        data = pl.read_parquet(path)
        data.with_columns(pl.lit(plan["p_start"]).str.to_date().alias("future_end")).write_parquet(
            path
        )
    else:
        config = config.model_copy(update={"selection_fraction": 0.2})
    run = tmp_path / "run"
    with pytest.raises(DataContractError):
        with training_run(
            config,
            snapshot,
            repository_root=Path(__file__).resolve().parents[3],
            model_type="finance_patch_transformer",
            control=TrainingControl(),
            resume_from=None,
            run_dir=run,
            data_protocol_loader=lambda dataset, saved: finance_data_protocol(
                dataset, saved, config
            ),
        ):
            pytest.fail("invalid inputs cannot enter training")
    assert not (run / "manifest.json").exists()
    assert not (run / "resolved_config.yaml").exists()
    assert not (run / "progress.jsonl").exists()
    assert not (run / "checkpoints").exists()


def test_shared_encoder_binding_accepts_stage_specific_universes(finance_snapshot):
    snapshot, manifest = finance_snapshot
    supervised, probe, plan = load_finance_selection(snapshot, manifest)
    assert (
        supervised.filter(pl.col("split") == "train_fit").height == plan["supervised"]["fit_rows"]
    )
    assert probe.filter(pl.col("split") == "probe_fit").height == plan["probe"]["fit_rows"]
    model_config = experiment()
    ssl_config = FinancePretrainingExperimentConfig.model_validate(
        {
            "model": {"statistics_windows": [5, 20]},
            "training": {"probe": {"fit_dates": 5, "selection_dates": 4}},
        }
    )
    expected = finance_data_protocol(snapshot, manifest, model_config)
    encoder = finance_data_protocol(snapshot, manifest, ssl_config)
    assert encoder["computational_universe"] != expected["computational_universe"]
    validate_encoder_data_protocol({"data_protocol": encoder}, expected)
    for field in ("dataset_id", "selection_plan", "feature_scaler_sha256", "label_support"):
        changed = copy.deepcopy(encoder)
        changed[field] = "incorrect"
        with pytest.raises(DataContractError, match="protocol"):
            validate_encoder_data_protocol({"data_protocol": changed}, expected)
