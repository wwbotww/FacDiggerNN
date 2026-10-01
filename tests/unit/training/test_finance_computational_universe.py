from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch

from facdigger.data.contracts import DataContractError
from facdigger.datasets.index import align_labelled_samples
from facdigger.inference.runner import _predict_factor_model
from facdigger.training.common import load_snapshot_inference_rows
from facdigger.training.finance_transformer_engine import (
    _supervised_target_lookup,
    evaluate_finance_transformer_selection,
)


def _rows() -> pl.DataFrame:
    days = [date(2024, 1, 10), date(2024, 1, 11)]
    return pl.DataFrame([
        {
            "sample_id": f"{security}|{day}", "security_id": security, "symbol": security,
            "asof_date": day, "feature_start": day - timedelta(days=5),
            "feature_end": day, "eligible": True,
        }
        for day in days for security in ("A", "B", "C")
    ])


def _labels(rows: pl.DataFrame) -> pl.DataFrame:
    return rows.filter(pl.col("security_id") != "B").with_columns(
        pl.when(pl.col("security_id") == "A").then(-0.2).otherwise(0.3).alias("target"),
        pl.lit("valid").alias("split"),
    ).with_columns(*[pl.col("target").alias(f"target_{h}") for h in (1, 5, 20)])


def test_label_alignment_preserves_requested_order_and_rejects_invalid_identities() -> None:
    computational = _rows().reverse()
    labelled = _labels(_rows())
    aligned = align_labelled_samples(computational, labelled)
    assert aligned["_computational_row"].to_list() == [5, 3, 2, 0]
    assert aligned.select(labelled.columns).equals(labelled)
    with pytest.raises(DataContractError, match="duplicate"):
        align_labelled_samples(computational, pl.concat([labelled, labelled.head(1)]))
    with pytest.raises(DataContractError, match="duplicate"):
        align_labelled_samples(pl.concat([computational, computational.head(1)]), labelled)
    with pytest.raises(DataContractError, match="absent"):
        align_labelled_samples(computational.filter(pl.col("security_id") != "A"), labelled)
    for field in ("feature_start", "feature_end", "sample_id"):
        invalid = labelled.with_columns(
            (pl.col(field) + timedelta(days=1) if field != "sample_id" else pl.lit("wrong"))
            .alias(field)
        )
        with pytest.raises(DataContractError, match="disagree|duplicate"):
            align_labelled_samples(computational, invalid)


def test_target_lookup_ranks_only_labels_and_enforces_labelled_minimum() -> None:
    rows = _rows()
    labelled = _labels(rows).reverse()
    dataset = SimpleNamespace(sample_rows=rows, asof_dates=rows["asof_date"].to_list())
    # The engine only needs len(dataset), rows and dates for this contract.
    class Dataset:
        sample_rows = dataset.sample_rows
        asof_dates = dataset.asof_dates

        def __len__(self):
            return self.sample_rows.height

    config = SimpleNamespace(
        horizons=[1, 5, 20],
        training=SimpleNamespace(objective=SimpleNamespace(minimum_cross_section_size=2)),
    )
    ranks, mapping = _supervised_target_lookup(config, Dataset(), labelled)
    assert mapping.tolist() == [3, -1, 2, 1, -1, 0]
    torch.testing.assert_close(ranks[:, 0], torch.tensor([1.0, -1.0, 1.0, -1.0]))
    config.training.objective.minimum_cross_section_size = 3
    with pytest.raises(ValueError, match="smaller than minimum"):
        _supervised_target_lookup(config, Dataset(), labelled)


def test_selection_scores_complete_pool_but_measures_only_labelled_rows(monkeypatch) -> None:
    rows = _rows()
    labelled = _labels(rows).reverse()

    class Dataset:
        sample_rows = rows
        asof_dates = rows["asof_date"].to_list()

        def __len__(self):
            return self.sample_rows.height

    calls = []

    def predict(model, dataset, **kwargs):
        calls.append(dataset.sample_rows)
        return np.asarray([-1.0, 9999.0, 1.0, -1.0, -9999.0, 1.0])

    monkeypatch.setattr(
        "facdigger.training.finance_transformer_engine.predict_finance_transformer", predict
    )
    audit = evaluate_finance_transformer_selection(
        None, Dataset(), labelled_rows=labelled, batch_size=2, device="cpu", precision="fp32",
        num_workers=0, minimum_cross_section_size=2, minimum_dates=2, minimum_coverage=1.0,
        subperiods=2, stability_penalty=0.2,
    )
    assert calls[0].equals(rows)
    assert audit["mean_rank_ic"] == pytest.approx(1.0)
    assert audit["score_std"] == 1.0
    assert audit["computational_rows"] == 6
    assert audit["labelled_rows"] == 4


def test_full_pool_loader_is_target_free_and_restricts_dates(tmp_path) -> None:
    rows = _rows()
    # The projection must never consume these unrelated columns.
    rows.with_columns(pl.lit("unread").alias("target")).write_parquet(tmp_path / "index.parquet")
    manifest = {"artifacts": {"inference_index": "index.parquet"}}
    selected = load_snapshot_inference_rows(
        tmp_path, manifest, asof_dates=[date(2024, 1, 10)]
    )
    assert selected.height == 3
    assert "target" not in selected.columns
    assert selected["security_id"].to_list() == ["A", "B", "C"]
    with pytest.raises(DataContractError, match="missing requested dates"):
        load_snapshot_inference_rows(tmp_path, manifest, asof_dates=[date(2024, 2, 1)])


@pytest.mark.parametrize("new_protocol,damage", [
    (False, None), (True, None), (True, "checkpoint_missing"),
    (False, "manifest_missing"), (True, "conflict"), (False, "unknown_contract"),
])
def test_historical_replay_chooses_recorded_universe_and_projects_keys(
    tmp_path, monkeypatch, new_protocol, damage
) -> None:
    rows = _rows()
    labels = _labels(rows)
    rows.write_parquet(tmp_path / "inference.parquet")
    seen = []

    protocol = {"computational_universe": "target_free_inference_index"}
    checkpoint_audit = {"checkpoint_contract": "finance_patch_transformer_checkpoint"}
    if new_protocol:
        checkpoint_audit["checkpoint_data_protocol"] = protocol
    if damage == "checkpoint_missing":
        checkpoint_audit.pop("checkpoint_data_protocol")
    elif damage == "manifest_missing":
        checkpoint_audit["checkpoint_data_protocol"] = protocol
    elif damage == "conflict":
        checkpoint_audit["checkpoint_data_protocol"] = {**protocol, "selection_plan": "wrong"}
    elif damage == "unknown_contract":
        checkpoint_audit["checkpoint_contract"] = "unknown"

    class Backend:
        audit = checkpoint_audit

        def predict(self, path, manifest, index):
            seen.append(index)
            # Mimic a cross-sectional effect: labelled scores also depend on pool size.
            return index.select("security_id", "asof_date").with_columns(
                (pl.col("security_id").replace_strict({"A": 1.0, "B": 2.0, "C": 3.0})
                 + index.height).alias("score")
            )

    monkeypatch.setattr(
        "facdigger.inference.runner.load_checkpoint_backend", lambda *a, **k: Backend()
    )
    manifest = {"model_type": "finance_patch_transformer", "dataset_id": "fixture"}
    if new_protocol:
        manifest["data_protocol"] = {"computational_universe": "target_free_inference_index"}
    kwargs = dict(
        manifest=manifest, config_payload={}, checkpoint_path=tmp_path / "unused.pt",
        dataset_path=tmp_path,
        dataset_manifest={
            "config": {"features": {"context_length": 6}},
            "artifacts": {"inference_index": "inference.parquet"},
        },
        frames={"sample_index": labels}, split="valid", device_preference="cpu",
    )
    if damage is not None:
        with pytest.raises(DataContractError, match="protocols differ|unknown.*contract"):
            _predict_factor_model(**kwargs)
        return
    scores, returned, _ = _predict_factor_model(**kwargs)
    assert returned.equals(labels)
    assert seen[0].height == (6 if new_protocol else 4)
    np.testing.assert_array_equal(scores, np.asarray([1, 3, 1, 3]) + seen[0].height)
