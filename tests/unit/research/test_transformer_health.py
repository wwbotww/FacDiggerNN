from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from facdigger.data.contracts import DataContractError
from facdigger.data.snapshots import sha256_file
from facdigger.experiments.manifest import sha256_json
from facdigger.research.transformer_config import load_transformer_comparison_config
from facdigger.research.transformer_health import audit_transformer_health, training_health
from facdigger.training.runtime import write_json


def history():
    return [
        {
            "epoch": i + 1,
            "global_step": (i + 1) * 100,
            "gradient_clip_ratio": 0.01,
            "score_std_1": score,
            "score_std_5": score,
            "score_std_20": score,
            "amp_skipped_optimizer_steps": 0,
        }
        for i, score in enumerate([1, 2, 1, 1, 2, 1])
    ]


def test_health_weights_successful_updates_and_uses_strict_threshold():
    rows = history()
    # Epoch four: 1 / 100 clipped; five: 1 / 10; six: 0 / 1. Weighted < 10%.
    rows[3].update(global_step=400, gradient_clip_ratio=0.01)
    rows[4].update(global_step=410, gradient_clip_ratio=0.1)
    rows[5].update(global_step=411, gradient_clip_ratio=0)
    report = training_health({"history": rows})
    assert report["epochs"] == [4, 5, 6]
    assert report["gradient_clip_ratio"] == pytest.approx(2 / 111)
    assert report["status"] == "passed"
    for index, row in enumerate(rows, start=1):
        row["global_step"] = index * 100
        row["gradient_clip_ratio"] = 0.1
    assert training_health({"history": rows})["status"] == "failed"


def test_health_requires_enough_evidence_and_flags_monotonic_primary_scores():
    rows = history()
    for row in rows:
        row["score_std_5"] = float(row["epoch"])
    report = training_health({"history": rows})
    assert report["status"] == "failed"
    assert report["score_std_strictly_increasing"]["5"]
    assert training_health({"history": rows[:4]})["status"] == "insufficient_history"
    rows[5]["score_std_5"] = float("nan")
    assert training_health({"history": rows})["status"] == "insufficient_history"
    assert training_health({})["status"] == "insufficient_history"


def test_frozen_matrix_audit_verifies_all_children_and_does_not_rewrite_acceptance(tmp_path):
    root = tmp_path / "comparison"
    root.mkdir()
    repository = Path(__file__).resolve().parents[3]
    config = load_transformer_comparison_config(
        repository / "configs/research/finance_transformer_streamlined.yaml"
    )
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json")))
    records = []
    for fold in config.folds:
        for stage in ("pretraining", "scratch", "finance_pretrained"):
            child = root / "runs" / fold.fold_id / stage
            child.mkdir(parents=True)
            checkpoint = child / "best.pt"
            checkpoint.write_bytes(b"fixture checkpoint")
            write_json(
                child / "manifest.json",
                {
                    "status": "complete",
                    "training": {"history": history()},
                    "checkpoint": {"file": "best.pt", "sha256": sha256_file(checkpoint)},
                },
            )
            records.append(
                {
                    "fold_id": fold.fold_id,
                    "stage": stage,
                    "status": "complete",
                    "run_dir": str(child),
                    "manifest_sha256": sha256_file(child / "manifest.json"),
                }
            )
    write_json(root / "matrix.json", {"stages": records})
    write_json(root / "comparison.json", {"acceptance": {"status": "go"}})
    write_json(
        root / "manifest.json",
        {
            "status": "complete",
            "config_hash": sha256_json(config.model_dump(mode="json")),
            "comparison_sha256": sha256_file(root / "comparison.json"),
        },
    )
    original = (root / "comparison.json").read_bytes()
    report = audit_transformer_health(root)
    assert report["health_checks_status"] == "passed"
    assert report["requires_manual_review"] is True and len(report["cells"]) == 6
    assert (root / "comparison.json").read_bytes() == original
    (Path(records[0]["run_dir"]) / "best.pt").write_bytes(b"damaged pretraining artifact")
    with pytest.raises(DataContractError, match="artifact changed"):
        audit_transformer_health(root)
    # Omitting records must not silently make an incomplete matrix pass.
    write_json(root / "matrix.json", {"stages": records[1:]})
    with pytest.raises(DataContractError, match="nine stages"):
        audit_transformer_health(root)
    assert json.loads((root / "comparison.json").read_text())["acceptance"]["status"] == "go"
