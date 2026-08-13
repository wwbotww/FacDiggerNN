from __future__ import annotations

import json
from datetime import date, datetime, timezone

import polars as pl
import pytest

from facdigger.data.contracts import DataContractError
from facdigger.inference.factor_batch import (
    FactorBatchInput,
    FactorBatchModel,
    FactorBatchSource,
    FactorBatchTime,
    build_factor_frame,
    factor_universe_sha256,
    load_factor_batch,
    publish_factor_batch,
    validate_factor_frame,
)

DIGEST = "a" * 64


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": ["eodhd:isin:US0000000001", "eodhd:isin:US0000000002"],
            "symbol": ["AAA", "BBB"],
            "asof_date": [date(2026, 8, 11), date(2026, 8, 11)],
            "score": [0.25, None],
            "eligible": [True, False],
        },
        schema={
            "security_id": pl.String,
            "symbol": pl.String,
            "asof_date": pl.Date,
            "score": pl.Float64,
            "eligible": pl.Boolean,
        },
    )


def _metadata() -> tuple[FactorBatchSource, FactorBatchModel, FactorBatchInput, FactorBatchTime]:
    return (
        FactorBatchSource(
            kind="signal_inference",
            repository="FacDiggerNN",
            commit="1" * 40,
            run_id="run-1",
            run_manifest_sha256=DIGEST,
        ),
        FactorBatchModel(
            release_id="b" * 64,
            model_id="model-1",
            model_type="financial_pretrained_patchtst",
            checkpoint_sha256="c" * 64,
            training_dataset_id="dataset-1",
            forecast_horizon_sessions=5,
            score_semantics="raw_cross_sectional_rank_score",
        ),
        FactorBatchInput(
            snapshot_id="inference-snapshot-1",
            snapshot_manifest_sha256="d" * 64,
            universe_semantics="complete_candidate_cross_section",
            universe_sha256=factor_universe_sha256(
                _frame().select("security_id", "symbol", "asof_date", "eligible")
            ),
            identity_policy="eodhd_isin_only",
        ),
        FactorBatchTime(
            calendar_version="2026.1",
            minimum_asof_date=date(2026, 8, 11),
            maximum_asof_date=date(2026, 8, 11),
        ),
    )


def _input_for_frame(
    input_metadata: FactorBatchInput, frame: pl.DataFrame
) -> FactorBatchInput:
    return input_metadata.model_copy(
        update={
            "universe_sha256": factor_universe_sha256(
                frame.select("security_id", "symbol", "asof_date", "eligible")
            )
        }
    )


def test_factor_batch_is_semantically_addressed_and_idempotent(tmp_path) -> None:
    source, model, input_metadata, time_metadata = _metadata()
    first, first_manifest = publish_factor_batch(
        _frame(),
        tmp_path,
        source=source,
        model=model,
        input_metadata=input_metadata,
        time_metadata=time_metadata,
        created_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
    )
    second, second_manifest = publish_factor_batch(
        _frame(),
        tmp_path,
        source=source,
        model=model,
        input_metadata=input_metadata,
        time_metadata=time_metadata,
        created_at=datetime(2026, 8, 13, tzinfo=timezone.utc),
    )

    assert first == second
    assert first_manifest.delivery_id == second_manifest.delivery_id == first.name
    assert "schema_version" not in json.loads(
        (first / "manifest.json").read_text(encoding="utf-8")
    )
    assert first_manifest.coverage.candidate_rows == 2
    assert first_manifest.coverage.scored_eligible_rows == 1
    assert set(path.name for path in first.iterdir()) == {
        "factors.parquet",
        "manifest.json",
    }
    assert load_factor_batch(first).delivery_id == first_manifest.delivery_id


def test_factor_batch_identity_changes_when_semantics_change(tmp_path) -> None:
    source, model, input_metadata, time_metadata = _metadata()
    first, _ = publish_factor_batch(
        _frame(),
        tmp_path,
        source=source,
        model=model,
        input_metadata=input_metadata,
        time_metadata=time_metadata,
    )
    second, _ = publish_factor_batch(
        _frame(),
        tmp_path,
        source=source,
        model=model.model_copy(update={"model_id": "model-2"}),
        input_metadata=input_metadata,
        time_metadata=time_metadata,
    )
    assert first != second


def test_factor_batch_rejects_noncanonical_scores_and_tampering(tmp_path) -> None:
    invalid = _frame().with_columns(pl.col("score").fill_null(1.0))
    with pytest.raises(DataContractError, match="ineligible factor scores"):
        validate_factor_frame(invalid)

    unsorted = _frame().reverse()
    with pytest.raises(DataContractError, match="sorted"):
        validate_factor_frame(unsorted)

    source, model, input_metadata, time_metadata = _metadata()
    bundle, _ = publish_factor_batch(
        _frame(),
        tmp_path,
        source=source,
        model=model,
        input_metadata=input_metadata,
        time_metadata=time_metadata,
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model"]["model_id"] = "substituted"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataContractError, match="semantic identity"):
        load_factor_batch(bundle)


def test_factor_batch_rejects_undeclared_files(tmp_path) -> None:
    source, model, input_metadata, time_metadata = _metadata()
    bundle, _ = publish_factor_batch(
        _frame(),
        tmp_path,
        source=source,
        model=model,
        input_metadata=input_metadata,
        time_metadata=time_metadata,
    )
    (bundle / ".extra").write_text("not part of the contract", encoding="utf-8")
    with pytest.raises(DataContractError, match="exactly two"):
        load_factor_batch(bundle)


def test_factor_batch_rejects_undeclared_directory(tmp_path) -> None:
    source, model, input_metadata, time_metadata = _metadata()
    bundle, _ = publish_factor_batch(
        _frame(),
        tmp_path,
        source=source,
        model=model,
        input_metadata=input_metadata,
        time_metadata=time_metadata,
    )
    (bundle / "undeclared").mkdir()
    with pytest.raises(DataContractError, match="exactly two"):
        load_factor_batch(bundle)


def test_factor_batch_rejects_source_semantics_mismatch(tmp_path) -> None:
    source, model, input_metadata, time_metadata = _metadata()
    with pytest.raises(ValueError, match="source kind and universe semantics"):
        publish_factor_batch(
            _frame(),
            tmp_path,
            source=source.model_copy(update={"kind": "evaluation_predictions"}),
            model=model,
            input_metadata=input_metadata,
            time_metadata=time_metadata,
        )


def test_factor_batch_source_kind_enforces_date_and_coverage_semantics(tmp_path) -> None:
    source, model, input_metadata, time_metadata = _metadata()
    multi_date = pl.concat(
        [
            _frame(),
            _frame().with_columns(pl.lit(date(2026, 8, 12)).alias("asof_date")),
        ]
    ).sort("asof_date", "security_id")
    with pytest.raises(ValueError, match="exactly one as-of date"):
        publish_factor_batch(
            multi_date,
            tmp_path,
            source=source,
            model=model,
            input_metadata=_input_for_frame(input_metadata, multi_date),
            time_metadata=time_metadata.model_copy(
                update={"maximum_asof_date": date(2026, 8, 12)}
            ),
        )

    evaluation_input = input_metadata.model_copy(
        update={"universe_semantics": "eligible_scored_cross_section"}
    )
    with pytest.raises(ValueError, match="only eligible scored rows"):
        publish_factor_batch(
            _frame(),
            tmp_path,
            source=source.model_copy(update={"kind": "evaluation_predictions"}),
            model=model,
            input_metadata=evaluation_input,
            time_metadata=time_metadata,
        )


def test_factor_frame_requires_every_eligible_candidate_score() -> None:
    candidates = _frame().select("security_id", "symbol", "asof_date", "eligible")
    scored = _frame().filter(pl.col("eligible")).select(
        "security_id", "asof_date", "score"
    )
    assert build_factor_frame(candidates, scored).equals(_frame(), null_equal=True)

    with pytest.raises(DataContractError, match="eligible candidate rows have no model score"):
        build_factor_frame(candidates, scored.head(0))


def test_factor_batch_allows_complete_no_signal_cross_section(tmp_path) -> None:
    frame = _frame().with_columns(
        pl.lit(None, dtype=pl.Float64).alias("score"),
        pl.lit(False).alias("eligible"),
    )
    source, model, input_metadata, time_metadata = _metadata()
    bundle, manifest = publish_factor_batch(
        frame,
        tmp_path,
        source=source,
        model=model,
        input_metadata=_input_for_frame(input_metadata, frame),
        time_metadata=time_metadata,
    )
    assert manifest.coverage.expected_eligible_rows == 0
    assert load_factor_batch(bundle).coverage.scored_eligible_rows == 0


def test_factor_batch_publish_failure_leaves_no_visible_or_temporary_bundle(
    tmp_path, monkeypatch
) -> None:
    source, model, input_metadata, time_metadata = _metadata()

    def fail_manifest_write(*args, **kwargs) -> None:
        raise OSError("simulated manifest write failure")

    monkeypatch.setattr(
        "facdigger.inference.factor_batch._write_json", fail_manifest_write
    )
    with pytest.raises(OSError, match="simulated manifest write failure"):
        publish_factor_batch(
            _frame(),
            tmp_path,
            source=source,
            model=model,
            input_metadata=input_metadata,
            time_metadata=time_metadata,
        )

    assert list(tmp_path.iterdir()) == []
