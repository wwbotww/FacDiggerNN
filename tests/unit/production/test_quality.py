from datetime import date, timedelta

import polars as pl
import pytest

from facdigger.data.contracts import DataContractError
from facdigger.inference.delivery import DeliveryConfig
from facdigger.production.config import ProductionInferenceConfig, ProductionQualityConfig
from facdigger.production.quality import (
    assess_daily_quality,
    assess_market_quality,
    build_quality_reference,
)

DAY = date(2026, 8, 17)


def _delivery(count=5):
    return DeliveryConfig.model_validate({
        "targets": [{"instrument_id": f"S{i}.US"} for i in range(count)],
        "identities": [{
            "instrument_id": f"S{i}.US", "security_id": f"sec-{i}",
            "valid_from": "2026-01-01", "valid_to": "2026-12-31", "evidence": "Fixture",
        } for i in range(count)],
    })


def _candidates(count=20, missing=()):
    return pl.DataFrame({
        "security_id": [f"sec-{i}" for i in range(count)],
        "symbol": [f"S{i}" for i in range(count)],
        "asof_date": [DAY] * count,
        "eligible": [i not in missing for i in range(count)],
    }).sort("asof_date", "security_id")


def _reference(count=20):
    return {
        "target_date": DAY.isoformat(), "reference_date": "2026-08-14",
        "security_ids": [f"sec-{i}" for i in range(count)],
        "market_eligible_rows": {"2026-08-14": count},
    }


def _report(count=20, missing=(), **kwargs):
    return assess_daily_quality(
        _candidates(count, missing), reference=_reference(count), delivery=_delivery(),
        inference=ProductionInferenceConfig(minimum_candidate_rows=10, minimum_eligible_rows=10),
        policy=ProductionQualityConfig(), stage="inference", unscorable=[{
            "security_id": f"sec-{i}", "symbol": f"S{i}", "asof_date": DAY.isoformat(),
            "reason": "missing_target_bar",
        } for i in missing], **kwargs,
    )


def test_local_missing_is_degraded_not_incomplete_delivery():
    report = _report(missing=[0])
    assert report["status"] == "degraded"
    assert report["violations"] == []
    assert report["computation"]["missing_fraction"] == 0.05
    assert report["delivery"]["candidate_rows"] == 5
    assert report["delivery"]["eligible_rows"] == 4
    assert report["delivery"]["unscorable_source_ids"] == ["sec-0"]


def test_computational_and_delivery_gates_are_independent():
    broad = _report(missing=[18, 19])
    assert broad["delivery"]["unscorable_fraction"] == 0
    assert broad["violations"] == ["computational_missing_fraction_exceeded"]
    narrow = _report(count=100, missing=[0, 1])
    assert narrow["computation"]["missing_fraction"] == 0.02
    assert narrow["violations"] == ["delivery_unscorable_fraction_exceeded"]


def test_replacement_stocks_cannot_hide_missing_reference_members():
    candidates = _candidates().with_columns(
        pl.col("security_id").replace({"sec-18": "new-18", "sec-19": "new-19"})
    )
    report = assess_daily_quality(
        candidates, reference=_reference(), delivery=_delivery(),
        inference=ProductionInferenceConfig(minimum_candidate_rows=10, minimum_eligible_rows=10),
        policy=ProductionQualityConfig(), stage="inference", unscorable=[],
    )
    assert report["computation"]["eligible_rows"] == 20
    assert report["computation"]["missing_fraction"] == 0.1
    assert report["status"] == "insufficient"


def test_market_failure_blocks_even_when_all_stocks_have_scores():
    report = _report(market={"violations": ["target_market_channels_unavailable"]})
    assert report["computation"]["missing_fraction"] == 0
    assert report["status"] == "insufficient"
    assert report["violations"] == ["target_market_channels_unavailable"]


def test_quarantined_members_still_count_when_new_stocks_replace_them():
    candidates = _candidates(22, missing=[18, 19])
    report = assess_daily_quality(
        candidates, reference=_reference(), delivery=_delivery(),
        inference=ProductionInferenceConfig(minimum_candidate_rows=10, minimum_eligible_rows=10),
        policy=ProductionQualityConfig(), stage="inference", unscorable=[{
            "security_id": f"sec-{i}", "symbol": f"S{i}", "asof_date": str(DAY),
            "reason": "source_quality_quarantined",
        } for i in [18, 19]],
    )
    assert report["computation"]["candidate_rows"] == 22
    assert report["computation"]["eligible_rows"] == 20
    assert report["computation"]["reference_eligible_rows"] == 20
    assert report["computation"]["missing_fraction"] == 0.1
    assert report["violations"] == ["computational_missing_fraction_exceeded"]


@pytest.mark.parametrize("identity", ["sec-0", "sec-18", "sec-21"])
def test_unresolved_identity_is_not_ordinary_missing_data(identity):
    missing = int(identity.split("-")[1])
    kwargs = dict(
        candidates=_candidates(22, missing=[missing]), reference=_reference(), delivery=_delivery(),
        inference=ProductionInferenceConfig(minimum_candidate_rows=10, minimum_eligible_rows=10),
        policy=ProductionQualityConfig(), stage="inference", unscorable=[{
            "security_id": identity, "symbol": f"S{missing}", "asof_date": str(DAY),
            "reason": "unresolved_security_identity",
        }],
    )
    if identity == "sec-0":
        with pytest.raises(DataContractError, match="delivery identity is unresolved"):
            assess_daily_quality(**kwargs)
    else:
        report = assess_daily_quality(**kwargs)
        assert report["status"] == "degraded" and report["violations"] == []
        assert report["computation"]["reference_eligible_rows"] == 20
        assert report["computation"]["missing_fraction"] == (0.05 if missing == 18 else 0)
        assert report["unscorable"][0]["security_id"] == identity


def test_reference_excludes_target_day_and_short_history():
    history = pl.DataFrame([
        {"security_id": stock, "trade_date": DAY - timedelta(days=offset), "eligible": offset > 0}
        for stock, offsets in (("old", range(5)), ("new", range(2))) for offset in offsets
    ])
    reference = build_quality_reference(history, target_date=DAY, context_length=3)
    assert reference["security_ids"] == ["old"]
    assert reference["reference_date"] == "2026-08-16"
    assert DAY.isoformat() not in reference["market_eligible_rows"]


def test_market_gate_measures_actual_channels_and_contributor_coverage(tmp_path):
    from facdigger.data.config import MARKET_CONTEXT_CHANNELS

    days = [date(2026, 8, 14), DAY]
    universe = pl.DataFrame([
        {"security_id": f"sec-{i}", "trade_date": day, "eligible": True}
        for day in days for i in range(20)
    ])
    features = universe.drop("eligible").with_columns(
        ~((pl.col("trade_date") == DAY) & pl.col("security_id").is_in(["sec-0", "sec-1"]))
        .alias("observed_r_close")
    )
    features.write_parquet(tmp_path / "features.parquet")
    market = pl.DataFrame({
        "trade_date": days,
        **{f"observed_{channel}": [True, True] for channel in MARKET_CONTEXT_CHANNELS},
    })
    market.write_parquet(tmp_path / "market.parquet")
    manifest = {
        "feature_contract": {
            "feature_set": "finance_transformer", "context_length": 2,
            "market_channels": MARKET_CONTEXT_CHANNELS,
        },
        "artifacts": {"features": "features.parquet", "market_features": "market.parquet"},
    }
    report = assess_market_quality(
        tmp_path, manifest, universe, _reference(), ProductionQualityConfig(),
    )
    assert report["target_channels_complete"] is True
    assert report["violations"] == ["market_contributor_coverage_insufficient"]
    assert report["deficient_contributor_dates"] == [
        {"date": DAY.isoformat(), "expected": 20, "observed": 18},
    ]
    market.filter(pl.col("trade_date") != DAY).write_parquet(tmp_path / "market.parquet")
    report = assess_market_quality(
        tmp_path, manifest, universe, _reference(), ProductionQualityConfig(),
    )
    assert report["missing_dates"] == [DAY.isoformat()]
    assert "target_market_channels_unavailable" in report["violations"]
