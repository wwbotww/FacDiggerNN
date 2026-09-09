from datetime import date

import polars as pl
import pytest

from facdigger.data.contracts import DataContractError
from facdigger.inference.delivery import DeliveryConfig, load_delivery_config, resolve_delivery
from facdigger.inference.factor_batch import build_factor_frame

DAY = date(2026, 8, 12)
AAPL = "eodhd:isin:US0378331005"


def profile(*, source_id=AAPL, valid_to=DAY):
    return DeliveryConfig.model_validate({
        "targets": [{"instrument_id": "AAPL.US"}],
        "identities": [{
            "instrument_id": "AAPL.US", "security_id": AAPL,
            "source_security_id": source_id,
            "valid_from": "2026-08-01", "valid_to": valid_to,
            "evidence": "Synthetic, consumer-confirmed fixture; not a live mapping.",
        }],
    })


def candidates(*, source_id=AAPL, eligible=True):
    return pl.DataFrame({
        "security_id": [source_id, "eodhd:symbol:UNKNOWN.US"],
        "symbol": ["AAPL", "UNKNOWN"], "asof_date": [DAY, DAY],
        "eligible": [eligible, True],
    }).sort("asof_date", "security_id")


def test_delivery_excludes_unmapped_non_targets_without_changing_scores():
    universe = candidates()
    selection = resolve_delivery(universe, profile())
    scores = universe.select("security_id", "asof_date").with_columns(
        pl.Series("score", [0.25, 9.0], dtype=pl.Float64)
    )
    full = build_factor_frame(universe, scores)
    delivered = build_factor_frame(selection.candidates, selection.project_scores(scores))
    assert delivered.equals(full.filter(pl.col("security_id") == AAPL))
    assert selection.audit["computational_candidate_rows"] == 2
    assert selection.audit["requested_target_rows"] == 1


def test_fallback_requires_an_explicit_valid_mapping():
    source = "eodhd:symbol:AAPL.US"
    with pytest.raises(DataContractError, match="require a delivery profile"):
        resolve_delivery(candidates(source_id=source), None)
    selected = resolve_delivery(candidates(source_id=source), profile(source_id=source))
    assert selected.source_candidates["security_id"].to_list() == [source]
    assert selected.candidates["security_id"].to_list() == [AAPL]


def test_expired_mapping_does_not_shrink_the_expected_set():
    with pytest.raises(DataContractError, match="missing or expired"):
        resolve_delivery(candidates(), profile(valid_to=date(2026, 8, 11)))


def test_matching_symbol_does_not_resolve_an_unknown_identity():
    with pytest.raises(DataContractError, match="absent from the candidate"):
        resolve_delivery(candidates(source_id="eodhd:symbol:AAPL.US"), profile())


def test_ineligible_rows_still_require_identity_and_are_delivered_without_scores():
    selection = resolve_delivery(candidates(eligible=False), profile())
    empty = pl.DataFrame(schema={
        "security_id": pl.String, "asof_date": pl.Date, "score": pl.Float64,
    })
    assert build_factor_frame(selection.candidates, empty)["score"].to_list() == [None]
    with pytest.raises(DataContractError, match="missing or expired"):
        resolve_delivery(candidates(eligible=False), profile(valid_to=date(2026, 8, 11)))


def test_missing_eligible_score_cannot_be_hidden_by_projection():
    selection = resolve_delivery(candidates(), profile())
    scores = pl.DataFrame(schema={
        "security_id": pl.String, "asof_date": pl.Date, "score": pl.Float64,
    })
    with pytest.raises(DataContractError, match="no model score"):
        build_factor_frame(selection.candidates, selection.project_scores(scores))


@pytest.mark.parametrize("key", ["instrument_id", "source_security_id", "security_id"])
def test_overlapping_identity_intervals_are_rejected(key):
    payload = profile(source_id="eodhd:symbol:AAPL.US").model_dump()
    second = {
        **payload["identities"][0], "instrument_id": "OTHER.US",
        "source_security_id": "eodhd:symbol:MSFT.US",
        "security_id": "eodhd:isin:US5949181045",
    }
    payload["targets"].append({"instrument_id": "OTHER.US"})
    second[key] = payload["identities"][0][key]
    payload["identities"].append(second)
    with pytest.raises(ValueError, match="overlapping"):
        DeliveryConfig.model_validate(payload)


def test_do_not_rewrite_old_isin_to_current_isin():
    with pytest.raises(ValueError, match="historical ISIN"):
        profile(source_id="eodhd:isin:US30231G1022")


def test_verified_mapping_can_target_an_explicit_non_isin_consumer_id():
    payload = profile(source_id=AAPL).model_dump()
    payload["identities"][0]["security_id"] = "heyboss:instrument:AAPL.US"
    mapped = resolve_delivery(candidates(), DeliveryConfig.model_validate(payload))
    assert mapped.candidates["security_id"].to_list() == ["heyboss:instrument:AAPL.US"]
    assert mapped.source_candidates["security_id"].to_list() == [AAPL]


def test_delivery_profile_loader_requires_mapping_root(tmp_path):
    path = tmp_path / "delivery.yaml"
    path.write_text(profile().model_dump_json())
    assert load_delivery_config(path) == profile()
    path.write_text("[]")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_delivery_config(path)


def test_eligible_only_delivery_does_not_hide_unknown_or_expired_mapping():
    with pytest.raises(DataContractError, match="absent"):
        resolve_delivery(
            candidates(source_id="not-in-this-eligible-prediction-table"), profile(),
            eligible_only=True,
        )
    with pytest.raises(DataContractError, match="expired"):
        resolve_delivery(
            candidates(), profile(valid_to=date(2026, 8, 11)),
            eligible_only=True,
        )
