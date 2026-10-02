from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from facdigger.data.contracts import DataContractError
from facdigger.inference.runner import _verify_replay
from facdigger.research.finance_diagnostics import diagnostic_daily, fixed_dates


def test_fixed_panel_is_date_only_and_stratified():
    dates = [date(2000, 1, 1) + timedelta(days=i) for i in range(2309)]
    rows = pl.DataFrame({"asof_date": dates, "target": np.arange(len(dates))})
    panel = fixed_dates(rows)
    assert len(panel) == len(set(panel)) == 60
    assert panel == sorted(panel)
    assert panel[0] == dates[0] and panel[-1] == dates[-1]
    assert panel == fixed_dates(rows.reverse().with_columns(pl.lit(-999).alias("target")))


def test_replay_subset_keeps_original_tolerance_and_full_keys(tmp_path):
    dates = [date(2020, 1, 1), date(2020, 1, 2)]
    original = pl.DataFrame(
        {
            "security_id": ["A", "B"] * 2,
            "asof_date": [dates[0]] * 2 + [dates[1]] * 2,
            "target": [1.0, 2.0, 3.0, 4.0],
            "score_raw": [1.0, 2.0, 3.0, 4.0],
        }
    )
    original.write_parquet(tmp_path / "predictions.parquet")
    subset = original.filter(pl.col("asof_date") == dates[0])
    kw = dict(require_match=True, asof_dates=[dates[0]])
    assert _verify_replay(tmp_path, {"evaluation_split": "valid"}, "valid", subset, **kw)["matched"]
    with pytest.raises(RuntimeError, match="scores differ"):
        _verify_replay(
            tmp_path,
            {"evaluation_split": "valid"},
            "valid",
            subset.with_columns(pl.col("score_raw") + 0.00001),
            **kw,
        )
    with pytest.raises(DataContractError, match="keys or targets"):
        _verify_replay(tmp_path, {"evaluation_split": "valid"}, "valid", subset.head(1), **kw)


def test_diagnostics_preserve_computation_pool_and_fixed_label_support():
    d = date(2020, 1, 1)
    rows = pl.DataFrame(
        {
            "sample_id": ["A", "B", "C", "D"],
            "security_id": ["A", "B", "C", "D"],
            "asof_date": [d] * 4,
            "feature_start": [d] * 4,
            "feature_end": [d] * 4,
        }
    )
    labelled = rows.head(3).with_columns(
        *[pl.Series(f"target_{h}", [2.0, 0.0, 1.0]) for h in [1, 5, 20]]
    )
    config = SimpleNamespace(
        horizons=[1, 5, 20],
        training=SimpleNamespace(
            objective=SimpleNamespace(
                minimum_cross_section_size=2,
                horizon_weights={1: 0.2, 5: 0.6, 20: 0.2},
                epsilon=1e-6,
                scale_regularization=0.01,
            )
        ),
    )
    scores = np.tile(np.asarray([3.0, 1.0, 2.0, -999.0])[:, None], (1, 3))
    daily = diagnostic_daily(scores, rows, labelled, config)
    assert daily["labelled_rows"].to_list() == [3] * 3
    assert daily["computational_rows"].to_list() == [4] * 3
    assert np.allclose(daily["rank_ic"], 1.0)
    assert daily["full_objective_available"].all()
    with pytest.raises(DataContractError, match="incomplete"):
        diagnostic_daily(scores[:3], rows, labelled, config)
    scores[-1, 0] = np.nan  # Even an unlabelled failed prediction must remain visible.
    with pytest.raises(DataContractError, match="non-finite"):
        diagnostic_daily(scores, rows, labelled, config)


def test_paired_report_uses_fixed_support_and_no_hac_on_sparse_fit():
    from facdigger.research.finance_diagnostic_report import paired_daily

    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(70)]
    base = pl.DataFrame(
        {
            "asof_date": days,
            "horizon": [5] * 70,
            "labelled_rows": [100] * 70,
            "computational_rows": [105] * 70,
            "rank_ic": np.sin(np.arange(70)) * 0.02,
        }
    )
    candidate = base.with_columns(pl.col("rank_ic") + 0.002)
    report = paired_daily(candidate, base)
    assert report["mean_delta"] == pytest.approx(0.002)
    assert len(report["HAC"]) == 6
    assert len(report["stride5_all_offsets"]) == 5
    assert "HAC" not in paired_daily(candidate, base, sparse=True)
    with pytest.raises(DataContractError, match="no intersection"):
        paired_daily(candidate.head(69), base)
    invalid = candidate.with_columns(pl.lit(None, dtype=pl.Float64).alias("rank_ic"))
    assert paired_daily(invalid, base)["status"] == "undefined_rank_ic"
