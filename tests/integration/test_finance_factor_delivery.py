from __future__ import annotations

import json
import shutil
from pathlib import Path

import factor_fixtures
import numpy as np
import polars as pl
import pytest
from factor_fixtures import build_price_volume_snapshot
from polars.testing import assert_frame_equal

from facdigger.data.config import (
    FINANCE_TRANSFORMER_CHANNELS,
    MARKET_CONTEXT_CHANNELS,
    DatasetBuildConfig,
    InferenceSnapshotConfig,
)
from facdigger.data.contracts import DataContractError, table_audit
from facdigger.data.inference_snapshots import build_inference_snapshot, load_inference_snapshot
from facdigger.data.market_calendar import regular_session_frame
from facdigger.data.provenance import build_standardization_contract
from facdigger.data.session_store import bootstrap_production_store
from facdigger.data.snapshots import build_dataset_snapshot, sha256_file
from facdigger.inference.delivery import DeliveryConfig
from facdigger.inference.factor_batch import load_factor_batch, publish_evaluation_factor_batch
from facdigger.inference.history import (
    HistoricalReplayConfig,
    run_historical_replay,
    verify_historical_replay,
)
from facdigger.inference.releases import create_model_release, load_model_release
from facdigger.inference.runner import run_inference, run_signal_inference
from facdigger.inference.scoring import load_factor_inference_runtime, score_inference_rows
from facdigger.training.finance_transformer import run_finance_transformer
from facdigger.training.finance_transformer_config import FinanceTransformerExperimentConfig


@pytest.fixture
def finance_delivery(tmp_path, monkeypatch, request):
    import torch

    torch.set_num_threads(1)
    calendar = factor_fixtures.sessions(165)
    original = build_price_volume_snapshot(tmp_path, count=165)
    source = json.loads((original / "manifest.json").read_text())["source_paths"]
    universe_path = Path(source["universe"])
    universe = pl.read_parquet(universe_path).with_columns(
        (
            ~((pl.col("security_id") == "sec-0") & (pl.col("trade_date") >= calendar[160]))
            & ~((pl.col("security_id") == "sec-5") & (pl.col("trade_date") < calendar[65]))
        ).alias("eligible")
    )
    universe.write_parquet(universe_path)
    dataset, _ = build_dataset_snapshot(
        DatasetBuildConfig.model_validate(
            {
                "sources": source,
                "output_root": tmp_path / "finance-snapshots",
                "features": {
                    "name": "finance_transformer",
                    "context_length": 20,
                    "channels": FINANCE_TRANSFORMER_CHANNELS,
                    "market_channels": MARKET_CONTEXT_CHANNELS,
                },
                "label": {"horizon": 5, "auxiliary_horizons": [1, 20]},
                "split": {
                    "train_end": calendar[100],
                    "valid_end": calendar[130],
                    "test_end": calendar[164],
                    "embargo_sessions": 0,
                },
            }
        )
    )
    config = FinanceTransformerExperimentConfig.model_validate(
        {
            "output_root": tmp_path / "runs",
            "selection_fraction": 0.3,
            "model": {
                "patch_length": 4,
                "patch_stride": 4,
                "local_d_model": 8,
                "local_num_attention_heads": 2,
                "local_num_hidden_layers": 1,
                "local_ffn_dim": 16,
                "market_d_model": 8,
                "market_num_attention_heads": 2,
                "market_num_hidden_layers": 1,
                "market_ffn_dim": 16,
                "embedding_dim": 8,
                "cross_num_attention_heads": 2,
                "cross_num_hidden_layers": 1,
                "cross_ffn_dim": 16,
                "statistics_output_dim": 8,
                "statistics_windows": [4, 8, 16],
                "dropout": 0.0,
            },
            "training": {
                "batch_size": 2,
                "max_epochs": 1,
                "minimum_epochs": 1,
                "patience": 1,
                "dates_per_optimizer_step": 2,
                "device": "cpu",
                "precision": "fp32",
                "objective": {
                    "minimum_cross_section_size": 4,
                    "minimum_selection_dates": 2,
                    "minimum_selection_coverage": 1.0,
                    "selection_subperiods": 1,
                },
            },
        }
    )

    if getattr(request, "param", "scratch") == "finance_pretrained":
        from facdigger.models.finance_patch_transformer import build_finance_transformer_model

        # A small encoder export exercises supervised initialization, not pretraining quality.
        encoder = build_finance_transformer_model(config, context_length=20)
        encoder_path = tmp_path / "encoder.pt"
        torch.save(
            {
                "contract": "finance_patch_pretrain_encoder",
                "dataset_id": dataset.name,
                "context_length": 20,
                "channels": config.channels,
                "market_channels": config.market_channels,
                "local_encoder_state": encoder.local_encoder.state_dict(),
                "market_encoder_state": encoder.market_encoder.state_dict(),
            },
            encoder_path,
        )
        config = FinanceTransformerExperimentConfig.model_validate(
            {
                **config.model_dump(),
                "initialization": "finance_pretrained",
                "pretrained_checkpoint": encoder_path,
            }
        )

    def clean(root):
        return {"commit": "1" * 40, "branch": "fixture", "dirty": False}

    monkeypatch.setattr("facdigger.training.finance_transformer.collect_git_state", clean)
    monkeypatch.setattr("facdigger.inference.releases.collect_git_state", clean)
    run, _ = run_finance_transformer(config, dataset, repository_root=tmp_path)
    release_dir, release = create_model_release(
        run, tmp_path / "releases", repository_root=tmp_path
    )
    if config.pretrained_checkpoint is not None:
        config.pretrained_checkpoint.rename(tmp_path / "offline-encoder.pt")
    inference_config = InferenceSnapshotConfig.model_validate(
        {
            "sources": source,
            "output_root": tmp_path / "inference",
        }
    )
    return run, dataset, release_dir, release, inference_config


@pytest.mark.parametrize("finance_delivery", ["scratch", "finance_pretrained"], indirect=True)
def test_finance_release_replay_daily_and_subset_history_share_forward(finance_delivery, tmp_path):
    run, dataset, release_dir, release, config = finance_delivery
    _, replay = run_inference(
        run,
        dataset_dir=dataset,
        device="cpu",
        require_replay_match=True,
        output_dir=tmp_path / "replay",
    )
    assert replay["replay_verification"]["matched"] is True
    assert load_model_release(release_dir).release_id == release.release_id
    assert release.feature_contract.market_channels == MARKET_CONTEXT_CHANNELS
    evaluation_dir, _ = publish_evaluation_factor_batch(
        run / "predictions.parquet", release_dir, tmp_path / "evaluation"
    )
    assert load_factor_batch(evaluation_dir).model.model_type == "finance_patch_transformer"

    history_snapshot, history_manifest = build_inference_snapshot(config, release_dir)
    last = factor_fixtures.sessions(165)[-1]
    daily_snapshot, _ = build_inference_snapshot(config, release_dir, asof_date=last)
    # Dynamic historical membership must not be replaced by today's membership.
    window_start = factor_fixtures.sessions(165)[-20]
    market = pl.read_parquet(daily_snapshot / "market_features.parquet").filter(
        pl.col("trade_date") >= window_start
    )
    expected_market = pl.read_parquet(history_snapshot / "market_features.parquet").filter(
        pl.col("trade_date").is_in(market["trade_date"].to_list())
    )
    assert_frame_equal(market, expected_market)
    for channel in ["rank_r_close", "rank_range"]:
        actual = (
            pl.read_parquet(daily_snapshot / "features.parquet")
            .filter(pl.col("trade_date") >= window_start)
            .select("security_id", "trade_date", channel)
        )
        expected = (
            pl.read_parquet(history_snapshot / "features.parquet")
            .join(
                actual.select("security_id", "trade_date"),
                on=["security_id", "trade_date"],
                how="semi",
            )
            .select(actual.columns)
        )
        assert_frame_equal(actual, expected)

    # Once released, none of the production entry points need the training run or labels.
    dataset.rename(tmp_path / "offline-training-snapshot")
    run.rename(tmp_path / "offline-training-run")
    batch_dir, batch = run_signal_inference(
        release_dir, dataset_dir=daily_snapshot, output_root=tmp_path / "daily"
    )
    batch_meta = load_factor_batch(batch_dir)
    factors = pl.read_parquet(batch_dir / "factors.parquet")
    assert batch["model"]["model_type"] == "finance_patch_transformer"
    assert batch_meta.source.kind == "signal_inference"
    assert factors.height == 6
    assert factors.filter(~pl.col("eligible"))["security_id"].to_list() == ["sec-0"]
    assert factors.filter(~pl.col("eligible"))["score"].null_count() == 1
    repeated, _ = run_signal_inference(
        release_dir, dataset_dir=daily_snapshot, output_root=tmp_path / "daily"
    )
    assert repeated == batch_dir

    _, frames = load_inference_snapshot(history_snapshot, release)
    rows = frames["inference_index"].filter(pl.col("asof_date") == last)
    runtime = load_factor_inference_runtime(release_dir, batch_size=2)
    full = score_inference_rows(
        runtime, snapshot_dir=history_snapshot, snapshot_manifest=history_manifest, rows=rows
    )
    other_runtime = load_factor_inference_runtime(release_dir, batch_size=5)
    other = score_inference_rows(
        other_runtime, snapshot_dir=history_snapshot, snapshot_manifest=history_manifest, rows=rows
    )
    np.testing.assert_allclose(full["score"], other["score"], atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(
        factors.filter(pl.col("eligible"))["score"], full["score"], atol=2e-6, rtol=2e-5
    )
    subset = rows.filter(pl.col("security_id") == "sec-1")
    subset_scores = score_inference_rows(
        runtime, snapshot_dir=history_snapshot, snapshot_manifest=history_manifest, rows=subset
    )
    assert_frame_equal(subset_scores, full.filter(pl.col("security_id") == "sec-1"))

    history_config = HistoricalReplayConfig.model_validate(
        {
            "history_id": "finance-subset",
            "release_dir": release_dir,
            "inference_snapshot_dir": history_snapshot,
            "output_root": tmp_path / "history",
            "start_date": last,
            "end_date": last,
            "security_ids": ["sec-1"],
            "acknowledge_non_oos": True,
            "batch_size": 2,
        }
    )
    export, manifest = run_historical_replay(history_config)
    assert verify_historical_replay(export).release_id == release.release_id
    partition_dir = export / manifest.partitions[0].path
    load_factor_batch(partition_dir)
    historical = pl.read_parquet(partition_dir / "factors.parquet")
    assert_frame_equal(
        historical.select("security_id", "symbol", "asof_date", "score"), subset_scores
    )


def test_finance_scoring_rejects_targets_and_wrong_membership(finance_delivery):
    _, _, release_dir, release, config = finance_delivery
    snapshot, manifest = build_inference_snapshot(config, release_dir)
    _, frames = load_inference_snapshot(snapshot, release)
    rows = frames["inference_index"].filter(
        pl.col("asof_date") == factor_fixtures.sessions(165)[-1]
    )
    runtime = load_factor_inference_runtime(release_dir)
    with pytest.raises(DataContractError, match="targets"):
        score_inference_rows(
            runtime,
            snapshot_dir=snapshot,
            snapshot_manifest=manifest,
            rows=rows.with_columns(pl.lit(1.0).alias("target_5")),
        )
    with pytest.raises(DataContractError, match="scoring universe"):
        score_inference_rows(
            runtime,
            snapshot_dir=snapshot,
            snapshot_manifest=manifest,
            rows=rows.with_columns(pl.lit("wrong").alias("symbol")),
        )


def test_partial_finance_signal_keeps_null_row_and_scores_remaining_cross_section(
    finance_delivery, tmp_path,
):
    _, _, release_dir, release, config = finance_delivery
    last = factor_fixtures.sessions(165)[-1]
    # Leave the generic source flag true to prove that a missing real D bar
    # cannot be scored merely because a provider supplied an optimistic flag.
    bars = pl.read_parquet(config.sources.bars).filter(
        ~((pl.col("security_id") == "sec-1") & (pl.col("trade_date") == last))
    )
    bars_path = tmp_path / "partial-bars.parquet"
    bars.write_parquet(bars_path)
    payload = config.model_dump()
    payload["sources"]["bars"] = bars_path
    snapshot, manifest = build_inference_snapshot(
        InferenceSnapshotConfig.model_validate(payload), release_dir, asof_date=last,
    )
    _, frames = load_inference_snapshot(snapshot, release)
    full = score_inference_rows(
        load_factor_inference_runtime(release_dir), snapshot_dir=snapshot,
        snapshot_manifest=manifest, rows=frames["inference_index"],
    )
    assert full.height == 4  # sec-0 outside model pool; sec-1 has no observed D bar.
    delivery = DeliveryConfig.model_validate({
        "targets": [{"instrument_id": f"S{i}.US"} for i in range(1, 6)],
        "identities": [{
            "instrument_id": f"S{i}.US", "security_id": f"sec-{i}",
            "valid_from": last, "valid_to": last, "evidence": "Synthetic fixture",
        } for i in range(1, 6)],
    })
    bundle, result = run_signal_inference(
        release_dir, dataset_dir=snapshot, output_root=tmp_path / "partial-batches",
        delivery=delivery, asof=last.isoformat(),
    )
    frame = pl.read_parquet(bundle / "factors.parquet")
    assert frame.height == 5
    assert frame.filter(~pl.col("eligible")).select("security_id", "score").rows() == [
        ("sec-1", None),
    ]
    assert_frame_equal(
        frame.filter(pl.col("eligible")).select("security_id", "asof_date", "score"),
        full.select("security_id", "asof_date", "score"),
    )
    assert result["coverage"]["ratio"] == 1.0
    assert result["coverage"]["missing_eligible_rows"] == 0
    assert load_factor_batch(bundle).coverage.scored_eligible_rows == 4
    audit = json.loads((snapshot / "audit.json").read_text())
    reasons = audit["delivery_universe"]["latest_unscorable"]
    assert next(row for row in reasons if row["security_id"] == "sec-1")["reason"] == (
        "missing_target_bar"
    )


def test_cached_index_cannot_score_unobserved_target_bar(finance_delivery):
    _, _, release_dir, release, config = finance_delivery
    last = factor_fixtures.sessions(165)[-1]
    snapshot, manifest = build_inference_snapshot(config, release_dir, asof_date=last)
    features_path = snapshot / manifest["artifacts"]["features"]
    features = pl.read_parquet(features_path).with_columns(
        (pl.col("observed_range") & ~(
            (pl.col("security_id") == "sec-1") & (pl.col("trade_date") == last)
        )).alias("observed_range"),
    )
    features.write_parquet(features_path)
    # Reproduce a self-consistent older artifact: hashes pass, but its optimistic
    # index still declares an unobserved D bar scorable. Do not silently reuse it.
    manifest["artifact_hashes"]["features"] = sha256_file(features_path)
    (snapshot / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DataContractError, match="unobserved as-of bar"):
        load_inference_snapshot(snapshot, release)


def test_exact_finance_snapshot_can_record_no_scorable_target_without_filling(
    finance_delivery, tmp_path,
):
    _, _, release_dir, release, config = finance_delivery
    last = factor_fixtures.sessions(165)[-1]
    universe = pl.read_parquet(config.sources.universe).with_columns(
        (pl.col("eligible") & (pl.col("trade_date") != last)).alias("eligible")
    )
    path = tmp_path / "unavailable-universe.parquet"
    universe.write_parquet(path)
    payload = config.model_dump()
    payload["sources"]["universe"] = path
    snapshot, _ = build_inference_snapshot(
        InferenceSnapshotConfig.model_validate(payload), release_dir, asof_date=last,
    )
    _, frames = load_inference_snapshot(snapshot, release)
    assert frames["inference_index"].is_empty()
    assert frames["delivery_universe"].height == 6
    assert frames["delivery_universe"]["eligible"].sum() == 0
    audit = json.loads((snapshot / "audit.json").read_text())
    assert audit["inference_index"]["maximum_asof_date"] is None
    assert audit["delivery_universe"]["latest_cross_section_rows"] == 6


def test_finance_inference_consumes_complete_production_hot_history(finance_delivery, tmp_path):
    _, _, release_dir, _, config = finance_delivery
    source = tmp_path / "production-bronze"
    source.mkdir()
    dates = factor_fixtures.sessions(165)
    calendar = regular_session_frame(dates[0], dates[-1])
    evidence = {}
    for name, original_path in [
        ("bars", config.sources.bars),
        ("universe", config.sources.universe),
    ]:
        frame = pl.read_parquet(original_path).join(calendar, on="trade_date", how="semi")
        path = source / f"{name}_daily.parquet"
        frame.write_parquet(path)
        evidence[name] = {
            **table_audit(frame, "trade_date"),
            "file": path.name,
            "sha256": sha256_file(path),
        }
    (source / "eodhd_ingestion_manifest.json").write_text(
        json.dumps(
            {
                "provider": "eodhd",
                "source_revision": "synthetic",
                "warnings": [],
                "ingested_at": "2022-08-20T00:00:00+00:00",
                "standardization": build_standardization_contract(evidence, research_ready=False),
            }
        )
    )
    current = bootstrap_production_store(source, tmp_path / "store", history_sessions=40)
    hot_config = InferenceSnapshotConfig.model_validate(
        {
            "sources": {
                "bars": current.root / "bars_daily.parquet",
                "universe": current.root / "universe_daily.parquet",
                "source_manifest": current.root / "eodhd_ingestion_manifest.json",
            },
            "output_root": tmp_path / "hot-snapshots",
        }
    )
    snapshot, _ = build_inference_snapshot(hot_config, release_dir, asof_date=dates[-1])
    bundle, _ = run_signal_inference(
        release_dir, dataset_dir=snapshot, output_root=tmp_path / "hot"
    )
    assert load_factor_batch(bundle).coverage.ratio == 1.0
    universe = pl.read_parquet(current.root / "universe_daily.parquet")
    assert universe["trade_date"].n_unique() == 40
    assert not universe.filter(
        (pl.col("security_id") == "sec-0") & (pl.col("trade_date") == dates[-1])
    )["eligible"].item()


@pytest.mark.parametrize("mutation", ["checkpoint_config", "market_channels", "horizons"])
def test_finance_release_binds_checkpoint_config_and_market_contract(
    finance_delivery, tmp_path, mutation
):
    import torch

    run, _, _, _, _ = finance_delivery
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "checkpoint_config":
        checkpoint = run / manifest["checkpoint"]["file"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        payload["config"]["primary_horizon"] = 1
        torch.save(payload, checkpoint)
        manifest["checkpoint"]["sha256"] = sha256_file(checkpoint)
        message = "checkpoint configuration"
    else:
        manifest["input"][mutation] = list(reversed(manifest["input"][mutation]))
        message = "inputs or horizons"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DataContractError, match=message):
        create_model_release(run, tmp_path / "bad-releases", repository_root=tmp_path)


def test_mixed_identity_pool_delivers_one_target_after_complete_scoring(
    finance_delivery, tmp_path, monkeypatch,
):
    _, _, release_dir, release, config = finance_delivery
    payload = config.model_dump()
    for name in ("bars", "universe"):
        source = pl.read_parquet(getattr(config.sources, name)).with_columns(
            pl.when(pl.col("security_id") == "sec-2")
            .then(pl.lit("eodhd:symbol:UNKNOWN.US"))
            .otherwise(pl.col("security_id")).alias("security_id")
        )
        target = tmp_path / f"mixed-{name}.parquet"
        source.write_parquet(target)
        payload["sources"][name] = target
    payload["output_root"] = tmp_path / "mixed-inference"
    last = factor_fixtures.sessions(165)[-1]
    snapshot, manifest = build_inference_snapshot(
        InferenceSnapshotConfig.model_validate(payload), release_dir, asof_date=last,
    )
    _, frames = load_inference_snapshot(snapshot, release)
    rows = frames["inference_index"]
    assert "eodhd:symbol:UNKNOWN.US" in rows["security_id"].to_list()
    full = score_inference_rows(
        load_factor_inference_runtime(release_dir), snapshot_dir=snapshot,
        snapshot_manifest=manifest, rows=rows,
    )
    delivery = DeliveryConfig.model_validate({
        "targets": [{"instrument_id": "AAPL.US"}],
        "identities": [{
            "instrument_id": "AAPL.US", "source_security_id": "sec-1",
            "security_id": "eodhd:isin:US0378331005",
            "valid_from": last, "valid_to": last, "evidence": "Synthetic fixture only",
        }],
    })
    bundle, result = run_signal_inference(
        release_dir, dataset_dir=snapshot, output_root=tmp_path / "subset-daily",
        delivery=delivery,
    )
    delivered = pl.read_parquet(bundle / "factors.parquet")
    assert delivered["security_id"].to_list() == ["eodhd:isin:US0378331005"]
    np.testing.assert_array_equal(
        delivered["score"], full.filter(pl.col("security_id") == "sec-1")["score"],
    )
    assert result["coverage"]["candidate_rows"] == 1
    receipt = json.loads((
        tmp_path / "subset-daily_delivery_audits" / f"{bundle.name}.json"
    ).read_text())
    assert receipt["computational_eligible_rows"] == rows.height
    assert receipt["delivered_candidate_rows"] == 1
    history, historical = run_historical_replay(HistoricalReplayConfig(
        history_id="mixed-identity-history", release_dir=release_dir,
        inference_snapshot_dir=snapshot, output_root=tmp_path / "mixed-history",
        delivery=delivery, acknowledge_non_oos=True,
    ))
    assert historical.security_scope == "delivery_profile"
    assert_frame_equal(
        pl.read_parquet(history / historical.partitions[0].path / "factors.parquet"), delivered,
    )
    assert verify_historical_replay(history).row_count == 1

    from datetime import datetime, time
    from types import SimpleNamespace
    from zoneinfo import ZoneInfo

    from facdigger.production.config import ProductionServiceConfig
    from facdigger.production.runner import run_production_tick

    # Only source IO is substituted. Release loading, snapshots, model execution,
    # delivery projection, deadline guard and durable daily state are real.
    monkeypatch.setattr("facdigger.production.runner.load_current_revision", lambda _: (
        SimpleNamespace(manifest={"resolved_end": last.isoformat()})
    ))
    monkeypatch.setattr("facdigger.production.runner._source_config", lambda *_: (
        InferenceSnapshotConfig.model_validate(payload)
    ))
    monkeypatch.setattr("facdigger.production.runner.load_eodhd_config", lambda _: (
        SimpleNamespace(refresh=True, cache_ttl_hours=0)
    ))
    monkeypatch.setattr("facdigger.production.runner.EODHDProvider", lambda _: (
        SimpleNamespace(client=lambda: None)
    ))
    monkeypatch.setattr("facdigger.production.runner.fetch_daily_revision", lambda *a, **kw: (
        SimpleNamespace(request_log=({"path": "eod-bulk-last-day/US", "cache_hit": False},))
    ))
    monkeypatch.setattr(
        "facdigger.production.runner.publish_daily_source_revision",
        lambda *a, **kw: SimpleNamespace(manifest={"resolved_end": last.isoformat()}),
    )
    production = ProductionServiceConfig.model_validate({
        "data": {"provider_config": tmp_path / "unused-provider.yaml"},
        "model": {"release_root": release_dir.parent, "release_id": release.release_id},
        "inference": {
            "output_root": tmp_path / "production-inference",
            "minimum_candidate_rows": 6, "minimum_eligible_rows": 5,
        },
        "factor_batch": {"output_root": tmp_path / "production-factors", "delivery": delivery},
        "quality": {"minimum_delivery_eligible_rows": 1},
        "state_database": tmp_path / "production.sqlite3",
    })
    observed = datetime.combine(last, time(20), tzinfo=ZoneInfo("America/New_York"))
    tick = run_production_tick(production, now=observed, now_provider=lambda: observed)
    assert tick.action == "published", tick.error
    assert load_factor_batch(
        production.factor_batch.output_root / tick.delivery_id
    ).coverage.candidate_rows == 1
    assert run_production_tick(production, now=observed).action == "already_published"


def test_historical_export_verification_and_resume_survive_relocation(finance_delivery, tmp_path):
    _, _, release_dir, _, config = finance_delivery
    snapshot, _ = build_inference_snapshot(config, release_dir)
    last = factor_fixtures.sessions(165)[-1]
    request = HistoricalReplayConfig(
        history_id="relocated-history", release_dir=release_dir, inference_snapshot_dir=snapshot,
        output_root=tmp_path / "original", start_date=last, end_date=last,
        security_ids=["sec-1"], acknowledge_non_oos=True,
    )
    export, manifest = run_historical_replay(request)
    original_config = (export / "resolved_config.yaml").read_bytes()
    target = tmp_path / "another-machine"
    target.mkdir()
    moved_release = target / release_dir.name
    moved_snapshot = target / snapshot.name
    moved_export = target / export.name
    shutil.move(release_dir, moved_release)
    shutil.move(snapshot, moved_snapshot)
    shutil.move(export, moved_export)
    with pytest.raises(FileNotFoundError, match="specify --release"):
        verify_historical_replay(moved_export)
    assert verify_historical_replay(
        moved_export, release_dir=moved_release, inference_snapshot_dir=moved_snapshot,
    ) == manifest
    resumed = request.model_copy(update={
        "release_dir": moved_release, "inference_snapshot_dir": moved_snapshot,
        "output_root": target,
    })
    assert run_historical_replay(resumed)[1] == manifest
    # The completed partition is reusable even when the final aggregate was lost.
    (moved_export / "manifest.json").unlink()
    _, completed = run_historical_replay(resumed)
    assert completed.partitions == manifest.partitions
    assert (moved_export / "resolved_config.yaml").read_bytes() == original_config
    with pytest.raises(DataContractError, match="different resolved configuration"):
        run_historical_replay(resumed.model_copy(update={"security_ids": ["sec-2"]}))
