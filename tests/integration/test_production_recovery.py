"""Publication/ledger crash recovery with real release, snapshot and FactorBatch IO."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import factor_fixtures
import polars as pl
import pytest
from test_finance_factor_delivery import finance_delivery  # noqa: F401

from facdigger.data.config import InferenceSnapshotConfig
from facdigger.data.market_calendar import next_regular_session, regular_session
from facdigger.inference import factor_batch
from facdigger.production import runner
from facdigger.production.config import ProductionServiceConfig
from facdigger.production.recovery import resume_blocked_production
from facdigger.production.state import ProductionState

NY = ZoneInfo("America/New_York")


@pytest.fixture
def production_case(request, tmp_path, monkeypatch):
    _, _, release_dir, release, source_config = request.getfixturevalue("finance_delivery")
    target = factor_fixtures.sessions(165)[-1]
    config = ProductionServiceConfig.model_validate({
        "data": {
            "provider_config": tmp_path / "unused-provider.yaml",
            "store_root": tmp_path / "unused-store",
            "bootstrap_source": tmp_path / "unused-bronze",
        },
        "model": {"release_root": release_dir.parent, "release_id": release.release_id},
        "inference": {
            "output_root": tmp_path / "daily-snapshots",
            "minimum_candidate_rows": 6, "minimum_eligible_rows": 5,
        },
        "factor_batch": {
            "output_root": tmp_path / "daily-factors",
            "delivery": {
                "targets": [{"instrument_id": f"S{i}"} for i in range(1, 6)],
                "identities": [{
                    "instrument_id": f"S{i}", "security_id": f"sec-{i}",
                    "valid_from": target, "valid_to": target,
                    "evidence": "Synthetic test mapping only",
                } for i in range(1, 6)],
            },
        },
        "state_database": tmp_path / "production.sqlite3",
    })
    source = InferenceSnapshotConfig.model_validate({
        **source_config.model_dump(), "output_root": config.inference.output_root,
    })
    case = SimpleNamespace(
        config=config, source=source, release=release,
        now=datetime.combine(target, time(20), tzinfo=NY), fetches=0,
    )

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return case.now.astimezone(tz)

    monkeypatch.setattr(factor_batch, "datetime", Clock)
    current = SimpleNamespace(revision_id="fixture", manifest={"resolved_end": target.isoformat()})
    monkeypatch.setattr(runner, "load_current_revision", lambda _: current)
    monkeypatch.setattr(runner, "_source_config", lambda *_: source)
    monkeypatch.setattr(runner, "load_eodhd_config", lambda _: (
        SimpleNamespace(refresh=True, cache_ttl_hours=0)
    ))
    monkeypatch.setattr(runner, "EODHDProvider", lambda _: SimpleNamespace(client=lambda: None))

    def fetch(*args, **kwargs):
        case.fetches += 1
        return SimpleNamespace(
            request_log=({"path": "eod-bulk-last-day/US", "cache_hit": False},),
            backfilled_provider_symbols=(),
        )

    monkeypatch.setattr(runner, "fetch_daily_revision", fetch)
    monkeypatch.setattr(runner, "publish_daily_source_revision", lambda *a, **kw: current)
    return case


def _tick(case):
    return runner.run_production_tick(case.config, now=case.now, now_provider=lambda: case.now)


def _interrupt_after_publish(case, monkeypatch):
    original_put = ProductionState.put

    def put(self, target, release_id, status, **kwargs):
        if status == "published":
            raise SystemExit("exit after atomic directory rename, before ledger commit")
        return original_put(self, target, release_id, status, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(ProductionState, "put", put)
        with pytest.raises(SystemExit, match="before ledger commit"):
            _tick(case)
    bundles = list(case.config.factor_batch.output_root.iterdir())
    assert len(bundles) == 1
    manifest = factor_batch.load_factor_batch(bundles[0])
    with ProductionState(case.config.state_database) as state:
        pending = state.get(case.now.date())
        assert pending.status == "running" and pending.delivery_id is None
        assert pending.quality_report["snapshot_id"] == manifest.input.snapshot_id
    return bundles[0], manifest


@pytest.mark.parametrize("after_cutoff", [False, True])
def test_operator_resume_reconciles_original_without_requeue_or_fresh_data(
    production_case, monkeypatch, after_cutoff,
):
    case = production_case
    bundle, original = _interrupt_after_publish(case, monkeypatch)
    before = {p.name: p.read_bytes() for p in bundle.iterdir()}
    target = case.now.date()
    with ProductionState(case.config.state_database) as state:
        previous = state.get(target)
        blocked = state.put(target, previous.release_id, "blocked", attempts=previous.attempts,
                            error="operator observed a post-publication interruption")
    if after_cutoff:
        case.now = regular_session(next_regular_session(target)).open_utc + timedelta(minutes=1)
    result = resume_blocked_production(case.config, target_date=target,
                                      expected_updated_at=blocked.updated_at, reason="reviewed",
                                      now_provider=lambda: case.now)
    assert result["action"] == "already_published"
    assert result["delivery_id"] == original.delivery_id
    assert before == {p.name: p.read_bytes() for p in bundle.iterdir()}
    assert case.fetches == 1
    with ProductionState(case.config.state_database) as state:
        assert state.get(target).status == "published"


def test_operator_resume_cannot_clear_ambiguous_originals(production_case, monkeypatch):
    case = production_case
    bundle, original = _interrupt_after_publish(case, monkeypatch)
    frame = pl.read_parquet(bundle / "factors.parquet").with_columns(pl.col("score") + 0.01)
    factor_batch.publish_factor_batch(
        frame, case.config.factor_batch.output_root, source=original.source, model=original.model,
        input_metadata=original.input, time_metadata=original.time,
    )
    assert _tick(case).action == "blocked"
    with ProductionState(case.config.state_database) as state:
        blocked = state.get(case.now.date())
    with pytest.raises(ValueError, match="ambiguous"):
        resume_blocked_production(case.config, target_date=case.now.date(),
                                  expected_updated_at=blocked.updated_at, reason="cannot bypass",
                                  now_provider=lambda: case.now)
    with ProductionState(case.config.state_database) as state:
        assert state.get(case.now.date()) == blocked
    assert case.fetches == 1


def test_restart_recovers_original_delivery_before_revised_data(production_case, monkeypatch):
    case = production_case
    bundle, original = _interrupt_after_publish(case, monkeypatch)
    before = {p.name: p.read_bytes() for p in bundle.iterdir()}
    # A revised source is now available. The original frozen snapshot must win;
    # running the old path again would create a different snapshot/delivery ID.
    revised = (pl.col("security_id") == "sec-1") & (pl.col("trade_date") == case.now.date())
    bars = pl.read_parquet(case.source.sources.bars).with_columns(
        pl.lit("revised-after-process-exit").alias("source_revision"),
        *[pl.when(revised).then(pl.col(c) * 1.01).otherwise(pl.col(c)).alias(c)
          for c in ("open", "high", "low", "close")],
    ).with_columns(
        (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
    )
    bars.write_parquet(case.source.sources.bars)
    case.now = case.now.replace(minute=30)
    resumed = _tick(case)
    assert resumed.action == "already_published", resumed.error
    assert resumed.delivery_id == original.delivery_id
    assert resumed.snapshot_id == original.input.snapshot_id
    assert resumed.attempts == 1 and case.fetches == 1
    assert _tick(case).action == "already_published"
    assert list(case.config.factor_batch.output_root.iterdir()) == [bundle]
    assert {p.name: p.read_bytes() for p in bundle.iterdir()} == before
    assert factor_batch.load_factor_batch(bundle).created_at == original.created_at
    assert original.created_at == datetime.combine(case.now.date(), time(20), tzinfo=NY).astimezone(
        timezone.utc
    )


@pytest.mark.parametrize("hours_after_open", [0, 1, 11])
def test_late_restart_only_records_original_timely_publication(
    production_case, monkeypatch, hours_after_open,
):
    case = production_case
    bundle, original = _interrupt_after_publish(case, monkeypatch)
    cutoff = regular_session(next_regular_session(case.now.date())).open_utc
    case.now = cutoff + timedelta(hours=hours_after_open)
    result = _tick(case)
    assert result.action == "already_published", result.error
    assert result.phase == "expired"
    assert result.target_date == original.time.maximum_asof_date
    assert result.delivery_id == original.delivery_id and case.fetches == 1
    assert factor_batch.load_factor_batch(bundle) == original
    if hours_after_open < 11:
        # A new date must not receive yesterday's factor as a fallback.
        next_tick = _tick(case)
        assert next_tick.action == "not_due" and next_tick.delivery_id is None


def test_exit_before_rename_staging_output_is_not_a_publication(production_case, monkeypatch):
    case = production_case

    def exit_before_commit(*args):
        raise SystemExit("exit before rename")

    with monkeypatch.context() as fault:
        fault.setattr(runner, "_publish_guard", exit_before_commit)
        with pytest.raises(SystemExit, match="before rename"):
            _tick(case)
    staging = list(case.config.factor_batch.output_root.iterdir())
    assert len(staging) == 1 and staging[0].name.startswith(".tmp-factor-batch-")
    # Staging alone is not externally visible, so a fresh, timely retry is valid.
    case.now += timedelta(minutes=30)
    result = _tick(case)
    assert result.action == "published", result.error
    assert case.fetches == 2 and result.attempts == 2
    assert factor_batch.load_factor_batch(
        case.config.factor_batch.output_root / result.delivery_id
    ).time.maximum_asof_date == case.now.date()


def test_late_restart_without_completed_directory_expires(production_case, monkeypatch):
    case = production_case
    original_guard = runner._publish_guard

    def exit_before_commit(*args):
        original_guard(*args)
        raise SystemExit("exit before rename")

    with monkeypatch.context() as fault:
        fault.setattr(runner, "_publish_guard", exit_before_commit)
        with pytest.raises(SystemExit):
            _tick(case)
    before = sorted(case.config.factor_batch.output_root.iterdir())
    case.now = regular_session(next_regular_session(case.now.date())).open_utc
    result = _tick(case)
    assert result.action == "expired" and result.delivery_id is None
    assert case.fetches == 1
    assert sorted(case.config.factor_batch.output_root.iterdir()) == before


def test_inference_reaching_cutoff_cannot_publish(production_case, monkeypatch):
    case = production_case
    original_guard = runner._publish_guard
    cutoff = regular_session(next_regular_session(case.now.date())).open_utc

    def elapsed(cutoff_at, clock):
        case.now = cutoff
        original_guard(cutoff_at, clock)

    monkeypatch.setattr(runner, "_publish_guard", elapsed)
    result = _tick(case)
    assert result.action == "expired" and result.delivery_id is None
    assert not list(case.config.factor_batch.output_root.iterdir())


@pytest.mark.parametrize(
    "timestamp", ["before_first_attempt", "at_cutoff", "after_cutoff", "future"],
)
def test_orphan_with_out_of_window_timestamp_blocks(production_case, monkeypatch, timestamp):
    case = production_case
    bundle, _ = _interrupt_after_publish(case, monkeypatch)
    cutoff = regular_session(next_regular_session(case.now.date())).open_utc
    values = {
        "before_first_attempt": case.now.replace(hour=18),
        "at_cutoff": cutoff,
        "after_cutoff": cutoff + timedelta(seconds=1),
        "future": case.now + timedelta(minutes=1),
    }
    manifest_path = bundle / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["created_at"] = values[timestamp].astimezone(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(payload))
    # created_at is intentionally not part of the external semantic ID.
    factor_batch.load_factor_batch(bundle)
    if timestamp in {"at_cutoff", "after_cutoff"}:
        case.now = cutoff + timedelta(minutes=2)
    result = _tick(case)
    assert result.action == "blocked" and "publication window" in result.error
    assert case.fetches == 1 and list(case.config.factor_batch.output_root.iterdir()) == [bundle]


@pytest.mark.parametrize("different_release", [False, True])
def test_two_same_day_deliveries_block_even_if_one_matches(
    production_case, monkeypatch, different_release,
):
    case = production_case
    bundle, original = _interrupt_after_publish(case, monkeypatch)
    frame = pl.read_parquet(bundle / "factors.parquet").with_columns(pl.col("score") + 0.01)
    second, _ = factor_batch.publish_factor_batch(
        frame, case.config.factor_batch.output_root, source=original.source,
        model=original.model.model_copy(update={"release_id": "9" * 64})
        if different_release else original.model,
        input_metadata=original.input, time_metadata=original.time,
    )
    result = _tick(case)
    assert result.action == "blocked" and "ambiguous" in result.error
    assert original.delivery_id in result.error and second.name in result.error
    assert case.fetches == 1
    assert _tick(case).action == "blocked"
    assert len(list(case.config.factor_batch.output_root.iterdir())) == 2


@pytest.mark.parametrize("damage", [
    "manifest_json", "manifest_semantics", "factor_bytes", "missing_factor",
    "missing_manifest", "extra_file", "snapshot_bytes", "missing_snapshot",
    "release_bytes", "snapshot_binding", "quality_record", "missing_ledger",
    "delivery_targets", "identity_expiry", "model_lineage", "snapshot_hash", "calendar",
])
def test_unverifiable_original_blocks_without_republishing(production_case, monkeypatch, damage):
    case = production_case
    bundle, manifest = _interrupt_after_publish(case, monkeypatch)
    target = case.now.date()
    snapshot = case.config.inference.output_root / target.isoformat() / manifest.input.snapshot_id
    if damage == "manifest_json":
        (bundle / "manifest.json").write_text("{broken")
    elif damage == "manifest_semantics":
        payload = json.loads((bundle / "manifest.json").read_text())
        payload["model"]["model_id"] = "different-model"
        (bundle / "manifest.json").write_text(json.dumps(payload))
    elif damage in {"model_lineage", "snapshot_hash", "calendar"}:
        payload = manifest.model_dump(mode="json")
        if damage == "model_lineage":
            payload["model"]["model_id"] = "another-valid-model"
        elif damage == "snapshot_hash":
            payload["input"]["snapshot_manifest_sha256"] = "0" * 64
        else:
            payload["time"]["calendar_version"] = "another-calendar-build"
        changed = factor_batch.FactorBatchManifest.model_validate(payload)
        payload["delivery_id"] = factor_batch.factor_batch_delivery_id(changed)
        (bundle / "manifest.json").write_text(json.dumps(payload))
        bundle = bundle.rename(bundle.with_name(payload["delivery_id"]))
        # The generic bundle is internally consistent, but not the original
        # production release/snapshot/calendar and must still be refused.
        factor_batch.load_factor_batch(bundle)
    elif damage == "factor_bytes":
        (bundle / "factors.parquet").write_bytes(b"broken parquet")
    elif damage == "missing_factor":
        (bundle / "factors.parquet").unlink()
    elif damage == "missing_manifest":
        (bundle / "manifest.json").unlink()
    elif damage == "extra_file":
        (bundle / "undeclared").write_text("not a two-file delivery")
    elif damage == "snapshot_bytes":
        (snapshot / "scaler.json").write_text("{}")
    elif damage == "missing_snapshot":
        snapshot.rename(snapshot.with_name("unavailable-snapshot"))
    elif damage == "release_bytes":
        release_root = case.config.model.release_root / case.config.model.release_id
        (release_root / "scaler.json").write_text("{}")
    elif damage == "missing_ledger":
        with ProductionState(case.config.state_database) as state:
            state._connection.execute("DELETE FROM production_runs")
    elif damage in {"snapshot_binding", "quality_record"}:
        with ProductionState(case.config.state_database) as state:
            pending = state.get(target)
            quality = pending.quality_report
            if damage == "snapshot_binding":
                quality["snapshot_id"] = "0" * 64
            else:
                quality["violations"] = ["computational_missing_fraction_exceeded"]
            state.put(target, pending.release_id, "running", attempts=1, quality_report=quality)
    else:
        payload = case.config.model_dump()
        delivery = payload["factor_batch"]["delivery"]
        if damage == "delivery_targets":
            delivery["targets"] = delivery["targets"][:-1]
            delivery["identities"] = delivery["identities"][:-1]
        else:
            delivery["identities"][0].update(valid_from=target-timedelta(days=1),
                                             valid_to=target-timedelta(days=1))
        case.config = ProductionServiceConfig.model_validate(payload)
    before = {p.name: p.read_bytes() for p in bundle.iterdir()}
    result = _tick(case)
    assert result.action == "blocked" and result.error
    assert result.delivery_id is None and case.fetches == 1
    assert list(case.config.factor_batch.output_root.iterdir()) == [bundle]
    assert {p.name: p.read_bytes() for p in bundle.iterdir()} == before
    assert _tick(case).action == "blocked" and case.fetches == 1


def test_process_exit_and_fresh_process_recover_same_delivery(
    production_case, monkeypatch, tmp_path,
):
    case = production_case
    original_infer = runner.run_signal_inference

    # Persist the real quality-bound snapshot, but let a separate interpreter
    # execute scoring, the atomic publisher, and then an uncatchable os._exit.
    def subprocess_publish(*args, **kwargs):
        request_path = tmp_path / "crash-request.json"
        request_path.write_text(json.dumps({
            "config": case.config.model_dump(mode="json"),
            "snapshot": str(kwargs["dataset_dir"]), "now": case.now.isoformat(),
        }))
        script = Path(__file__).with_name("production_recovery_process.py")
        result = subprocess.run(
            [sys.executable, str(script), "publish-and-exit", str(request_path)],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PYTHONPATH": str(Path(runner.__file__).parents[2])},
        )
        assert result.returncode == 73, result.stderr
        raise SystemExit("child exited before ledger commit")

    with monkeypatch.context() as fault:
        fault.setattr(runner, "run_signal_inference", subprocess_publish)
        with pytest.raises(SystemExit):
            _tick(case)
    assert runner.run_signal_inference is original_infer
    bundle, = case.config.factor_batch.output_root.iterdir()
    original = factor_batch.load_factor_batch(bundle)
    before = {p.name: p.read_bytes() for p in bundle.iterdir()}
    with ProductionState(case.config.state_database) as state:
        assert state.get(case.now.date()).status == "running"
    # No provider setup exists in the second interpreter. Recovery must work
    # solely with the SQLite row, release, frozen snapshot, and completed output.
    script = Path(__file__).with_name("production_recovery_process.py")
    result = subprocess.run(
        [sys.executable, str(script), "recover", str(tmp_path / "crash-request.json")],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": str(Path(runner.__file__).parents[2])},
    )
    assert result.returncode == 0, result.stderr
    recovered = json.loads(result.stdout)
    assert recovered["action"] == "already_published", recovered
    assert recovered["delivery_id"] == original.delivery_id
    assert recovered["attempts"] == 1
    assert list(case.config.factor_batch.output_root.iterdir()) == [bundle]
    assert {p.name: p.read_bytes() for p in bundle.iterdir()} == before


def test_recovery_itself_can_exit_and_restart_again(production_case, monkeypatch):
    case = production_case
    bundle, original = _interrupt_after_publish(case, monkeypatch)
    again, _ = _interrupt_after_publish(case, monkeypatch)
    assert again == bundle and case.fetches == 1
    result = _tick(case)
    assert result.action == "already_published" and result.delivery_id == original.delivery_id
    assert result.attempts == 1 and case.fetches == 1


def test_orphan_is_reconciled_before_retry_timer(production_case, monkeypatch):
    case = production_case
    _, original = _interrupt_after_publish(case, monkeypatch)
    with ProductionState(case.config.state_database) as state:
        state.put(case.now.date(), case.release.release_id, "waiting_data", attempts=1,
                  next_retry_at=case.now + timedelta(minutes=30))
    result = _tick(case)
    assert result.action == "already_published" and result.delivery_id == original.delivery_id
    assert case.fetches == 1


def test_other_dates_do_not_create_a_same_day_conflict(production_case, monkeypatch):
    case = production_case
    bundle, original = _interrupt_after_publish(case, monkeypatch)
    other_date = next_regular_session(case.now.date())
    frame = pl.read_parquet(bundle / "factors.parquet").with_columns(
        pl.lit(other_date).alias("asof_date")
    )
    factor_batch.publish_factor_batch(
        frame, case.config.factor_batch.output_root, source=original.source, model=original.model,
        input_metadata=original.input.model_copy(update={
            "universe_sha256": factor_batch.factor_universe_sha256(
                frame.select("security_id", "symbol", "asof_date", "eligible")
            ),
        }),
        time_metadata=original.time.model_copy(update={
            "minimum_asof_date": other_date, "maximum_asof_date": other_date,
        }),
    )
    result = _tick(case)
    assert result.action == "already_published" and result.delivery_id == original.delivery_id
    assert case.fetches == 1 and len(list(case.config.factor_batch.output_root.iterdir())) == 2


def test_degraded_original_keeps_its_unscorable_row_on_recovery(production_case, monkeypatch):
    case = production_case
    payload = case.config.model_dump()
    # One missing stock in this five-stock synthetic pool represents tolerated
    # local loss. Production's 5% full-pool tolerance is not changed.
    payload["quality"]["max_computational_missing_fraction"] = 0.20
    payload["inference"]["minimum_eligible_rows"] = 4
    case.config = ProductionServiceConfig.model_validate(payload)
    bars = pl.read_parquet(case.source.sources.bars).filter(
        ~((pl.col("security_id") == "sec-1") & (pl.col("trade_date") == case.now.date()))
    )
    bars.write_parquet(case.source.sources.bars)
    bundle, original = _interrupt_after_publish(case, monkeypatch)
    result = _tick(case)
    assert result.action == "already_published", result.error
    assert result.quality["status"] == "degraded" and case.fetches == 1
    assert result.delivery_id == original.delivery_id
    factors = pl.read_parquet(bundle / "factors.parquet")
    missing = factors.filter(pl.col("security_id") == "sec-1")
    assert missing["score"].item() is None and not missing["eligible"].item()
