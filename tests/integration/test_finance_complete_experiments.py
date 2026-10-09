from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event

import pytest
import torch
import yaml
from test_finance_diagnostic_prefix import _cached
from test_finance_interruptions import _equal
from test_finance_transformer_training import _config, _datasets

from facdigger.data.contracts import DataContractError
from facdigger.datasets.window import FinanceTransformerInferenceWindowDataset
from facdigger.research import finance_experiment_plan as planning
from facdigger.research import finance_experiment_runner as runner
from facdigger.research import finance_prefix_diagnostics as prefix
from facdigger.research.finance_experiment_config import (
    FinanceExperimentConfig,
    candidate_settings,
    experiment_cells,
    load_finance_experiment_config,
)
from facdigger.research.finance_experiment_state import file_inventory, verify_inventory
from facdigger.training import finance_transformer_engine as engine
from facdigger.training.runtime import (
    DatasetLocation,
    TrainingControl,
    TrainingPaused,
    TrainingRuntimeConfig,
)

ENVIRONMENT = {
    "python": "3.12.3 fixture", "machine": "fixture",
    "dependencies": [
        {"name": name, "installed_version": "fixture"}
        for name in ("numpy", "torch", "transformers")
    ],
    "torch": {"cuda_available": False, "cuda_device_name": None},
}
RUNTIME = TrainingRuntimeConfig(max_walltime_seconds=600, shutdown_margin_seconds=30)


def setup_cells(tmp_path, monkeypatch, candidate="statistics_mlp", *, seeds=(42,), epochs=3):
    torch.set_num_threads(1)
    train, selection = _datasets()
    raw = _config().model_dump(mode="json")
    raw["model"].update(dropout=0.1, statistics_windows=[5, 20])
    raw["training"].update(
        max_epochs=epochs, minimum_epochs=6 if epochs == 10 else epochs, patience=3,
    )
    config = _config().model_validate(raw)
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    cache = tmp_path / "cache"
    cached = {
        "F": _cached(train, cache / "F", windows=(5, 20)),
        "S": _cached(selection, cache / "S", windows=(5, 20)),
    }
    labelled = {"F": train.sample_rows, "S": selection.sample_rows}
    pools = {key: data.sample_rows for key, data in cached.items()}
    windows = {
        phase: FinanceTransformerInferenceWindowDataset(
            feature_store=data.feature_store, market_store=data.market_store,
            inference_index=pools[phase], channels=config.channels,
            market_channels=config.market_channels, context_length=32, primary_horizon=5,
        ) for phase, data in {"F": train, "S": selection}.items()
    }
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    checksums = tmp_path / "checksums.json"
    checksums.write_text("{}")
    spec = FinanceExperimentConfig.model_validate({
        "research_id": "fixture", "supervised_config": experiment_path,
        "seeds": list(seeds), "candidates": [candidate],
        "folds": [{
            "fold_id": "wf", "train_end": "2020-12-31",
            "valid_end": "2021-12-31", "test_end": "2022-12-31", "embargo_sessions": 20,
        }],
    })
    manifest = {
        "dataset_id": "fixture",
        "config": {"features": {"context_length": 32}},
        "input_file_hashes": {"fixture": "hash"},
    }
    protocol = {"dataset_id": "fixture"}
    requested = []

    def inputs(*args, phases=("F", "S"), **kwargs):
        assert phases == ("F", "S"), "V/test targets must never be requested"
        requested.append(phases)
        return manifest, protocol, labelled, pools

    monkeypatch.setattr(planning, "_fold_inputs", inputs)
    monkeypatch.setattr(
        planning, "collect_git_state", lambda *a: {"dirty": False, "commit": "fixture"},
    )
    monkeypatch.setattr(runner, "collect_environment", lambda: ENVIRONMENT)
    monkeypatch.setattr(
        runner, "window_datasets", lambda *a: windows,
    )
    runtime = TrainingRuntimeConfig(
        fold_snapshots={"wf": DatasetLocation(path=snapshot, checksums=checksums)},
    )

    def plan(name):
        root = tmp_path / name
        planning.freeze_experiment_plan(spec, runtime, root, repository_root=tmp_path)
        return root

    def run(root, seed=42, **kwargs):
        return runner.run_experiment_cell(
            root, f"wf/seed-{seed}/{candidate}", repository_root=tmp_path,
            runtime=RUNTIME, cache=cache, **kwargs,
        )

    return {
        "config": config, "spec": spec, "plan": plan, "run": run, "cache": cache,
        "cached": cached, "labelled": labelled, "pools": pools,
        "snapshot": snapshot, "checksums": checksums,
        "inputs": inputs, "protocol": protocol, "requested": requested,
        "datasets": windows,
    }


def last(root, candidate="statistics_mlp", seed=42):
    return root / f"cells/wf/seed-{seed}/{candidate}/checkpoints/last.pt"


def test_different_cells_wait_for_the_shared_binding_without_entering_training(
    tmp_path, monkeypatch,
):
    from facdigger.training import runtime

    setup = setup_cells(tmp_path, monkeypatch, seeds=(17, 42))
    root = setup["plan"]("concurrent")
    ready, waiting, release = Barrier(2), Event(), Event()
    original_write, original_sleep = runner.write_json, time.sleep

    def environment():
        ready.wait(timeout=5)
        return ENVIRONMENT

    def delayed_binding(path, value):
        if path.name == "execution.json":
            assert release.wait(timeout=5)
        original_write(path, value)

    def observed_sleep(delay):
        waiting.set()
        original_sleep(delay)

    class BeforeData(Exception):
        pass

    def stop_before_data(*args, **kwargs):
        raise BeforeData

    monkeypatch.setattr(runner, "collect_environment", environment)
    monkeypatch.setattr(runner, "write_json", delayed_binding)
    monkeypatch.setattr(runtime.time, "sleep", observed_sleep)
    monkeypatch.setattr(runner, "load_fold_inputs", stop_before_data)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(setup["run"], root, seed) for seed in (17, 42)]
        try:
            assert waiting.wait(timeout=3), "the second cell must wait for the shared binding"
        finally:
            release.set()
        for future in futures:
            with pytest.raises(BeforeData):
                future.result(timeout=5)
    for seed in (17, 42):
        cell = root / f"cells/wf/seed-{seed}/statistics_mlp"
        assert json.loads((cell / "cell.json").read_text())["error"] == "BeforeData: "
        assert not (cell / "checkpoints").exists()


@pytest.mark.parametrize(
    "candidate", ["statistics_linear", "statistics_mlp", "statistics_mlp_dropout03", "finance"],
)
def test_complete_cells_match_the_existing_trainer_and_seal_without_v(
    tmp_path, monkeypatch, candidate,
):
    setup = setup_cells(tmp_path, monkeypatch, candidate)
    root = setup["plan"]("complete")
    result = setup["run"](root)
    assert result["status"] == "complete" and result["epoch"] == 3
    assert result["global_step"] == 6 and result["validation_used"] is False
    data = setup["datasets"] if candidate == "finance" else setup["cached"]
    kind, dropout = candidate_settings(candidate)
    engine.train_finance_transformer(
        setup["config"], train_dataset=data["F"], valid_dataset=data["S"],
        train_labelled_rows=setup["labelled"]["F"], valid_labelled_rows=setup["labelled"]["S"],
        data_protocol=setup["protocol"], dataset_id="fixture",
        checkpoint_dir=tmp_path / "reference",
        control=TrainingControl(RUNTIME), diagnostic_model=kind, statistics_dropout=dropout,
    )
    expected = torch.load(tmp_path / "reference/last.pt", weights_only=False)
    actual = torch.load(last(root, candidate), weights_only=False)
    _equal(expected, actual)
    cell_root = last(root, candidate).parents[1]
    assert len(list((cell_root / "observations").glob("*.source.json"))) == 4
    assert all((cell_root / f"observations/epoch-{epoch}.json").exists() for epoch in range(4))
    assert not list(root.rglob("*-V-*")) and not (setup["cache"] / "V").exists()
    seal = runner.seal_experiment_selections(root, repository_root=tmp_path)
    assert seal["holdout_used"] is False and seal["validation_execution_implemented"] is False
    assert planning.experiment_status(root)["selections_sealed"] is True

    def forbidden(*a, **kw):
        raise AssertionError("completed cell must not start a trainer")

    monkeypatch.setattr(runner, "train_finance_transformer", forbidden)
    hashes = file_inventory(root, [p for p in root.rglob("*") if p.is_file()])
    assert setup["run"](root)["status"] == "complete"
    verify_inventory(root, hashes)
    assert runner.seal_experiment_selections(root, repository_root=tmp_path) == seal


@pytest.mark.parametrize("phase", ["observation", "update", "final_commit"])
def test_complete_cell_resumes_without_changing_optimizer_clock_or_rng(
    tmp_path, monkeypatch, phase,
):
    setup = setup_cells(tmp_path, monkeypatch)
    continuous = setup["plan"]("continuous")
    setup["run"](continuous)
    resumed = setup["plan"]("resumed")
    if phase == "observation":
        original = prefix._write_chunk

        def interrupted(path, tables):
            original(path, tables)
            if path.name == "epoch-1-F-0000":
                raise TrainingPaused("fixture observation", last(resumed))

        monkeypatch.setattr(prefix, "_write_chunk", interrupted)
        def restore():
            monkeypatch.setattr(prefix, "_write_chunk", original)
    elif phase == "update":
        controls = []

        def make_control(*a, **kw):
            control = TrainingControl(*a, **kw)
            controls.append(control)
            return control

        monkeypatch.setattr(runner, "TrainingControl", make_control)
        original = runner.append_progress

        def interrupted(path, event, **kwargs):
            original(path, event, **kwargs)
            if event["event"] == "optimizer_progress" and event["global_step"] == 1:
                controls[-1].request_stop("fixture update")

        monkeypatch.setattr(runner, "append_progress", interrupted)
        def restore():
            monkeypatch.setattr(runner, "append_progress", original)
    else:
        original = engine.save_checkpoint

        def interrupted(path, payload):
            original(path, payload)
            if path.name == "last.pt" and payload.get("progress", {}).get("finished"):
                raise TrainingPaused("fixture final commit", path)

        monkeypatch.setattr(engine, "save_checkpoint", interrupted)
        def restore():
            monkeypatch.setattr(engine, "save_checkpoint", original)
    assert setup["run"](resumed)["status"] == "paused"
    restore()
    assert setup["run"](resumed)["status"] == "complete"
    _equal(
        torch.load(last(continuous), weights_only=False),
        torch.load(last(resumed), weights_only=False),
    )
    if phase == "observation":
        events = [
            json.loads(line)
            for line in (last(resumed).parents[1] / "progress.jsonl").read_text().splitlines()
        ]
        assert any(
            row["event"] == "observation_chunk_reused" and row["epoch"] == 1 for row in events
        )


def test_prefix_import_is_read_only_and_retains_the_original_ten_epoch_clock(tmp_path, monkeypatch):
    setup = setup_cells(tmp_path, monkeypatch, "statistics_mlp_dropout03", epochs=10)
    source = tmp_path / "old-prefix"
    monkeypatch.setattr(prefix, "diagnostic_inputs", setup["inputs"])
    monkeypatch.setattr(
        prefix, "collect_git_state", lambda *a: {"dirty": False, "commit": "old-fixture"},
    )
    monkeypatch.setattr(prefix, "collect_environment", lambda: ENVIRONMENT)
    monkeypatch.setattr(
        prefix, "fixed_dates", lambda rows: rows["asof_date"].unique().sort().to_list(),
    )
    result = prefix.run_prefix_diagnostics(
        setup["snapshot"], setup["checksums"], setup["config"], setup["cache"], source,
        candidate="statistics_mlp", statistics_dropout=0.3, repository_root=tmp_path,
        budget_seconds=600, observation_scope="fit-selection",
        observation_precision="fp32", full_fit=True,
    )
    assert result["status"] == "observations_complete"
    protected = file_inventory(source, [p for p in source.rglob("*") if p.is_file()])
    root = setup["plan"]("imported")
    result = setup["run"](root, parent_prefix=source)
    assert result["status"] == "complete" and result["epoch"] >= 6
    verify_inventory(source, protected)
    checkpoint = torch.load(last(root, "statistics_mlp_dropout03"), weights_only=False)
    assert checkpoint["config"]["training"]["max_epochs"] == 10
    continuous = setup["plan"]("fresh")
    setup["run"](continuous)
    _equal(
        checkpoint,
        torch.load(last(continuous, "statistics_mlp_dropout03"), weights_only=False),
    )
    import_path = last(root, "statistics_mlp_dropout03").parents[1] / "parent/import.json"
    assert json.loads(import_path.read_text())["files"]["checkpoints/last.pt"] == (
        protected["checkpoints/last.pt"]
    )


def test_incomplete_matrix_corruption_and_mismatched_code_fail_closed(tmp_path, monkeypatch):
    setup = setup_cells(tmp_path, monkeypatch, seeds=(17, 42))
    root = setup["plan"]("matrix")
    with pytest.raises(DataContractError, match="every cell"):
        runner.seal_experiment_selections(root, repository_root=tmp_path)
    assert not (root / "selection-seal.json").exists()
    setup["run"](root, seed=17)
    with pytest.raises(DataContractError, match="every cell"):
        runner.seal_experiment_selections(root, repository_root=tmp_path)
    monkeypatch.setattr(
        planning, "collect_git_state", lambda *a: {"dirty": False, "commit": "changed"},
    )
    with pytest.raises(DataContractError, match="code commit"):
        setup["run"](root)
    monkeypatch.setattr(
        planning, "collect_git_state", lambda *a: {"dirty": False, "commit": "fixture"},
    )
    setup["run"](root)
    damaged = last(root).parents[1] / "chunks/epoch-1-F-0000/daily.parquet"
    with damaged.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(DataContractError, match="checksum"):
        setup["run"](root)
    with pytest.raises(DataContractError, match="checksum"):
        runner.seal_experiment_selections(root, repository_root=tmp_path)


def test_registered_reference_has_sixty_cells_and_rejects_unsafe_configs():
    config = load_finance_experiment_config(
        Path(__file__).resolve().parents[2] / "configs/research/finance_complete_reference.yaml"
    )
    assert len(experiment_cells(config)) == 60
    for update in ({"seeds": [42, 42]}, {"candidates": ["../bad"]}, {"seeds": [-1]}):
        with pytest.raises(ValueError):
            FinanceExperimentConfig.model_validate(config.model_dump() | update)


def test_real_snapshot_plan_and_cache_reentry_verify_phase_boundaries(tmp_path, monkeypatch):
    from test_finance_selection_snapshot import finance_config

    from facdigger.data.snapshots import build_dataset_snapshot
    from facdigger.research import finance_experiment_cache as preparation
    from facdigger.training.runtime import write_snapshot_checksums

    torch.set_num_threads(1)
    dataset_config = finance_config(tmp_path)
    snapshot, _ = build_dataset_snapshot(dataset_config)
    checksums = tmp_path / "checksums.json"
    write_snapshot_checksums(snapshot, checksums)
    raw = _config().model_dump(mode="json")
    raw["model"]["statistics_windows"] = [5, 20]
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(raw))
    spec = FinanceExperimentConfig.model_validate({
        "research_id": "real", "supervised_config": experiment_path,
        "seeds": [42], "candidates": ["statistics_linear"],
        "folds": [{"fold_id": "wf", **dataset_config.split.model_dump(mode="json")}],
    })
    runtime = TrainingRuntimeConfig(
        fold_snapshots={"wf": DatasetLocation(path=snapshot, checksums=checksums)},
    )
    monkeypatch.setattr(
        planning, "collect_git_state", lambda *a: {"dirty": False, "commit": "fixture"},
    )
    root = tmp_path / "experiment"
    plan = planning.freeze_experiment_plan(spec, runtime, root, repository_root=tmp_path)
    assert plan["identity"]["folds"]["wf"]["support"]["F"]["labelled_rows"] > 0
    assert set(plan["identity"]["folds"]["wf"]["support"]) == {"F", "S"}
    assert planning.freeze_experiment_plan(spec, runtime, root, repository_root=tmp_path) == plan
    original = preparation.cache_statistics
    calls = []

    def interrupt(dataset, path, **kwargs):
        calls.append(path.name)
        if path.name == ".S.pending":
            raise TrainingPaused("fixture cache", path)
        return original(dataset, path, **kwargs)

    monkeypatch.setattr(preparation, "cache_statistics", interrupt)
    assert preparation.prepare_experiment_cache(
        root, "wf", repository_root=tmp_path, runtime=RUNTIME,
    )["status"] == "paused"
    assert (root / "cache/wf/F/manifest.json").exists()
    monkeypatch.setattr(preparation, "cache_statistics", original)
    assert preparation.prepare_experiment_cache(
        root, "wf", repository_root=tmp_path, runtime=RUNTIME,
    )["status"] == "complete"
    assert not (root / "cache/wf/V").exists()

    def forbidden(*a, **kw):
        raise AssertionError("verified cache must be reused")

    monkeypatch.setattr(preparation, "cache_statistics", forbidden)
    assert preparation.prepare_experiment_cache(
        root, "wf", repository_root=tmp_path, runtime=RUNTIME,
    )["completed_phases"] == ["F", "S"]
    assert calls.count(".F.pending") == 1
    checksums.write_text("{}")
    with pytest.raises(DataContractError, match="inventory changed"):
        planning.load_fold_inputs(plan, "wf", seed=42)


def test_runtime_budget_and_environment_are_separate_from_scientific_identity(
    tmp_path, monkeypatch,
):
    setup = setup_cells(tmp_path, monkeypatch)
    root = setup["plan"]("budget")
    original = prefix._write_chunk

    def interrupted(path, tables):
        original(path, tables)
        raise TrainingPaused("fixture", last(root))

    monkeypatch.setattr(prefix, "_write_chunk", interrupted)
    assert setup["run"](root, cumulative_budget_seconds=600)["status"] == "paused"
    monkeypatch.setattr(prefix, "_write_chunk", original)
    audit_path = last(root).parents[1] / "cell.json"
    audit = json.loads(audit_path.read_text())
    # A killed attempt has a reserved cost and no settlement.
    audit["attempts"][-1].pop("elapsed_seconds")
    audit_path.write_text(json.dumps(audit))
    assert setup["run"](root, cumulative_budget_seconds=600)["stop_reason"] == "cumulative_budget"
    assert setup["run"](root, cumulative_budget_seconds=1200)["status"] == "complete"
    other = setup["plan"]("environment")
    monkeypatch.setattr(prefix, "_write_chunk", interrupted)
    assert setup["run"](other)["status"] == "paused"
    monkeypatch.setattr(prefix, "_write_chunk", original)
    monkeypatch.setattr(
        runner, "collect_environment", lambda: {**ENVIRONMENT, "python": "3.13.0"},
    )
    with pytest.raises(DataContractError, match="execution environment"):
        setup["run"](other)


def test_prefix_import_rejects_another_seed_before_copying_state(tmp_path, monkeypatch):
    setup = setup_cells(tmp_path, monkeypatch, seeds=(17, 42), epochs=10)
    source = tmp_path / "wrong-prefix"
    source.mkdir()
    origin = {
        "candidate": "statistics_mlp", "config": setup["config"].model_dump(mode="json"),
        "data_protocol": setup["protocol"], "prefix_epochs": 2,
    }
    (source / "diagnostic.json").write_text(json.dumps({
        "identity": origin, "status": "observations_complete", "holdout_used": False,
        "training_status": "paused",
    }))
    root = setup["plan"]("wrong-seed")
    with pytest.raises(DataContractError, match="matching completed"):
        setup["run"](root, seed=17, parent_prefix=source)
    assert not last(root, seed=17).exists()


@pytest.mark.parametrize("epoch,damage", [(1, "weights"), (3, "weights"), (1, "binding")])
def test_partial_observations_cannot_mix_chunks_from_changed_weights(
    tmp_path, monkeypatch, epoch, damage,
):
    setup = setup_cells(tmp_path, monkeypatch)
    root = setup["plan"]("partial-integrity")
    original = prefix._write_chunk

    def interrupted(path, tables):
        original(path, tables)
        if path.name == f"epoch-{epoch}-F-0000":
            raise TrainingPaused("fixture partial source", last(root))

    monkeypatch.setattr(prefix, "_write_chunk", interrupted)
    assert setup["run"](root)["status"] == "paused"
    monkeypatch.setattr(prefix, "_write_chunk", original)
    state_path = last(root).parents[1] / f"observations/epoch-{epoch}.pt"
    if damage == "weights":
        state = torch.load(state_path, weights_only=False)
        next(iter(state["model_state"].values())).add_(1)
        torch.save(state, state_path)
    else:
        state_path.with_suffix(".source.json").unlink()
    with pytest.raises(DataContractError, match="observation.*(weights|source binding)"):
        setup["run"](root)
