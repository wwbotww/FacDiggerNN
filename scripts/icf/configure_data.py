"""Materialize isolated, reviewable ICF data paths without changing research semantics."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from facdigger.data.config import load_dataset_build_config
from facdigger.data.providers.eodhd.config import load_eodhd_config
from facdigger.research.transformer_config import load_transformer_comparison_config
from facdigger.training.runtime import atomic_write_text


def configure(root: Path, code: Path) -> list[Path]:
    if not root.is_absolute() or str(root.resolve()).startswith("/afs/"):
        raise ValueError("ICF root must be an absolute persistent path outside AFS")
    root, code = root.resolve(), code.resolve()
    configs = root / "configs"
    provider = load_eodhd_config(code / "configs/data/eodhd_historical_liquid.yaml")
    provider = provider.model_copy(
        update={
            "output_dir": root / "data/bronze/eodhd_us_historical_liquid",
            "cache_dir": root / "cache/eodhd_historical_liquid",
            "state_dir": root / "state/eodhd_historical_liquid",
        }
    )
    dataset = load_dataset_build_config(
        code / "configs/datasets/eodhd_historical_liquid_transformer.yaml"
    )
    dataset = dataset.model_copy(
        update={
            "sources": dataset.sources.model_validate(
                {
                    key: provider.output_dir / path.name if path is not None else None
                    for key, path in dataset.sources.model_dump().items()
                }
            ),
            "output_root": root / "inputs/snapshots",
        }
    )
    research = load_transformer_comparison_config(
        code / "configs/research/finance_transformer_streamlined.yaml"
    )
    research = research.model_copy(
        update={
            "base_dataset_config": configs / "dataset.yaml",
            "snapshot_output_root": root / "inputs/snapshots",
            "output_root": root / "artifacts/transformer_comparison",
            "admission_report": root / "artifacts/benchmarks/finance-transformer-icf.json",
            "experiments": research.experiments.model_validate(
                {key: code / path for key, path in research.experiments.model_dump().items()}
            ),
        }
    )
    payloads = {
        configs / name: config.model_dump(mode="json")
        for name, config in (
            ("eodhd.yaml", provider),
            ("dataset.yaml", dataset),
            ("transformer.yaml", research),
        )
    }
    for path, payload in payloads.items():
        if path.exists() and yaml.safe_load(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(
                f"existing deployment config differs; review without overwriting: {path}"
            )
    for path, payload in payloads.items():
        if not path.exists():
            atomic_write_text(path, yaml.safe_dump(payload, allow_unicode=True, sort_keys=True))
    return list(payloads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    for output in configure(args.root, Path(__file__).resolve().parents[2]):
        print(output)
