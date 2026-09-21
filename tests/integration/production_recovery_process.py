"""Isolated hard-exit worker for test_production_recovery (never live API IO)."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from facdigger.inference import factor_batch
from facdigger.inference.runner import run_signal_inference
from facdigger.production.calendar import production_window
from facdigger.production.config import ProductionServiceConfig
from facdigger.production.runner import _publish_guard, run_production_tick


def main():
    operation, request_path = sys.argv[1:]
    request = json.loads(Path(request_path).read_text())
    config = ProductionServiceConfig.model_validate(request["config"])
    observed = datetime.fromisoformat(request["now"])
    if operation == "recover":
        result = run_production_tick(config, now=observed, now_provider=lambda: observed)
        print(json.dumps(result.as_dict()))
        return
    assert operation == "publish-and-exit"

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return observed.astimezone(tz)

    factor_batch.datetime = Clock
    window = production_window(observed, config.schedule)
    run_signal_inference(
        config.model.release_root / config.model.release_id,
        dataset_dir=request["snapshot"], output_root=config.factor_batch.output_root,
        asof=window.target_date.isoformat(), device="cpu", delivery=config.factor_batch.delivery,
        before_publish=lambda: _publish_guard(window.cutoff_at, lambda: observed),
    )
    # No finally handlers, SQLite update, or graceful Python cleanup can run.
    os._exit(73)


if __name__ == "__main__":
    main()
