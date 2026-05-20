"""Training callbacks: periodic dev-perplexity eval + JSON dump."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

log = logging.getLogger(__name__)


class JsonMetricsDump(TrainerCallback):
    """At each evaluation, append the metrics dict to results/runs/<hash>/metrics.jsonl."""

    def __init__(self, out_dir: str | Path):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out / "metrics.jsonl"

    def on_evaluate(self, args, state, control, metrics: dict | None = None, **kwargs):
        if metrics:
            with open(self.path, "a") as f:
                f.write(json.dumps({"step": state.global_step, **metrics}) + "\n")


class WandbLogger(TrainerCallback):
    """Mirror the trainer's HF logs to W&B if a run is active."""

    def __init__(self, run: Any | None):
        self.run = run

    def on_log(self, args, state, control, logs: dict | None = None, **kwargs):
        if self.run is None or not logs:
            return
        try:
            self.run.log({**logs, "step": state.global_step})
        except Exception as e:  # don't kill training on log failure
            log.warning("wandb log failed: %s", e)
