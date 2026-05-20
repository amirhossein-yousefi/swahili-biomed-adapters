"""§8 catastrophic-forgetting probe.

Evaluate the LA-only (C2) stack and the LA→DA (intermediate, no TA) stack on
MasakhaNER-Swahili. If F1 drops by >2 points under LA→DA vs LA-only,
forgetting is occurring — a known finding from Stickland et al. 2021 that we
explicitly test in the African low-resource setting (§10.1).

This module trains a tiny MasakhaNER TA on each of the two stacks (using
a shared train split) and reports F1 + delta. The "TA" here is a thin probe
matching the §8 mitigation: lower the DA learning rate or shrink the bottleneck
if degradation is severe.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..adapter_setup import build_adapter_config
from ..training.task_trainer import build_task_run, save_ta_only
from .ner import evaluate_ner

log = logging.getLogger(__name__)


def run_forgetting_probe(*, backbone_name: str, la_path: str, da_path: str | None,
                         masakhaner_dsd, label_list: list[str],
                         output_root: str | Path, adapter_cfg=None, seed: int = 42,
                         num_epochs: float = 3.0) -> dict:
    """Train a tiny NER TA on each of {LA-only, LA→DA} and report ΔF1."""
    cfg = adapter_cfg or build_adapter_config({"type": "pfeiffer", "reduction_factor": 16})
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    id2label = dict(enumerate(label_list))

    results: dict[str, dict[str, float]] = {}

    for tag, da in [("la_only", None), ("la_da", da_path)]:
        if tag == "la_da" and da is None:
            continue
        out = out_root / tag
        run = build_task_run(
            backbone_name=backbone_name, ta_name=f"forget_{tag}",
            adapter_cfg=cfg, task_type="ner",
            train_dataset=masakhaner_dsd["train"],
            eval_dataset=masakhaner_dsd.get("validation"),
            la_path=la_path, da_path=da,
            num_labels=len(label_list), id2label=id2label,
            output_dir=out, num_train_epochs=num_epochs, seed=seed,
        )
        run.trainer.train()
        save_ta_only(run)
        ev = evaluate_ner(run.model, run.tokenizer, masakhaner_dsd["test"], id2label)
        results[tag] = {"f1_strict": ev["f1_strict"], "f1_relaxed": ev["f1_relaxed"]}
        log.info("[%s] MasakhaNER F1 strict=%.4f relaxed=%.4f", tag,
                 ev["f1_strict"], ev["f1_relaxed"])

    delta = None
    if "la_da" in results:
        delta = results["la_da"]["f1_strict"] - results["la_only"]["f1_strict"]
    return {**results, "delta_strict": delta,
            "verdict": ("forgetting"
                        if (delta is not None and delta < -0.02) else "no_clear_forgetting")}
