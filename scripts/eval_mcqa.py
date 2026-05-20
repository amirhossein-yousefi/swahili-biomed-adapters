"""§7.1 — MCQA evaluation: accuracy + ECE + Brier, stratified by subject.

  python scripts/eval_mcqa.py compose=c4_la_da_ta eval=mcqa
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from multilingual.backbone import load_backbone
from multilingual.compose import build_composition
from multilingual.data import mmlu_prox_swh
from multilingual.eval.mcqa import evaluate_mcqa
from multilingual.eval.significance import paired_bootstrap
from multilingual.utils.io import write_run_artifacts
from multilingual.utils.logging import setup_logging

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    bb = load_backbone(cfg.backbone.name, freeze_base=True, bf16=cfg.run.bf16,
                       flash_attn=cfg.run.flash_attn)
    if cfg.compose.id != "C1":
        paths = {k: str(v) for k, v in cfg.compose.adapters.items()}
        build_composition(bb.model, cfg.compose.id, paths,
                          fusion_ckpt=cfg.compose.get("fusion_ckpt"))

    eval_ds = mmlu_prox_swh.load_mmlu_prox_swahili_clinical(cfg.eval.get("split", "test"))
    metrics = evaluate_mcqa(bb.model, bb.tokenizer, eval_ds,
                            batch_size=cfg.eval.get("batch_size", 8),
                            max_length=cfg.eval.get("max_length", 256),
                            num_choices=cfg.eval.get("num_choices", 4))

    if cfg.eval.get("compare_to"):
        try:
            import json
            other = json.loads(Path(cfg.eval.compare_to).read_text())
            our_correct = (np.asarray(metrics["_per_example"]["predictions"]) ==
                           np.asarray(metrics["_per_example"]["labels"])).astype(float)
            their_correct = (np.asarray(other["_per_example"]["predictions"]) ==
                             np.asarray(other["_per_example"]["labels"])).astype(float)
            metrics["paired_bootstrap_vs_compare"] = paired_bootstrap(our_correct, their_correct)
        except Exception as e:
            log.warning("compare_to failed: %s", e)

    out = Path(cfg.run.output_dir) / f"eval_mcqa_{cfg.compose.id}"
    write_run_artifacts(out, cfg, metrics={k: v for k, v in metrics.items()
                                           if not k.startswith("_")},
                        predictions=[
                            {"index": i, "subject": s, "label": l, "pred": p, "probs": pr}
                            for i, (s, l, p, pr) in enumerate(zip(
                                metrics["_per_example"]["subjects"],
                                metrics["_per_example"]["labels"],
                                metrics["_per_example"]["predictions"],
                                metrics["_per_example"]["probs"]))])
    log.info("MCQA acc=%.4f  ECE=%.4f  Brier=%.4f  (n=%d)",
             metrics["accuracy"], metrics["ece"], metrics["brier"], metrics["n"])


if __name__ == "__main__":
    main()
