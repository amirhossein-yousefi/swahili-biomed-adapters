"""§7.1 — NER evaluation: span-F1 strict + relaxed + per-entity-type.

  python scripts/eval_ner.py compose=c4_la_da_ta eval=ner
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from multilingual.backbone import load_backbone
from multilingual.compose import build_composition
from multilingual.data import bc5cdr
from multilingual.eval.ner import evaluate_ner
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

    swh_test_path = Path(cfg.eval.get("data_path", "data/processed/bc5cdr_swh_test.jsonl"))
    if not swh_test_path.exists():
        bc5cdr.write_swahili_test_jsonl(swh_test_path, en_split="test")
    eval_ds = bc5cdr.load_swahili_test_jsonl(swh_test_path)

    metrics = evaluate_ner(bb.model, bb.tokenizer, eval_ds, bc5cdr.ID2LABEL,
                           batch_size=cfg.eval.get("batch_size", 16),
                           max_length=cfg.eval.get("max_length", 256))

    if cfg.eval.get("compare_to"):
        try:
            other = json.loads(Path(cfg.eval.compare_to).read_text())
            our_f1 = np.asarray(metrics["_per_example"]["f1_per_sentence"])
            their_f1 = np.asarray(other["_per_example"]["f1_per_sentence"])
            metrics["paired_bootstrap_vs_compare"] = paired_bootstrap(our_f1, their_f1)
        except Exception as e:
            log.warning("compare_to failed: %s", e)

    if cfg.eval.get("clinician_validated_subset"):
        ids_path = Path(cfg.eval.clinician_validated_subset)
        try:
            keep = set(json.loads(ids_path.read_text()))
            f1s = metrics["_per_example"]["f1_per_sentence"]
            sub = [f1s[i] for i in range(len(f1s)) if i in keep]
            if sub:
                metrics["clinician_subset_f1_strict_mean"] = float(np.mean(sub))
                metrics["clinician_subset_n"] = len(sub)
        except Exception as e:
            log.warning("clinician_validated_subset failed: %s", e)

    out = Path(cfg.run.output_dir) / f"eval_ner_{cfg.compose.id}"
    write_run_artifacts(out, cfg, metrics={k: v for k, v in metrics.items()
                                           if not k.startswith("_")})
    log.info("NER strict-F1=%.4f  relaxed-F1=%.4f  (n=%d)",
             metrics["f1_strict"], metrics["f1_relaxed"], metrics["n"])


if __name__ == "__main__":
    main()
