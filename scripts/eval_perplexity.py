"""Held-out MLM perplexity sanity check on OpenWHO Swahili.

  python scripts/eval_perplexity.py compose=c2_la_only
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from multilingual.backbone import load_backbone
from multilingual.compose import build_composition
from multilingual.data import openwho
from multilingual.eval.perplexity import held_out_mlm_perplexity
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
        build_composition(bb.model, cfg.compose.id, paths)

    eval_path = Path(cfg.eval.get("data_path", "data/processed/openwho_swh_eval.txt"))
    if not eval_path.exists():
        openwho.held_out_eval_split(eval_path, n_sentences=cfg.eval.get("n_sentences", 2000))
    sents = eval_path.read_text().splitlines()

    metrics = held_out_mlm_perplexity(bb.model, bb.tokenizer, sents,
                                      max_length=cfg.eval.get("max_length", 512),
                                      batch_size=cfg.eval.get("batch_size", 8),
                                      mlm_prob=cfg.eval.get("mlm_prob", 0.15))
    out = Path(cfg.run.output_dir) / f"eval_perplexity_{cfg.compose.id}"
    write_run_artifacts(out, cfg, metrics=metrics)
    log.info("Held-out MLM ppl=%.4f loss=%.4f n=%d",
             metrics["perplexity"], metrics["loss"], metrics["n_masked_tokens"])


if __name__ == "__main__":
    main()
