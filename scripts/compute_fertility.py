"""§2.4 — fertility table on Swahili held-out slice.

  python scripts/compute_fertility.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from multilingual.data import openwho
from multilingual.tokenization.fertility import DEFAULT_TOKENIZERS, fertility_table
from multilingual.utils.logging import setup_logging

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    eval_path = Path(cfg.get("eval_path", "data/processed/openwho_swh_eval.txt"))
    if not eval_path.exists():
        openwho.held_out_eval_split(eval_path, n_sentences=cfg.get("n_sentences", 5_000))
    sents = eval_path.read_text().splitlines()
    table = fertility_table(sents, tokenizers=cfg.get("tokenizers", DEFAULT_TOKENIZERS))
    out = Path(cfg.get("output_dir", "results/tables")) / "fertility.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2))
    for row in table:
        log.info("%-40s subwords/word = %.3f  (n=%d)", row["tokenizer"], row["fertility"],
                 row["sentences"])


if __name__ == "__main__":
    main()
