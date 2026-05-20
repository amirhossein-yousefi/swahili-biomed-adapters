"""Topic classification eval (MedlinePlus-Swahili) — macro/weighted F1."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from datasets import Dataset
from omegaconf import DictConfig

from multilingual.backbone import load_backbone
from multilingual.compose import build_composition
from multilingual.data import medlineplus
from multilingual.eval.classification import evaluate_classification
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

    path = Path(cfg.eval.get("data_path", "data/processed/medlineplus_swh.jsonl"))
    if not path.exists():
        medlineplus.write_topic_classification_jsonl(path)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ds = Dataset.from_list(rows)
    id2label = {i: c for i, c in enumerate(medlineplus.CATEGORIES)}
    metrics = evaluate_classification(bb.model, bb.tokenizer, ds, id2label,
                                      batch_size=cfg.eval.get("batch_size", 16))
    out = Path(cfg.run.output_dir) / f"eval_classification_{cfg.compose.id}"
    write_run_artifacts(out, cfg, metrics={k: v for k, v in metrics.items()
                                           if not k.startswith("_")})
    log.info("topic-class macro-F1=%.4f weighted-F1=%.4f n=%d",
             metrics["macro_f1"], metrics["weighted_f1"], metrics["n"])


if __name__ == "__main__":
    main()
