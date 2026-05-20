"""Take a §5.5 composition config (C1–C10) and emit a ready-to-load model dir.

  python scripts/compose_and_save.py compose=c4_la_da_ta task=mcqa run.seed=42
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from multilingual.backbone import load_backbone
from multilingual.compose import build_composition, describe
from multilingual.utils.io import write_run_artifacts
from multilingual.utils.logging import setup_logging

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    bb = load_backbone(cfg.backbone.name, freeze_base=True, bf16=cfg.run.bf16,
                       flash_attn=cfg.run.flash_attn)
    paths = {k: str(v) for k, v in cfg.compose.adapters.items()}
    log.info("Composing %s — %s", cfg.compose.id, describe(cfg.compose.id))
    build_composition(bb.model, cfg.compose.id, paths,
                      fusion_ckpt=cfg.compose.get("fusion_ckpt"),
                      merge_method=cfg.compose.get("merge_method", "ties"))
    out = Path(cfg.run.output_dir) / f"composed_{cfg.compose.id}"
    out.mkdir(parents=True, exist_ok=True)
    bb.model.save_pretrained(str(out))
    bb.tokenizer.save_pretrained(str(out))
    write_run_artifacts(out, cfg, metrics={"compose": cfg.compose.id})
    log.info("Composed model saved to %s", out)


if __name__ == "__main__":
    main()
