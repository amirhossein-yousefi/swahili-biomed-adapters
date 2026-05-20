"""§8 catastrophic-forgetting probe on MasakhaNER-Swahili.

  python scripts/eval_forgetting.py compose=c2_la_only
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from multilingual.adapter_setup import build_adapter_config
from multilingual.data import masakhaner
from multilingual.eval.forgetting import run_forgetting_probe
from multilingual.utils.io import write_run_artifacts
from multilingual.utils.logging import setup_logging
from multilingual.utils.seed import set_seed

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    set_seed(cfg.run.seed)

    dsd = masakhaner.load_masakhaner_swh()
    labels = masakhaner.label_list(dsd)
    out_dir = Path(cfg.run.output_dir) / "eval_forgetting"
    metrics = run_forgetting_probe(
        backbone_name=cfg.backbone.name,
        la_path=cfg.eval.la_path,
        da_path=cfg.eval.get("da_path"),
        masakhaner_dsd=dsd, label_list=labels,
        output_root=out_dir,
        adapter_cfg=build_adapter_config(cfg.adapter),
        seed=cfg.run.seed, num_epochs=cfg.eval.get("num_epochs", 3.0),
    )
    write_run_artifacts(out_dir, cfg, metrics=metrics)
    log.info("Forgetting verdict: %s  ΔF1_strict=%s",
             metrics.get("verdict"), metrics.get("delta_strict"))


if __name__ == "__main__":
    main()
