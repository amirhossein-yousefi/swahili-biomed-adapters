"""Lightweight logging: stdlib + optional W&B."""
from __future__ import annotations

import logging
import os

_INITIALIZED = False


def setup_logging(level: str = "INFO") -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper()),
    )
    _INITIALIZED = True


def maybe_init_wandb(cfg) -> "wandb.Run | None":
    if not getattr(getattr(cfg, "run", cfg).wandb, "enabled", False):
        return None
    try:
        import wandb
    except ImportError:
        logging.getLogger(__name__).warning("wandb requested but not installed")
        return None
    return wandb.init(
        project=cfg.run.wandb.project,
        config=dict(cfg) if not hasattr(cfg, "to_container") else cfg.to_container(resolve=True),
        dir=os.environ.get("WANDB_DIR", "./wandb"),
    )
