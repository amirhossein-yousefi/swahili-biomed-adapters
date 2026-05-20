"""Build adapter configs (Pfeiffer / Houlsby / Parallel / LoRA) via the `adapters` lib.

All MAD-X primitives — Pfeiffer placement, invertible NICE coupling on
embeddings, Houlsby placement, parallel composition (He et al. 2021),
LoRA — come from the library directly. We never re-implement them.
"""
from __future__ import annotations

import logging
from typing import Any

from adapters import (
    DoubleSeqBnConfig,      # Houlsby-style bottleneck on both attn + FFN sub-layers
    DoubleSeqBnInvConfig,   # Houlsby + invertible NICE on embeddings
    LoRAConfig,
    ParBnConfig,            # Parallel composition (He et al. 2021)
    SeqBnConfig,            # Pfeiffer-style sequential bottleneck (post-FFN)
    SeqBnInvConfig,         # SeqBn + invertible NICE on embeddings (the LA configuration)
)

log = logging.getLogger(__name__)


def build_adapter_config(cfg: dict | Any):
    """Translate a YAML adapter spec into an `adapters` AdapterConfig.

    Recognised keys (all under `cfg.adapter`):
      type: pfeiffer | pfeiffer_inv | houlsby | houlsby_inv | parallel | lora
      reduction_factor: int (default 16)
      non_linearity:    str (default "gelu")
      inv_reduction_factor: int (default 2, only when *_inv)
      r:               int (LoRA rank, default 16)
      alpha:           int (LoRA alpha, default 16)
    """
    t = str(getattr(cfg, "type", cfg.get("type", "pfeiffer"))).lower()
    rf = int(getattr(cfg, "reduction_factor", cfg.get("reduction_factor", 16)))
    nl = str(getattr(cfg, "non_linearity", cfg.get("non_linearity", "gelu")))
    inv = bool(getattr(cfg, "invertible", cfg.get("invertible", False)))
    inv_rf = int(getattr(cfg, "inv_reduction_factor", cfg.get("inv_reduction_factor", 2)))

    if t in {"pfeiffer", "seqbn"}:
        if inv:
            return SeqBnInvConfig(reduction_factor=rf, non_linearity=nl,
                                  inv_adapter="nice", inv_adapter_reduction_factor=inv_rf)
        return SeqBnConfig(reduction_factor=rf, non_linearity=nl)
    if t == "pfeiffer_inv":
        return SeqBnInvConfig(reduction_factor=rf, non_linearity=nl,
                              inv_adapter="nice", inv_adapter_reduction_factor=inv_rf)
    if t == "houlsby":
        if inv:
            return DoubleSeqBnInvConfig(reduction_factor=rf, non_linearity=nl,
                                        inv_adapter="nice", inv_adapter_reduction_factor=inv_rf)
        return DoubleSeqBnConfig(reduction_factor=rf, non_linearity=nl)
    if t == "parallel":
        return ParBnConfig(reduction_factor=rf, non_linearity=nl)
    if t == "lora":
        r = int(getattr(cfg, "r", cfg.get("r", 16)))
        alpha = int(getattr(cfg, "alpha", cfg.get("alpha", 16)))
        return LoRAConfig(r=r, alpha=alpha)
    raise ValueError(f"Unknown adapter type: {t}")


def add_adapter(model, name: str, adapter_cfg, *, train: bool = True) -> None:
    """Attach an adapter under a name; activate train mode for that adapter only."""
    model.add_adapter(name, config=adapter_cfg)
    if train:
        model.train_adapter(name)
        log.info("Adapter %r added and set trainable", name)


def add_head(model, task_type: str, head_name: str, *, num_labels: int | None = None,
             num_choices: int | None = None, id2label: dict | None = None) -> None:
    """Attach a typed prediction head appropriate to the task.

    task_type: 'mcqa' | 'ner' | 'classification' | 'mlm'
    """
    if task_type == "mcqa":
        if num_choices is None:
            raise ValueError("mcqa head requires num_choices")
        model.add_multiple_choice_head(head_name, num_choices=num_choices, overwrite_ok=True)
    elif task_type == "ner":
        if num_labels is None:
            raise ValueError("ner head requires num_labels")
        model.add_tagging_head(head_name, num_labels=num_labels, id2label=id2label, overwrite_ok=True)
    elif task_type == "classification":
        if num_labels is None:
            raise ValueError("classification head requires num_labels")
        model.add_classification_head(head_name, num_labels=num_labels, id2label=id2label,
                                      overwrite_ok=True)
    elif task_type == "mlm":
        model.add_masked_lm_head(head_name, overwrite_ok=True)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
