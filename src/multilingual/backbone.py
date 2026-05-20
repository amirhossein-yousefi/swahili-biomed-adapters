"""Load + freeze the backbone (AfroXLMR-large primary; XLM-R-large secondary).

Uses HuggingFace `adapters` (Poth 2023) so we get the AdapterModel variants with
typed heads (multiple-choice, tagging, classification) and native composition
primitives. The base model weights are frozen — only adapter+head params train.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .utils.hardware import best_dtype, maybe_enable_flash_attn

log = logging.getLogger(__name__)


@dataclass
class BackboneBundle:
    model: Any                      # adapters.AutoAdapterModel
    tokenizer: PreTrainedTokenizerBase
    name: str
    hidden_size: int


def load_backbone(name: str, *, freeze_base: bool = True, bf16: bool = True,
                  flash_attn: bool = True) -> BackboneBundle:
    """Load AdapterModel for `name` (e.g. 'Davlan/afro-xlmr-large') and freeze base."""
    dtype = best_dtype(prefer_bf16=bf16)
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    model_kwargs = maybe_enable_flash_attn(model_kwargs, flash_attn)

    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    model = AutoAdapterModel.from_pretrained(name, **model_kwargs)

    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False
        log.info("Backbone %s frozen (%s params)", name, sum(p.numel() for p in model.parameters()))

    hidden = getattr(model.config, "hidden_size", None)
    return BackboneBundle(model=model, tokenizer=tokenizer, name=name, hidden_size=hidden)


def trainable_param_summary(model) -> dict[str, int]:
    """Return counts (total, frozen, trainable, by_top_module) — used in smoke tests."""
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_module: dict[str, int] = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            top = n.split(".")[0]
            by_module[top] = by_module.get(top, 0) + p.numel()
    return {"total": total, "trainable": train, "frozen": total - train, "by_module": by_module}


def assert_only_adapters_trainable(model) -> None:
    """Sanity check used by smoke tests + every trainer's first step."""
    bad = [n for n, p in model.named_parameters()
           if p.requires_grad and "adapter" not in n.lower() and "heads" not in n.lower()
           and "lora" not in n.lower() and "invertible" not in n.lower()]
    if bad:
        raise RuntimeError(f"{len(bad)} non-adapter params require grad; first 5: {bad[:5]}")
