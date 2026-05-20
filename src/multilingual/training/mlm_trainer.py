"""Shared MLM trainer for LA (§5.2) and DA (§5.3).

Both use the same loop: frozen backbone + a single trainable adapter (+optional
invertible) + an MLM head. We rely on `adapters.AdapterTrainer` so adapter +
head params are saved/loaded correctly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from adapters import AdapterTrainer
from datasets import IterableDataset
from transformers import TrainingArguments

from ..adapter_setup import add_adapter, add_head, build_adapter_config
from ..backbone import assert_only_adapters_trainable, load_backbone
from ..data.mlm_collator import make_mlm_collator
from .callbacks import JsonMetricsDump, WandbLogger

log = logging.getLogger(__name__)


@dataclass
class MlmRun:
    """Holds the fully-built model + tokenizer + trainer ready for `.train()`."""
    model: Any
    tokenizer: Any
    trainer: AdapterTrainer
    out_dir: Path
    resume_from_checkpoint: bool | str | None = None  # passed to trainer.train(...) by caller


def build_mlm_run(*, backbone_name: str, adapter_name: str, adapter_cfg, train_dataset,
                  eval_dataset=None, base_stack: list[str] | None = None,
                  base_stack_paths: dict[str, str] | None = None,
                  output_dir: str | Path = "./checkpoints/mlm_run",
                  num_train_epochs: float = 3.0, max_steps: int = -1,
                  per_device_batch_size: int = 16, grad_accum: int = 4,
                  learning_rate: float = 1e-4, warmup_steps: int = 2000,
                  weight_decay: float = 0.01, max_grad_norm: float = 1.0,
                  bf16: bool = True, flash_attn: bool = True,
                  span_mask: bool = False,
                  save_steps: int = 10_000, eval_steps: int = 5_000,
                  logging_steps: int = 100, seed: int = 42,
                  wandb_run=None,
                  resume_from_checkpoint: bool | str | None = None) -> MlmRun:
    """Common builder for LA and DA training.

    For DA mode (a) — train DA on top of a frozen LA — pass:
      base_stack = ["LA_en"]
      base_stack_paths = {"LA_en": "/path/to/la_en"}
    For mode (b) and for LA training, leave both as None.
    """
    bb = load_backbone(backbone_name, freeze_base=True, bf16=bf16, flash_attn=flash_attn)
    model, tok = bb.model, bb.tokenizer

    # Optionally load and freeze an existing LA underneath (DA "mode a")
    if base_stack:
        for nm in base_stack:
            assert base_stack_paths and nm in base_stack_paths, f"missing path for {nm}"
            model.load_adapter(base_stack_paths[nm], load_as=nm, set_active=False)

    # Add the new adapter being trained
    cfg_obj = build_adapter_config(adapter_cfg) if not hasattr(adapter_cfg, "__class__") \
        or not adapter_cfg.__class__.__name__.endswith("Config") else adapter_cfg
    add_adapter(model, adapter_name, cfg_obj, train=True)

    # MLM head shares projection with embedding for invertible-adapter recipe
    add_head(model, "mlm", adapter_name)

    # Active = stack(base_stack..., adapter_name) so the new adapter sits on top
    if base_stack:
        from adapters.composition import Stack
        model.active_adapters = Stack(*base_stack, adapter_name)
    else:
        model.set_active_adapters(adapter_name)

    assert_only_adapters_trainable(model)

    collator = make_mlm_collator(tok, span=span_mask)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        bf16=bf16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=5,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=eval_steps,
        report_to="none",  # we handle wandb via callback
        seed=seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = AdapterTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tok,
        callbacks=[JsonMetricsDump(out_dir), WandbLogger(wandb_run)] if wandb_run else
                  [JsonMetricsDump(out_dir)],
    )
    return MlmRun(model=model, tokenizer=tok, trainer=trainer, out_dir=out_dir,
                  resume_from_checkpoint=resume_from_checkpoint)


def save_adapter_only(run: MlmRun, name: str | None = None) -> Path:
    """Save just the adapter weights (not the frozen backbone) plus its head."""
    name = name or next(iter(run.model.adapters_config.adapters.keys()))
    out = run.out_dir / name
    out.mkdir(parents=True, exist_ok=True)
    run.model.save_adapter(str(out), name)
    run.tokenizer.save_pretrained(str(out))
    log.info("Saved adapter %r to %s", name, out)
    return out
