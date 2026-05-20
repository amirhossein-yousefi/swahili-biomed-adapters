"""Baseline (d): full continued pretraining (DAPT) on Swa-Med corpus.

This is intentionally NOT adapter-based — it full-fine-tunes the entire backbone
on a small Swahili biomedical corpus assembled from OpenWHO + MedlinePlus +
NLLB-translated PubMed abstracts (§5.6 d). Most expensive baseline (~2 days
wall-clock on DGX Spark per §6.2).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .callbacks import JsonMetricsDump, WandbLogger

log = logging.getLogger(__name__)


@dataclass
class FullFtRun:
    model: Any
    tokenizer: Any
    trainer: Trainer
    out_dir: Path


def build_full_dapt_run(*, backbone_name: str, train_dataset, eval_dataset=None,
                        output_dir: str | Path = "./checkpoints/baseline_d_full_dapt",
                        num_train_epochs: float = 3.0,
                        per_device_batch_size: int = 16, grad_accum: int = 4,
                        learning_rate: float = 5e-5,
                        warmup_steps: int = 1000, weight_decay: float = 0.01,
                        bf16: bool = True, save_steps: int = 5000, eval_steps: int = 5000,
                        logging_steps: int = 100, seed: int = 42,
                        wandb_run=None) -> FullFtRun:
    from .mlm_trainer import make_mlm_collator

    tok = AutoTokenizer.from_pretrained(backbone_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(backbone_name)
    collator = make_mlm_collator(tok)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        bf16=bf16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=eval_steps,
        report_to="none",
        seed=seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=collator, tokenizer=tok,
        callbacks=[JsonMetricsDump(out_dir), WandbLogger(wandb_run)] if wandb_run else
                  [JsonMetricsDump(out_dir)],
    )
    return FullFtRun(model=model, tokenizer=tok, trainer=trainer, out_dir=out_dir)
