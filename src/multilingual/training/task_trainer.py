"""Supervised TA training (§5.4) — MAD-X recipe: TA on top of frozen [LA→DA] stack.

Task types: 'mcqa' | 'ner' | 'classification'.
For 'mcqa', uses an MCQA head + per-option scoring (concatenate question with each option).
For 'ner', uses a tagging head + token-level cross-entropy.
For 'classification', uses a single-label sequence-classification head.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from adapters import AdapterTrainer
from adapters.composition import Stack
from datasets import Dataset
from transformers import (
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    TrainingArguments,
)

from ..adapter_setup import add_adapter, add_head, build_adapter_config
from ..backbone import assert_only_adapters_trainable, load_backbone
from .callbacks import JsonMetricsDump, WandbLogger

log = logging.getLogger(__name__)


@dataclass
class TaskRun:
    model: Any
    tokenizer: PreTrainedTokenizerBase
    trainer: AdapterTrainer
    out_dir: Path
    task_type: str


def build_task_run(*, backbone_name: str, ta_name: str, adapter_cfg,
                   task_type: str, train_dataset: Dataset, eval_dataset: Dataset | None,
                   la_path: str | None = None, da_path: str | None = None,
                   num_labels: int | None = None, num_choices: int | None = None,
                   id2label: dict | None = None,
                   output_dir: str | Path = "./checkpoints/ta_run",
                   num_train_epochs: float = 5.0,
                   per_device_batch_size: int = 16, grad_accum: int = 1,
                   learning_rate: float = 1e-4, warmup_ratio: float = 0.06,
                   weight_decay: float = 0.01, max_grad_norm: float = 1.0,
                   bf16: bool = True, flash_attn: bool = True,
                   logging_steps: int = 50, eval_steps: int = 500,
                   seed: int = 42, wandb_run=None,
                   max_length: int = 256) -> TaskRun:
    bb = load_backbone(backbone_name, freeze_base=True, bf16=bf16, flash_attn=flash_attn)
    model, tok = bb.model, bb.tokenizer

    base_stack: list[str] = []
    if la_path:
        model.load_adapter(la_path, load_as="LA", set_active=False)
        base_stack.append("LA")
    if da_path:
        model.load_adapter(da_path, load_as="DA", set_active=False)
        base_stack.append("DA")

    cfg_obj = build_adapter_config(adapter_cfg)
    add_adapter(model, ta_name, cfg_obj, train=True)
    add_head(model, task_type, ta_name, num_labels=num_labels, num_choices=num_choices,
             id2label=id2label)
    if base_stack:
        model.active_adapters = Stack(*base_stack, ta_name)
    else:
        model.set_active_adapters(ta_name)

    assert_only_adapters_trainable(model)

    train_proc, eval_proc, collator = _prepare(task_type, tok, train_dataset, eval_dataset,
                                               num_choices=num_choices, max_length=max_length)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        bf16=bf16,
        logging_steps=logging_steps,
        eval_strategy="steps" if eval_proc is not None else "no",
        eval_steps=eval_steps,
        save_strategy="steps" if eval_proc is not None else "no",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=eval_proc is not None,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none",
        seed=seed,
        remove_unused_columns=False,
    )
    trainer = AdapterTrainer(
        model=model, args=args,
        train_dataset=train_proc, eval_dataset=eval_proc,
        data_collator=collator, tokenizer=tok,
        callbacks=[JsonMetricsDump(out_dir), WandbLogger(wandb_run)] if wandb_run else
                  [JsonMetricsDump(out_dir)],
    )
    return TaskRun(model=model, tokenizer=tok, trainer=trainer, out_dir=out_dir, task_type=task_type)


def save_ta_only(run: TaskRun, name: str | None = None) -> Path:
    name = name or next(iter(run.model.adapters_config.adapters.keys()))
    out = run.out_dir / name
    out.mkdir(parents=True, exist_ok=True)
    run.model.save_adapter(str(out), name)
    run.tokenizer.save_pretrained(str(out))
    return out


# --------------- task-specific preprocessing ---------------

def _prepare(task_type: str, tok, train_ds, eval_ds, *, num_choices, max_length):
    if task_type == "mcqa":
        return _prepare_mcqa(tok, train_ds, eval_ds, num_choices=num_choices, max_length=max_length)
    if task_type == "ner":
        return _prepare_ner(tok, train_ds, eval_ds, max_length=max_length)
    if task_type == "classification":
        return _prepare_classification(tok, train_ds, eval_ds, max_length=max_length)
    raise ValueError(task_type)


def _prepare_mcqa(tok, train_ds, eval_ds, *, num_choices: int, max_length: int):
    def encode(batch):
        first, second = [], []
        for q, opts in zip(batch["question"], batch["options"]):
            if len(opts) < num_choices:
                opts = list(opts) + [""] * (num_choices - len(opts))
            opts = opts[:num_choices]
            for opt in opts:
                first.append(q)
                second.append(opt)
        enc = tok(first, second, truncation=True, max_length=max_length)
        # group num_choices contiguous rows together
        out = {k: [v[i:i + num_choices] for i in range(0, len(v), num_choices)]
               for k, v in enc.items()}
        out["labels"] = batch["answer"]
        return out

    train_p = train_ds.map(encode, batched=True, remove_columns=train_ds.column_names)
    eval_p = eval_ds.map(encode, batched=True, remove_columns=eval_ds.column_names) if eval_ds else None
    collator = MultipleChoiceCollator(tok)
    return train_p, eval_p, collator


def _prepare_ner(tok, train_ds, eval_ds, *, max_length: int):
    def encode(ex):
        enc = tok(ex["tokens"], is_split_into_words=True, truncation=True, max_length=max_length)
        word_ids = enc.word_ids()
        labels: list[int] = []
        prev = None
        for wi in word_ids:
            if wi is None:
                labels.append(-100)
            elif wi != prev:
                labels.append(int(ex["tags"][wi]))
            else:
                labels.append(int(ex["tags"][wi]))
            prev = wi
        enc["labels"] = labels
        return enc

    train_p = train_ds.map(encode, remove_columns=train_ds.column_names)
    eval_p = eval_ds.map(encode, remove_columns=eval_ds.column_names) if eval_ds else None
    collator = DataCollatorForTokenClassification(tok)
    return train_p, eval_p, collator


def _prepare_classification(tok, train_ds, eval_ds, *, max_length: int):
    def encode(b):
        enc = tok(b["text"], truncation=True, max_length=max_length)
        enc["labels"] = b["label"]
        return enc

    train_p = train_ds.map(encode, batched=True, remove_columns=train_ds.column_names)
    eval_p = eval_ds.map(encode, batched=True, remove_columns=eval_ds.column_names) if eval_ds else None
    collator = DataCollatorWithPadding(tok)
    return train_p, eval_p, collator


class MultipleChoiceCollator:
    """Pads (B, num_choices, L) batches for the multiple-choice head."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tok = tokenizer

    def __call__(self, features):
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        n_choices = len(features[0]["input_ids"])
        flat = []
        for f in features:
            for c in range(n_choices):
                flat.append({k: f[k][c] for k in ("input_ids", "attention_mask")})
        padded = self.tok.pad(flat, return_tensors="pt")
        bsz = len(features)
        out = {k: v.view(bsz, n_choices, -1) for k, v in padded.items()}
        out["labels"] = labels
        return out
