"""NER evaluation: span-level F1 (CoNLL eval), strict + relaxed, per-entity-type.

`seqeval` provides the strict CoNLL match. Relaxed match accepts overlapping
spans of the same type as a hit.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_ner(model, tokenizer, dataset: Dataset, id2label: dict[int, str], *,
                 batch_size: int = 16, max_length: int = 256, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    pred_labels: list[list[str]] = []
    true_labels: list[list[str]] = []

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        enc = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True,
                        max_length=max_length, padding=True, return_tensors="pt")
        word_ids_batch = [tokenizer(toks, is_split_into_words=True, truncation=True,
                                    max_length=max_length).word_ids() for toks in batch["tokens"]]
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.cpu().numpy()        # (B, L, K)
        preds = logits.argmax(axis=-1)
        for b in range(len(batch["tokens"])):
            wid = word_ids_batch[b]
            tags_true = [id2label[int(t)] for t in batch["tags"][b]]
            tags_pred: list[str] = []
            seen_words: set[int] = set()
            for tok_idx, word_idx in enumerate(wid):
                if word_idx is None or word_idx in seen_words:
                    continue
                seen_words.add(word_idx)
                tags_pred.append(id2label[int(preds[b, tok_idx])])
            # align lengths if tokenizer dropped trailing words
            L = min(len(tags_pred), len(tags_true))
            pred_labels.append(tags_pred[:L])
            true_labels.append(tags_true[:L])

    strict_f1 = f1_score(true_labels, pred_labels, mode="strict")
    relaxed_f1 = f1_score(true_labels, pred_labels)  # default = relaxed
    report = classification_report(true_labels, pred_labels, output_dict=True)
    per_sentence_f1 = _per_sentence_strict_f1(true_labels, pred_labels)

    return {
        "n": int(len(true_labels)),
        "f1_strict": float(strict_f1),
        "f1_relaxed": float(relaxed_f1),
        "per_type": _entities_only(report),
        "_per_example": {
            "f1_per_sentence": per_sentence_f1,
            "predictions": pred_labels, "labels": true_labels,
        },
    }


def _entities_only(report: dict) -> dict:
    return {k: {"precision": v["precision"], "recall": v["recall"], "f1": v["f1-score"],
                "support": v["support"]}
            for k, v in report.items()
            if isinstance(v, dict) and k not in ("micro avg", "macro avg", "weighted avg")}


def _per_sentence_strict_f1(true_labels, pred_labels) -> list[float]:
    """Span-F1 computed sentence-by-sentence, used by paired bootstrap."""
    out = []
    for t, p in zip(true_labels, pred_labels):
        try:
            out.append(float(f1_score([t], [p], mode="strict")))
        except Exception:
            out.append(0.0)
    return out
