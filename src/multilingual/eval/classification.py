"""Sequence classification eval: macro-F1 + weighted-F1 (§7.1)."""
from __future__ import annotations

import logging

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_classification(model, tokenizer, dataset: Dataset, id2label: dict[int, str] | None,
                            *, batch_size: int = 16, max_length: int = 256,
                            device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    preds: list[int] = []
    labels: list[int] = []
    for i in range(0, len(dataset), batch_size):
        b = dataset[i:i + batch_size]
        enc = tokenizer(b["text"], padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt").to(device)
        logits = model(**enc).logits.cpu().numpy()
        preds.extend(logits.argmax(axis=-1).tolist())
        labels.extend(b["label"])

    p = np.asarray(preds)
    l = np.asarray(labels)
    macro = f1_score(l, p, average="macro", zero_division=0)
    weighted = f1_score(l, p, average="weighted", zero_division=0)
    return {
        "n": int(len(labels)),
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "report": classification_report(l, p, output_dict=True, zero_division=0,
                                        labels=list(id2label) if id2label else None,
                                        target_names=[id2label[i] for i in sorted(id2label)]
                                                      if id2label else None),
        "_per_example": {"predictions": preds, "labels": labels},
    }
