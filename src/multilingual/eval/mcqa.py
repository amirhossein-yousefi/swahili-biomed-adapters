"""MCQA evaluation: accuracy + ECE + Brier, stratified by subject (§7.1, §7.4).

Per-option scoring: encode (question, option_i) as one input per option, take the
multiple-choice head's logits over options, softmax → predict argmax.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from .calibration import brier_multiclass, expected_calibration_error

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_mcqa(model, tokenizer, dataset: Dataset, *, batch_size: int = 8, max_length: int = 256,
                  device: str | None = None, num_choices: int = 4) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    preds: list[int] = []
    probs_all: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[str] = []

    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=list):
        first, second = [], []
        for ex in batch:
            opts = list(ex["options"])
            if len(opts) < num_choices:
                opts += [""] * (num_choices - len(opts))
            opts = opts[:num_choices]
            for opt in opts:
                first.append(ex["question"])
                second.append(opt)
        enc = tokenizer(first, second, padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt")
        bsz = len(batch)
        enc = {k: v.view(bsz, num_choices, -1).to(device) for k, v in enc.items()}
        out = model(**enc)
        logits = out.logits                                # (B, num_choices)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        pred = probs.argmax(axis=-1).tolist()
        preds.extend(pred)
        probs_all.extend(list(probs))
        labels.extend(int(ex["answer"]) for ex in batch)
        subjects.extend(str(ex.get("subject", "all")) for ex in batch)

    preds_a = np.asarray(preds)
    labels_a = np.asarray(labels)
    probs_a = np.asarray(probs_all)
    acc = float((preds_a == labels_a).mean())
    ece = expected_calibration_error(probs_a, labels_a)
    brier = brier_multiclass(probs_a, labels_a)

    by_subject: dict[str, dict] = {}
    for s in sorted(set(subjects)):
        mask = np.asarray([sub == s for sub in subjects])
        if not mask.any():
            continue
        by_subject[s] = {
            "n": int(mask.sum()),
            "accuracy": float((preds_a[mask] == labels_a[mask]).mean()),
            "ece": expected_calibration_error(probs_a[mask], labels_a[mask]),
            "brier": brier_multiclass(probs_a[mask], labels_a[mask]),
        }

    return {
        "n": int(len(labels)),
        "accuracy": acc,
        "ece": ece,
        "brier": brier,
        "by_subject": by_subject,
        "_per_example": {
            "predictions": preds, "labels": labels, "subjects": subjects,
            "probs": probs_a.tolist(),
        },
    }
