"""Smoke: MLM collator masks ~15% and produces correct label tensor shape."""
from __future__ import annotations


def test_collator_masks_about_15_percent():
    from transformers import AutoTokenizer

    from multilingual.data.mlm_collator import make_mlm_collator

    tok = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
    coll = make_mlm_collator(tok, mlm_prob=0.15, span=False)

    encs = [tok("Tunaweza kupata maambukizi ya virusi vya korona katika msimu wa baridi.",
                truncation=True, max_length=64) for _ in range(8)]
    batch = coll(encs)
    assert "input_ids" in batch and "labels" in batch
    n_total = (batch["attention_mask"] == 1).sum().item()
    n_mask = (batch["labels"] != -100).sum().item()
    frac = n_mask / max(1, n_total)
    assert 0.05 < frac < 0.30, f"expected ~15% masking, got {frac:.3f}"
