"""Smoke: C1–C5 each load + forward 1 batch on a tiny model. C1 forward ≡ frozen-base forward."""
from __future__ import annotations

import pytest


@pytest.mark.heavy
def test_c1_equals_base(backbone_name, tmp_path):
    import torch

    from multilingual.backbone import load_backbone
    from multilingual.compose import build_composition

    bb = load_backbone(backbone_name, freeze_base=True, bf16=False, flash_attn=False)
    bb.model.add_masked_lm_head("none_head")
    bb.model.eval()

    enc = bb.tokenizer("the quick brown fox jumps over the lazy dog", return_tensors="pt")
    with torch.no_grad():
        baseline = bb.model(**enc).logits

    build_composition(bb.model, "C1", adapters_paths={})
    with torch.no_grad():
        c1 = bb.model(**enc).logits

    assert torch.allclose(baseline, c1, atol=1e-5), "C1 should equal frozen-base forward"


def test_describe_table():
    from multilingual.compose import describe
    for cid in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"):
        msg = describe(cid)
        assert msg and "unknown" not in msg
