"""Smoke: Pfeiffer + invertible attaches; param counts; requires_grad correct."""
from __future__ import annotations

import pytest


@pytest.mark.heavy
def test_pfeiffer_attach(backbone_name):
    from multilingual.adapter_setup import add_adapter, build_adapter_config
    from multilingual.backbone import (
        assert_only_adapters_trainable,
        load_backbone,
        trainable_param_summary,
    )

    bb = load_backbone(backbone_name, freeze_base=True, bf16=False, flash_attn=False)
    cfg = build_adapter_config({"type": "pfeiffer", "reduction_factor": 16,
                                "non_linearity": "gelu", "invertible": True,
                                "inv_reduction_factor": 2})
    add_adapter(bb.model, "LA_swh", cfg, train=True)
    bb.model.add_masked_lm_head("LA_swh")

    summary = trainable_param_summary(bb.model)
    # base XLM-R-base has ~280M params; adapter rf=16 adds ~3-7M
    assert summary["trainable"] > 1_000_000
    assert summary["trainable"] < 50_000_000

    assert_only_adapters_trainable(bb.model)
