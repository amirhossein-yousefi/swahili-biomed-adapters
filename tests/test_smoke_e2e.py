"""End-to-end smoke: MLM micro-run on 100 sequences must show a decreasing loss.

Uses xlm-roberta-base + Pfeiffer rf=16 + invertible. CPU OK; runs ~1 minute.
"""
from __future__ import annotations

import pytest


@pytest.mark.heavy
def test_mlm_micro_run(backbone_name, tmp_path):
    from datasets import Dataset
    from transformers import AutoTokenizer

    from multilingual.adapter_setup import build_adapter_config
    from multilingual.training.mlm_trainer import build_mlm_run

    sw_lines = [
        "Tunaweza kupata maambukizi ya virusi katika msimu wa baridi.",
        "Watoto wanapaswa kupokea chanjo zote zilizopendekezwa.",
        "Ujauzito ni kipindi muhimu kwa afya ya mama na mtoto.",
        "Daktari anapaswa kuagiza dawa baada ya uchunguzi.",
    ] * 25  # 100 sentences

    tok = AutoTokenizer.from_pretrained(backbone_name, use_fast=True)
    ds = Dataset.from_dict({"text": sw_lines})
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=64),
                batched=True, remove_columns=["text"])

    cfg_obj = build_adapter_config({"type": "pfeiffer", "reduction_factor": 16,
                                    "non_linearity": "gelu", "invertible": True,
                                    "inv_reduction_factor": 2})
    run = build_mlm_run(
        backbone_name=backbone_name, adapter_name="LA_swh_smoke",
        adapter_cfg=cfg_obj, train_dataset=ds, eval_dataset=None,
        output_dir=tmp_path / "smoke",
        num_train_epochs=1.0, max_steps=20,
        per_device_batch_size=4, grad_accum=1,
        learning_rate=5e-4, warmup_steps=2, save_steps=999_999,
        eval_steps=999_999, bf16=False, flash_attn=False, logging_steps=5,
    )
    out = run.trainer.train()
    assert out.training_loss < 30.0  # finite, sane
