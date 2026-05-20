"""Baseline (g): single LoRA module fine-tuned on
union(Swahili raw, English biomedical raw, supervised task data).

Uses the `adapters` library's LoRAConfig so the comparison to the bottleneck
adapter pipeline is apples-to-apples (same backbone, same library).
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
from datasets import Dataset, concatenate_datasets
from omegaconf import DictConfig
from transformers import AutoTokenizer

from multilingual.adapter_setup import build_adapter_config
from multilingual.data import bc5cdr, corpora_bio, corpora_swh, medmcqa
from multilingual.training.mlm_trainer import build_mlm_run, save_adapter_only
from multilingual.utils.io import write_run_artifacts
from multilingual.utils.logging import maybe_init_wandb, setup_logging
from multilingual.utils.seed import set_seed

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Joint pretraining stage: MLM on (swh ∪ en-bio). The supervised task adapter
    is then trained separately on top via scripts/train_ta.py with the joint LoRA
    pre-loaded.
    """
    setup_logging()
    set_seed(cfg.run.seed)
    wandb_run = maybe_init_wandb(cfg)

    swh = (r["text"] for r in corpora_swh.stream_all_swh(
        max_docs_per_source=cfg.data.get("swh_max_per_source")))
    bio = (r["text"] for r in corpora_bio.stream_all_bio(
        max_abstracts=cfg.data.get("bio_max_abstracts"), max_pmc=0))

    def mixed():
        for s in swh:
            yield {"text": s, "src": "swh"}
        for s in bio:
            yield {"text": s, "src": "en_bio"}

    iter_ds = Dataset.from_generator(mixed)
    tok = AutoTokenizer.from_pretrained(cfg.backbone.name, use_fast=True)
    seq_len = int(cfg.train.get("seq_len", 512))
    train_ds = iter_ds.map(lambda b: tok(b["text"], truncation=True, max_length=seq_len),
                           batched=True, remove_columns=["text", "src"])

    adapter_cfg = build_adapter_config({"type": "lora",
                                        "r": cfg.adapter.get("r", 16),
                                        "alpha": cfg.adapter.get("alpha", 16)})
    out_dir = Path(cfg.run.output_dir) / "baseline_g_joint_lora"
    run = build_mlm_run(
        backbone_name=cfg.backbone.name, adapter_name="joint_lora",
        adapter_cfg=adapter_cfg, train_dataset=train_ds,
        output_dir=out_dir,
        num_train_epochs=cfg.train.get("num_epochs", 2.0),
        per_device_batch_size=cfg.train.get("batch_size", 16),
        grad_accum=cfg.train.get("grad_accum", 4),
        learning_rate=cfg.train.get("lr", 1e-4),
        bf16=cfg.run.bf16, flash_attn=cfg.run.flash_attn,
        seed=cfg.run.seed, wandb_run=wandb_run,
    )
    run.trainer.train()
    save_adapter_only(run, "joint_lora")
    write_run_artifacts(out_dir, cfg, metrics={})


if __name__ == "__main__":
    main()
