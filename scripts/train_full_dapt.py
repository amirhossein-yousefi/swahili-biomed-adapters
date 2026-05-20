"""Baseline (d): full DAPT (continued pretraining) on Swa-Med.

Most expensive baseline (~2 days wall-clock). Builds a Swahili biomedical
corpus from {OpenWHO Swahili} ∪ {MedlinePlus Swahili scrape} ∪
{NLLB-translated PubMed abstracts} and full-fine-tunes the backbone with MLM.
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer

from multilingual.data import medlineplus, openwho
from multilingual.data.corpora_bio import stream_pubmed_abstracts
from multilingual.training.full_ft_trainer import build_full_dapt_run
from multilingual.translate.nllb import translate_iter
from multilingual.utils.io import write_run_artifacts
from multilingual.utils.logging import maybe_init_wandb, setup_logging
from multilingual.utils.seed import set_seed

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    set_seed(cfg.run.seed)
    wandb_run = maybe_init_wandb(cfg)

    # 1. OpenWHO Swahili sentences
    openwho_iter = openwho.iter_swahili_sentences(split="train",
                                                  limit=cfg.data.get("openwho_max"))

    # 2. MedlinePlus Swahili scrape (text only)
    def medlineplus_iter():
        cache = Path(cfg.data.get("medlineplus_cache", "data/raw/medlineplus"))
        for rec in medlineplus.crawl(cache):
            yield rec["text"]

    # 3. NLLB-translated PubMed abstracts (capped)
    n_pub = int(cfg.data.get("translated_pubmed", 50_000))
    en_iter = (r["text"] for r in stream_pubmed_abstracts(max_docs=n_pub))
    swh_pub = translate_iter(en_iter, src_lang="eng_Latn", tgt_lang="swh_Latn",
                             batch_size=cfg.data.get("nllb_batch", 16))

    def all_text():
        for s in openwho_iter:
            yield {"text": s}
        for s in medlineplus_iter():
            yield {"text": s}
        for s in swh_pub:
            yield {"text": s}

    iter_ds = Dataset.from_generator(all_text)
    tok = AutoTokenizer.from_pretrained(cfg.backbone.name, use_fast=True)
    seq_len = int(cfg.train.get("seq_len", 512))
    train_ds = iter_ds.map(lambda b: tok(b["text"], truncation=True, max_length=seq_len),
                           batched=True, remove_columns=["text"])

    out_dir = Path(cfg.run.output_dir) / "baseline_d_full_dapt"
    run = build_full_dapt_run(
        backbone_name=cfg.backbone.name, train_dataset=train_ds,
        output_dir=out_dir,
        num_train_epochs=cfg.train.get("num_epochs", 3.0),
        per_device_batch_size=cfg.train.get("batch_size", 16),
        grad_accum=cfg.train.get("grad_accum", 4),
        learning_rate=cfg.train.get("lr", 5e-5),
        warmup_steps=cfg.train.get("warmup_steps", 1000),
        bf16=cfg.run.bf16, seed=cfg.run.seed, wandb_run=wandb_run,
    )
    run.trainer.train()
    run.trainer.save_model(str(out_dir / "model"))
    run.tokenizer.save_pretrained(str(out_dir / "model"))
    write_run_artifacts(out_dir, cfg, metrics={})


if __name__ == "__main__":
    main()
