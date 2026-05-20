"""§5.2 — Train the Swahili language adapter (LA) via MLM on Swahili raw text.

  python scripts/train_la.py adapter=pfeiffer_rf16_inv  \
                              data=la_swh train=la_mlm
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
from datasets import IterableDataset
from omegaconf import DictConfig, OmegaConf

from multilingual.adapter_setup import build_adapter_config
from multilingual.data import corpora_swh, openwho
from multilingual.data.dedup import dedup
from multilingual.data.lid import filter_lines
from multilingual.training.mlm_trainer import build_mlm_run, save_adapter_only
from multilingual.utils.io import write_run_artifacts
from multilingual.utils.logging import maybe_init_wandb, setup_logging
from multilingual.utils.seed import set_seed

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    set_seed(cfg.run.seed)
    wandb_run = maybe_init_wandb(cfg)

    # ---- data: load the prepared Swahili sentence file ----
    train_path = Path(cfg.data.get("train_data_path", "data/swh_raw.txt"))
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found — run `make data-swh` (or "
            f"`python scripts/prepare_data.py +source=swh`) first.")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.backbone.name, use_fast=True)
    seq_len = int(cfg.train.get("seq_len", 512))

    raw_ds = load_dataset("text", data_files={"train": str(train_path)}, split="train")
    log.info("loaded %d Swahili sentences from %s", len(raw_ds), train_path)
    train_ds = raw_ds.map(lambda b: tok(b["text"], truncation=True, max_length=seq_len),
                          batched=True, remove_columns=["text"])

    # held-out OpenWHO Swahili eval split for dev MLM ppl
    eval_ds = None
    eval_path = Path(cfg.data.get("eval_path", "data/openwho_swh_eval.txt"))
    try:
        if not eval_path.exists():
            openwho.held_out_eval_split(eval_path, n_sentences=cfg.data.get("eval_size", 2000))
        eval_lines = [l for l in eval_path.read_text().splitlines() if l.strip()]
        from datasets import Dataset
        eval_ds = Dataset.from_dict({"text": eval_lines}).map(
            lambda b: tok(b["text"], truncation=True, max_length=seq_len),
            batched=True, remove_columns=["text"])
        log.info("OpenWHO eval: %d sentences", len(eval_ds))
    except Exception as e:
        log.warning("OpenWHO held-out eval split unavailable (%s); skipping eval", e)

    # ---- adapter config (Pfeiffer + invertible NICE on embeddings for LA) ----
    adapter_cfg = build_adapter_config(cfg.adapter)

    # The adapter we're training (LA_swh by default; override for LA_en, etc.)
    adapter_name = cfg.train.get("adapter_name", "LA_swh")

    # STABLE training dir (resume-safe across crashes/reboots); artifact dir
    # stays under the Hydra run dir for per-launch config/metrics snapshots.
    stable_dir = Path(cfg.train.get(
        "stable_output_dir",
        f"{cfg.get('ckpt_dir', './checkpoints')}/{adapter_name.lower()}_run"))
    stable_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(cfg.run.output_dir) / adapter_name.lower()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    run = build_mlm_run(
        backbone_name=cfg.backbone.name, adapter_name=adapter_name, adapter_cfg=adapter_cfg,
        train_dataset=train_ds, eval_dataset=eval_ds,
        output_dir=stable_dir,
        num_train_epochs=cfg.train.get("num_epochs", 3.0),
        max_steps=cfg.train.get("max_steps", -1),
        per_device_batch_size=cfg.train.get("batch_size", 16),
        grad_accum=cfg.train.get("grad_accum", 4),
        learning_rate=cfg.train.get("lr", 1e-4),
        warmup_steps=cfg.train.get("warmup_steps", 2000),
        weight_decay=cfg.train.get("weight_decay", 0.01),
        max_grad_norm=cfg.train.get("max_grad_norm", 1.0),
        bf16=cfg.run.bf16, flash_attn=cfg.run.flash_attn,
        span_mask=cfg.train.get("span_mask", False),
        save_steps=cfg.train.get("save_steps", 1_000),
        eval_steps=cfg.train.get("eval_steps", 1_000),
        seed=cfg.run.seed, wandb_run=wandb_run,
        resume_from_checkpoint=cfg.train.get("resume", True),
    )

    # Resume only if at least one checkpoint exists in the stable dir; HF
    # raises if you pass resume=True with an empty output_dir.
    ckpts = sorted([p for p in stable_dir.iterdir() if p.name.startswith("checkpoint-")],
                   key=lambda p: int(p.name.split("-")[1]))
    resume_arg = None
    if ckpts and cfg.train.get("resume", True):
        latest = ckpts[-1]
        # The `adapters` library re-creates parameter objects on every
        # load_adapter call, so HF's optimizer-state restore raises
        # "loaded state dict contains a parameter group that doesn't match
        # the size of optimizer's group". Strip optimizer.pt / scheduler.pt
        # before resume so HF skips that step (Adam momentum is lost but
        # warms back in a few hundred steps; model weights + step counter
        # are preserved via trainer_state.json).
        for stale in ("optimizer.pt", "scheduler.pt"):
            stale_path = latest / stale
            if stale_path.exists():
                stale_path.unlink()
                log.info("stripped %s from %s for resume", stale, latest.name)
        log.info("resuming from %s", latest)
        resume_arg = str(latest)

    run.trainer.train(resume_from_checkpoint=resume_arg)
    save_adapter_only(run, adapter_name)
    metrics = run.trainer.evaluate() if eval_ds is not None else {}
    write_run_artifacts(artifact_dir, cfg, metrics=metrics)


if __name__ == "__main__":
    main()
