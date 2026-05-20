"""§5.3 — Train the English biomedical domain adapter (DA) via MLM on PubMed/PMC.

Two modes (both should be run as an ablation):
  train.mode=on_base : DA trained directly on the frozen base
  train.mode=on_la   : DA trained on top of frozen LA_en (cfg.la_en_path)

  python scripts/train_da.py train=da_mlm train.mode=on_base
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer

from multilingual.adapter_setup import build_adapter_config
from multilingual.data import corpora_bio
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

    # Load from the prepared bio raw-text file (output of
    # `prepare_data.py +source=bio`). Reading from disk via `load_dataset("text")`
    # avoids the dill-fingerprinting issue we hit with Dataset.from_generator on
    # complex closures (same fix landed in train_la.py).
    train_path = Path(cfg.data.get("train_data_path",
                                   f"{cfg.get('data_dir', './data')}/bio_raw.txt"))
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found — run `python scripts/prepare_data.py +source=bio` first.")
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(cfg.backbone.name, use_fast=True)
    seq_len = int(cfg.train.get("seq_len", 512))
    raw_ds = load_dataset("text", data_files={"train": str(train_path)}, split="train")
    log.info("loaded %d English biomedical abstracts/lines from %s", len(raw_ds), train_path)
    train_ds = raw_ds.map(lambda b: tok(b["text"], truncation=True, max_length=seq_len),
                          batched=True, remove_columns=["text"])

    base_stack: list[str] | None = None
    base_paths: dict[str, str] | None = None
    if cfg.train.get("mode", "on_base") == "on_la":
        la_en = cfg.train.get("la_en_path")
        if not la_en:
            raise ValueError("train.mode=on_la requires train.la_en_path")
        base_stack = ["LA_en"]
        base_paths = {"LA_en": la_en}

    adapter_cfg = build_adapter_config(cfg.adapter)

    # Stable training dir (resume-safe across crashes); artifact dir lives
    # under the Hydra run dir for per-launch config/metrics snapshots.
    mode = cfg.train.get("mode", "on_base")
    stable_dir = Path(cfg.train.get("stable_output_dir",
                                    f"{cfg.get('ckpt_dir', './checkpoints')}/da_eng_bio_{mode}_run"))
    stable_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(cfg.run.output_dir) / f"da_eng_bio_{mode}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    run = build_mlm_run(
        backbone_name=cfg.backbone.name, adapter_name="DA_eng", adapter_cfg=adapter_cfg,
        train_dataset=train_ds,
        base_stack=base_stack, base_stack_paths=base_paths,
        output_dir=stable_dir,
        num_train_epochs=cfg.train.get("num_epochs", 3.0),
        max_steps=cfg.train.get("max_steps", -1),
        per_device_batch_size=cfg.train.get("batch_size", 16),
        grad_accum=cfg.train.get("grad_accum", 4),
        learning_rate=cfg.train.get("lr", 1e-4),
        warmup_steps=cfg.train.get("warmup_steps", 2000),
        bf16=cfg.run.bf16, flash_attn=cfg.run.flash_attn,
        span_mask=cfg.train.get("span_mask", False),
        save_steps=cfg.train.get("save_steps", 1_000),
        eval_steps=cfg.train.get("eval_steps", 1_000),
        seed=cfg.run.seed, wandb_run=wandb_run,
        resume_from_checkpoint=cfg.train.get("resume", True),
    )
    ckpts = sorted([p for p in stable_dir.iterdir() if p.name.startswith("checkpoint-")],
                   key=lambda p: int(p.name.split("-")[1]))
    resume_arg = None
    if ckpts and cfg.train.get("resume", True):
        latest = ckpts[-1]
        # adapters lib re-creates parameter objects on load_adapter, breaking
        # HF's optimizer-state restore. Strip optimizer.pt/scheduler.pt so HF
        # skips that step. Adam momentum is lost (warms back in seconds);
        # model weights + step counter survive via trainer_state.json.
        for stale in ("optimizer.pt", "scheduler.pt"):
            stale_path = latest / stale
            if stale_path.exists():
                stale_path.unlink()
                log.info("stripped %s from %s for resume", stale, latest.name)
        log.info("resuming from %s", latest)
        resume_arg = str(latest)
    run.trainer.train(resume_from_checkpoint=resume_arg)
    save_adapter_only(run, "DA_eng")
    write_run_artifacts(artifact_dir, cfg, metrics={})


if __name__ == "__main__":
    main()
