"""§5.4 — Train task adapters on top of frozen [LA → DA] stack.

  python scripts/train_ta.py train=ta_mcqa
  python scripts/train_ta.py train=ta_ner
  python scripts/train_ta.py train=ta_topic

Per cfg.run.seed: defaults to seeds=[42, 7, 123] when run as a multirun:
  python scripts/train_ta.py -m run.seed=42,7,123 train=ta_mcqa
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from multilingual.adapter_setup import build_adapter_config
from multilingual.data import afrimed_qa, bc5cdr, masakhaner, medlineplus, medmcqa, mmlu_prox_swh
from multilingual.training.task_trainer import build_task_run, save_ta_only
from multilingual.utils.io import write_run_artifacts
from multilingual.utils.logging import maybe_init_wandb, setup_logging
from multilingual.utils.seed import set_seed

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    set_seed(cfg.run.seed)
    wandb_run = maybe_init_wandb(cfg)

    task = str(cfg.train.task_type)         # 'mcqa' | 'ner' | 'classification'
    train_ds, eval_ds, num_labels, num_choices, id2label = _load_task_data(task, cfg)

    adapter_cfg = build_adapter_config(cfg.adapter)
    out_dir = Path(cfg.run.output_dir) / f"ta_{task}" / f"seed{cfg.run.seed}"
    run = build_task_run(
        backbone_name=cfg.backbone.name, ta_name="TA",
        adapter_cfg=adapter_cfg, task_type=task,
        train_dataset=train_ds, eval_dataset=eval_ds,
        la_path=cfg.train.get("la_path"), da_path=cfg.train.get("da_path"),
        num_labels=num_labels, num_choices=num_choices, id2label=id2label,
        output_dir=out_dir,
        num_train_epochs=cfg.train.get("num_epochs", 5.0),
        per_device_batch_size=cfg.train.get("batch_size", 16),
        grad_accum=cfg.train.get("grad_accum", 1),
        learning_rate=cfg.train.get("lr", 1e-4),
        bf16=cfg.run.bf16, flash_attn=cfg.run.flash_attn,
        seed=cfg.run.seed, wandb_run=wandb_run,
        max_length=cfg.train.get("max_length", 256),
    )
    run.trainer.train()
    save_ta_only(run, "TA")
    metrics = run.trainer.evaluate() if eval_ds is not None else {}
    write_run_artifacts(out_dir, cfg, metrics=metrics)
    log.info("TA(%s) saved to %s", task, out_dir)


def _load_task_data(task: str, cfg: DictConfig):
    if task == "mcqa":
        train = medmcqa.load_medmcqa("train")
        eval_ = mmlu_prox_swh.load_mmlu_prox_swahili_clinical("test") if cfg.train.get("eval_swh") \
                else medmcqa.load_medmcqa("validation")
        return train, eval_, None, int(cfg.train.get("num_choices", 4)), None
    if task == "ner":
        en = bc5cdr.load_bc5cdr_en()
        return en["train"], en["validation"], len(bc5cdr.LABELS), None, bc5cdr.ID2LABEL
    if task == "classification":
        # MedlinePlus topic classification (English source for training);
        # eval set is the Swahili MedlinePlus subset
        path = Path(cfg.train.get("data_path", "data/processed/medlineplus_swh.jsonl"))
        if not path.exists():
            medlineplus.write_topic_classification_jsonl(path)
        from datasets import Dataset
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        ds = Dataset.from_list(rows).train_test_split(test_size=0.1, seed=42)
        n_labels = int(max(r["label"] for r in rows)) + 1
        return ds["train"], ds["test"], n_labels, None, {i: c for i, c in enumerate(
            medlineplus.CATEGORIES)}
    raise ValueError(f"Unknown task_type: {task}")


if __name__ == "__main__":
    main()
