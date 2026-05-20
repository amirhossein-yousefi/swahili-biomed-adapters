"""Baselines (e) translate-train and (f) translate-test, plus benchmark construction.

  python scripts/translate_corpus.py mode=train     # baseline (e)
  python scripts/translate_corpus.py mode=test      # baseline (f)
  python scripts/translate_corpus.py mode=benchmark # generate Swa-MedBench synthetic data

Caches every translation to data/translated/ via translate.nllb so reruns are cheap.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from multilingual.data import bc5cdr, medmcqa
from multilingual.translate.nllb import translate_iter
from multilingual.utils.io import write_run_artifacts
from multilingual.utils.logging import setup_logging

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    mode = str(cfg.get("mode", "train"))
    out_dir = Path(cfg.run.output_dir) / f"translate_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "train":
        # Translate English MedMCQA training set → Swahili; produces a JSONL the
        # task-adapter trainer can ingest.
        ds = medmcqa.load_medmcqa("train")
        path = out_dir / "medmcqa_swh_train.jsonl"
        _translate_mcq_dataset(ds, path)
    elif mode == "test":
        # Translate-test pipeline = translate Swahili eval inputs → English at
        # eval time. This script only emits the cached translations; the eval
        # harness (eval_mcqa with cfg.translate_test=true) consumes them.
        from multilingual.data import mmlu_prox_swh
        ds = mmlu_prox_swh.load_mmlu_prox_swahili_clinical("test")
        path = out_dir / "mmlu_prox_swh_test_to_en.jsonl"
        _translate_mcq_dataset(ds, path, src_lang="swh_Latn", tgt_lang="eng_Latn")
    elif mode == "benchmark":
        # Build BC5CDR-Disease Swahili test set (NLLB + awesome-align).
        path = out_dir / "bc5cdr_disease_swh_test.jsonl"
        bc5cdr.write_swahili_test_jsonl(path, en_split="test")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    write_run_artifacts(out_dir, cfg, metrics={"output": str(path)})


def _translate_mcq_dataset(ds, out_path: Path, *, src_lang: str = "eng_Latn",
                           tgt_lang: str = "swh_Latn") -> None:
    """Translate question + each option, preserving answer index."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    flat: list[str] = []
    shape: list[int] = []  # number of options per question
    for r in ds:
        flat.append(r["question"])
        flat.extend(r["options"])
        shape.append(len(r["options"]))

    translated = list(translate_iter(flat, src_lang=src_lang, tgt_lang=tgt_lang, batch_size=16))

    cursor = 0
    with open(out_path, "w") as f:
        for r, n_opts in zip(ds, shape):
            q = translated[cursor]; cursor += 1
            opts = translated[cursor:cursor + n_opts]; cursor += n_opts
            f.write(json.dumps({"question": q, "options": opts, "answer": int(r["answer"]),
                                "subject": r.get("subject", ""), "lang": tgt_lang.split("_")[0]}) + "\n")
    log.info("Wrote %d translated MCQ examples to %s", len(shape), out_path)


if __name__ == "__main__":
    main()
