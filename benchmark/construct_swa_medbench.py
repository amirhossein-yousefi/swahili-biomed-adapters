"""Orchestrate construction of Swa-MedBench v0.1.

  python benchmark/construct_swa_medbench.py output_dir=./benchmark/v0_1
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from multilingual.data import bc5cdr, medlineplus, medmcqa, mmlu_prox_swh, openwho
from multilingual.translate.nllb import translate_iter
from multilingual.utils.logging import setup_logging

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    out = Path(cfg.get("output_dir", "./benchmark/v0_1"))
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"name": "Swa-MedBench", "version": "0.1", "components": {}}

    # 1. MMLU-ProX-Swahili clinical
    mmlu = mmlu_prox_swh.load_mmlu_prox_swahili_clinical("test")
    mmlu_path = out / "mmlu_prox_swh_clinical_test"
    mmlu.save_to_disk(str(mmlu_path))
    manifest["components"]["mmlu_prox_swh_clinical"] = {"path": str(mmlu_path), "n": len(mmlu)}

    # 2. MedMCQA → Swahili (sample, not the entire 190K)
    medmcqa_en = medmcqa.load_medmcqa("validation")
    medmcqa_swh_path = out / "medmcqa_swh_dev.jsonl"
    _translate_mcq(medmcqa_en, medmcqa_swh_path, src="eng_Latn", tgt="swh_Latn",
                   limit=cfg.get("medmcqa_swh_limit", 2000))
    manifest["components"]["medmcqa_swh_dev"] = {"path": str(medmcqa_swh_path)}

    # 3. BC5CDR-Disease Swahili test (translate + project spans)
    bc_path = out / "bc5cdr_disease_swh_test.jsonl"
    if not bc_path.exists():
        bc5cdr.write_swahili_test_jsonl(bc_path, en_split="test")
    manifest["components"]["bc5cdr_disease_swh_test"] = {"path": str(bc_path)}

    # 4. OpenWHO Swahili held-out
    openwho_path = out / "openwho_swh_eval.txt"
    openwho.held_out_eval_split(openwho_path, n_sentences=cfg.get("openwho_n", 2000))
    manifest["components"]["openwho_swh_eval"] = {"path": str(openwho_path)}

    # 5. MedlinePlus Swahili topic classification
    medline_path = out / "medlineplus_swh_topic.jsonl"
    if not medline_path.exists():
        medlineplus.write_topic_classification_jsonl(medline_path)
    manifest["components"]["medlineplus_swh_topic"] = {"path": str(medline_path)}

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    log.info("Swa-MedBench v0.1 written to %s", out)


def _translate_mcq(ds, out_path: Path, *, src: str, tgt: str, limit: int):
    flat: list[str] = []
    shape: list[int] = []
    items = list(ds)[:limit]
    for r in items:
        flat.append(r["question"])
        flat.extend(r["options"])
        shape.append(len(r["options"]))
    translated = list(translate_iter(flat, src_lang=src, tgt_lang=tgt))
    cur = 0
    with open(out_path, "w") as f:
        for r, n_opts in zip(items, shape):
            q = translated[cur]; cur += 1
            opts = translated[cur:cur + n_opts]; cur += n_opts
            f.write(json.dumps({"question": q, "options": opts, "answer": int(r["answer"]),
                                "subject": r.get("subject", ""), "lang": tgt.split("_")[0]}) + "\n")
    log.info("Wrote %d translated MCQ examples to %s", len(items), out_path)


if __name__ == "__main__":
    main()
