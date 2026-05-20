"""Single dispatcher: download / clean / cache each data source.

  python scripts/prepare_data.py source=swh        # Swahili raw + dedup + LID
  python scripts/prepare_data.py source=bio        # PubMed/PMC abstracts cache
  python scripts/prepare_data.py source=openwho    # OpenWHO held-out probe
  python scripts/prepare_data.py source=medlineplus
  python scripts/prepare_data.py source=mmlu_prox_swh
  python scripts/prepare_data.py source=bc5cdr     # English splits + Swahili test
  python scripts/prepare_data.py source=medmcqa
  python scripts/prepare_data.py source=masakhaner
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from multilingual.data import (
    bc5cdr,
    corpora_bio,
    corpora_en,
    corpora_swh,
    masakhaner,
    medlineplus,
    medmcqa,
    mmlu_prox_swh,
    openwho,
)
from multilingual.utils.logging import setup_logging

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    src = cfg.get("source", "swh")
    out_root = Path(cfg.get("data_dir", "data/processed"))
    out_root.mkdir(parents=True, exist_ok=True)
    out = out_root / f"{src}_manifest.json"

    if src == "swh":
        from multilingual.data.dedup import dedup
        from multilingual.data.lid import filter_lines
        n = 0
        path = out_root / "swh_raw.txt"
        threshold = float(cfg.get("lid_threshold", 0.5))
        max_docs = int(cfg.get("max_docs_per_source", 200_000))
        with open(path, "w") as f:
            stream = (r["text"] for r in corpora_swh.stream_all_swh(max_docs_per_source=max_docs))
            for line in filter_lines(dedup(stream), lang="sw", threshold=threshold):
                f.write(line + "\n")
                n += 1
        out.write_text(json.dumps({"path": str(path), "n": n,
                                   "lid_threshold": threshold,
                                   "max_docs_per_source": max_docs}))
    elif src == "en":
        # English raw text for LA_en (MAD-X source-language adapter). Pipeline
        # is identical to swh but using English Wikipedia, threshold 0.7
        # (FastText lid.176 is well-calibrated on English). We also peel off
        # the first `eval_size` sentences as a held-out MLM-perplexity probe
        # so train_la.py has something to eval against (no English OpenWHO).
        from multilingual.data.dedup import dedup
        from multilingual.data.lid import filter_lines
        path = out_root / "en_raw.txt"
        eval_path = out_root / "openwho_en_eval.txt"
        threshold = float(cfg.get("lid_threshold", 0.7))
        max_docs = int(cfg.get("max_docs_per_source", 100_000))
        eval_size = int(cfg.get("eval_size", 2000))
        n_train = 0
        n_eval = 0
        with open(path, "w") as f_train, open(eval_path, "w") as f_eval:
            stream = (r["text"] for r in corpora_en.stream_all_en(max_docs_per_source=max_docs))
            for line in filter_lines(dedup(stream), lang="en", threshold=threshold):
                if n_eval < eval_size:
                    f_eval.write(line + "\n")
                    n_eval += 1
                else:
                    f_train.write(line + "\n")
                    n_train += 1
        out.write_text(json.dumps({"path": str(path), "n": n_train,
                                   "eval_path": str(eval_path), "n_eval": n_eval,
                                   "lid_threshold": threshold,
                                   "max_docs_per_source": max_docs}))
    elif src == "bio":
        n = 0
        path = out_root / "bio_raw.txt"
        with open(path, "w") as f:
            for r in corpora_bio.stream_all_bio(
                max_abstracts=cfg.get("max_abstracts", 1_000_000), max_pmc=0):
                t = r["text"].strip().replace("\n", " ")
                if t:
                    f.write(t + "\n")
                    n += 1
        out.write_text(json.dumps({"path": str(path), "n": n}))
    elif src == "openwho":
        path = out_root / "openwho_swh_eval.txt"
        openwho.held_out_eval_split(path, n_sentences=cfg.get("n_sentences", 2000))
        out.write_text(json.dumps({"path": str(path)}))
    elif src == "medlineplus":
        path = out_root / "medlineplus_swh.jsonl"
        medlineplus.write_topic_classification_jsonl(path)
        out.write_text(json.dumps({"path": str(path)}))
    elif src == "mmlu_prox_swh":
        ds = mmlu_prox_swh.load_mmlu_prox_swahili_clinical("test")
        path = out_root / "mmlu_prox_swh_test"
        ds.save_to_disk(str(path))
        out.write_text(json.dumps({"path": str(path), "n": len(ds)}))
    elif src == "bc5cdr":
        en = bc5cdr.load_bc5cdr_en()
        en_path = out_root / "bc5cdr_en"
        en.save_to_disk(str(en_path))
        swh_path = out_root / "bc5cdr_swh_test.jsonl"
        if not swh_path.exists():
            bc5cdr.write_swahili_test_jsonl(swh_path, en_split="test")
        out.write_text(json.dumps({"en_path": str(en_path), "swh_test": str(swh_path)}))
    elif src == "medmcqa":
        ds = medmcqa.load_medmcqa("train")
        path = out_root / "medmcqa_train"
        ds.save_to_disk(str(path))
        out.write_text(json.dumps({"path": str(path), "n": len(ds)}))
    elif src == "masakhaner":
        dsd = masakhaner.load_masakhaner_swh()
        path = out_root / "masakhaner_swh"
        dsd.save_to_disk(str(path))
        out.write_text(json.dumps({"path": str(path),
                                   "splits": {k: len(v) for k, v in dsd.items()}}))
    else:
        raise ValueError(f"unknown source: {src}")
    log.info("prepared %s; manifest at %s", src, out)


if __name__ == "__main__":
    main()
