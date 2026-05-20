"""Paired bootstrap (+ ASO) on two saved run-result JSONs.

  python scripts/run_significance.py \
      --a results/runs/...../predictions.jsonl \
      --b results/runs/...../predictions.jsonl \
      --metric accuracy
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from multilingual.eval.significance import almost_stochastic_order, paired_bootstrap
from multilingual.utils.logging import setup_logging

log = logging.getLogger(__name__)


def _scores(path: Path, metric: str) -> np.ndarray:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if metric == "accuracy":
        return np.asarray([1.0 if r["pred"] == r["label"] else 0.0 for r in rows])
    if metric == "f1_per_sentence":
        return np.asarray([float(r["f1"]) for r in rows])
    raise ValueError(f"unknown metric: {metric}")


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, type=Path)
    p.add_argument("--b", required=True, type=Path)
    p.add_argument("--metric", default="accuracy", choices=["accuracy", "f1_per_sentence"])
    p.add_argument("--n-resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    a = _scores(args.a, args.metric)
    b = _scores(args.b, args.metric)
    if len(a) != len(b):
        raise SystemExit(f"length mismatch: {len(a)} vs {len(b)} — examples must align")

    pb = paired_bootstrap(a, b, n_resamples=args.n_resamples, seed=args.seed)
    aso = almost_stochastic_order(a, b)
    log.info("paired_bootstrap: %s", pb)
    log.info("ASO:              %s", aso)
    print(json.dumps({"paired_bootstrap": pb, "aso": aso}, indent=2))


if __name__ == "__main__":
    main()
