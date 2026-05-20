"""Stratified sampler for clinician validation.

  python benchmark/clinician_validation/sampling.py \
      --input benchmark/v0_1/manifest.json --n 1000 --strata subject \
      --output benchmark/v0_1/clinician_sample.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from datasets import Dataset, load_from_disk

log = logging.getLogger(__name__)


def stratified_sample(items: list[dict], n: int, strata_key: str, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    by_stratum: dict[str, list[dict]] = {}
    for r in items:
        by_stratum.setdefault(str(r.get(strata_key, "all")), []).append(r)
    n_strata = len(by_stratum)
    per_stratum = max(1, n // n_strata)
    out: list[dict] = []
    for k, rows in by_stratum.items():
        rng.shuffle(rows)
        out.extend(rows[:per_stratum])
    if len(out) > n:
        out = out[:n]
    return out


def _load_items_from_manifest(manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    items: list[dict] = []
    for name, comp in manifest["components"].items():
        path = Path(comp["path"])
        if not path.exists():
            log.warning("manifest path missing: %s", path)
            continue
        if path.is_dir():
            try:
                ds = load_from_disk(str(path))
                if hasattr(ds, "column_names"):  # Dataset
                    for r in ds:
                        items.append({**r, "_component": name})
                else:                             # DatasetDict
                    for split, sds in ds.items():
                        for r in sds:
                            items.append({**r, "_component": name, "_split": split})
            except Exception as e:
                log.warning("could not load %s: %s", path, e)
        elif path.suffix == ".jsonl":
            for line in path.read_text().splitlines():
                if line.strip():
                    items.append({**json.loads(line), "_component": name})
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--strata", default="subject")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    items = _load_items_from_manifest(args.input)
    log.info("Loaded %d candidate items", len(items))
    sample = stratified_sample(items, n=args.n, strata_key=args.strata, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for i, r in enumerate(sample):
            f.write(json.dumps({"sample_index": i, "decision": "PENDING", **r}) + "\n")
    log.info("Wrote %d stratified samples to %s", len(sample), args.output)


if __name__ == "__main__":
    main()
