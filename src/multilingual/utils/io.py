"""Run-artifact I/O: dump config, metrics, predictions, package versions."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf


def ensure_dir(p: str | os.PathLike) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | os.PathLike, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def write_jsonl(path: str | os.PathLike, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def write_run_artifacts(out_dir: str | os.PathLike, cfg: DictConfig | dict, metrics: dict,
                        predictions: list[dict] | None = None) -> Path:
    out = ensure_dir(out_dir)
    if isinstance(cfg, DictConfig):
        Path(out, "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    else:
        Path(out, "config.yaml").write_text(yaml.safe_dump(cfg))
    write_json(out / "metrics.json", metrics)
    if predictions is not None:
        write_jsonl(out / "predictions.jsonl", predictions)
    Path(out, "git_sha.txt").write_text(_git_sha())
    Path(out, "package_versions.json").write_text(json.dumps(_pkg_versions(), indent=2))
    return out


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return "no-git"


def _pkg_versions() -> dict:
    pkgs = ["torch", "transformers", "adapters", "datasets", "accelerate", "numpy"]
    out: dict[str, str] = {"python": sys.version}
    for p in pkgs:
        try:
            mod = __import__(p)
            out[p] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[p] = "not-installed"
    return out
