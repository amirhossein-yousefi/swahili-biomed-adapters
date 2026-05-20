"""Minimal .env loader (no python-dotenv dependency).

Searches in order, loads the FIRST one found (first match wins; later files do not override):
  1. ./.env              (repo root, conventional location)
  2. src/multilingual/.env (next to this file, fallback)
  3. $MULTILINGUAL_ENV_FILE if set

Existing process env-vars are NEVER overwritten — set in the shell to override.

Picked-up keys of note:
  HF_TOKEN / HUGGING_FACE_HUB_TOKEN  → datasets + transformers gated repos
  WANDB_API_KEY                       → wandb auto-login
  RESULTS_DIR / CKPT_DIR / DATA_DIR  → output paths used by configs/default.yaml
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

_LOADED = False


def load_dotenv(*, verbose: bool = False) -> Path | None:
    """Load the first .env we find. Returns the path used or None.

    Idempotent: subsequent calls are no-ops.
    """
    global _LOADED
    if _LOADED:
        return None
    _LOADED = True

    here = Path(__file__).resolve()
    pkg_dir = here.parent.parent          # .../src/multilingual
    repo_root = pkg_dir.parent.parent     # .../multi-lingual

    candidates: list[Path] = [
        repo_root / ".env",
        pkg_dir / ".env",
    ]
    if extra := os.environ.get("MULTILINGUAL_ENV_FILE"):
        candidates.insert(0, Path(extra))

    for path in candidates:
        if not path.is_file():
            continue
        n_keys = _apply_env_file(path)
        if verbose:
            log.info("Loaded %d env keys from %s", n_keys, path)
        # mirror HF_TOKEN <-> HUGGING_FACE_HUB_TOKEN since the libs disagree
        if "HF_TOKEN" in os.environ and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
        elif "HUGGING_FACE_HUB_TOKEN" in os.environ and "HF_TOKEN" not in os.environ:
            os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]
        return path
    return None


def _apply_env_file(path: Path) -> int:
    """Parse a simple KEY=VALUE .env file. Skip blank/comment lines."""
    n = 0
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            n += 1
    return n
