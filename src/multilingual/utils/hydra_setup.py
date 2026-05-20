"""OmegaConf resolvers and run-dir hashing."""
from __future__ import annotations

import hashlib
import json

from omegaconf import OmegaConf


def register_resolvers() -> None:
    if not OmegaConf.has_resolver("hash"):
        OmegaConf.register_new_resolver("hash", _hash_short)


def _hash_short(*xs) -> str:
    s = "::".join(str(x) for x in xs)
    return hashlib.sha1(s.encode()).hexdigest()[:8]


def config_hash(cfg) -> str:
    payload = OmegaConf.to_container(cfg, resolve=True) if hasattr(cfg, "_metadata") else cfg
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
