"""NLLB-200-3.3B helper used by data prep and baselines (e), (f).

Cache aggressively — NLLB inference is the throughput bottleneck per §6.2.
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

log = logging.getLogger(__name__)

_DEFAULT = "facebook/nllb-200-3.3B"


@functools.lru_cache(maxsize=2)
def _load_nllb(model_name: str = _DEFAULT) -> tuple:
    log.info("Loading NLLB %s", model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() \
        else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return tok, model


def translate_batch(texts: list[str], src_lang: str, tgt_lang: str, *,
                    model_name: str = _DEFAULT, max_new_tokens: int = 384,
                    cache_dir: str | None = "data/translated") -> list[str]:
    """Translate a list of strings src_lang → tgt_lang. Caches by (model, src, tgt, sha1(text))."""
    tok, model = _load_nllb(model_name)
    out: list[str] = []
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cached: dict[int, str] = {}
    to_run: list[tuple[int, str]] = []
    for i, t in enumerate(texts):
        c = _cache_get(cache_dir, model_name, src_lang, tgt_lang, t)
        if c is not None:
            cached[i] = c
        else:
            to_run.append((i, t))

    if to_run:
        tok.src_lang = src_lang
        idxs, batch = zip(*to_run)
        enc = tok(list(batch), return_tensors="pt", padding=True, truncation=True, max_length=512)
        if torch.cuda.is_available():
            enc = {k: v.to("cuda") for k, v in enc.items()}
        with torch.no_grad():
            forced_bos = tok.convert_tokens_to_ids(tgt_lang)
            generated = model.generate(**enc, forced_bos_token_id=forced_bos,
                                       max_new_tokens=max_new_tokens, num_beams=4)
        decoded = tok.batch_decode(generated, skip_special_tokens=True)
        for idx, src, tgt_text in zip(idxs, batch, decoded):
            cached[idx] = tgt_text
            _cache_put(cache_dir, model_name, src_lang, tgt_lang, src, tgt_text)

    for i in range(len(texts)):
        out.append(cached[i])
    return out


def translate_iter(texts: Iterable[str], src_lang: str, tgt_lang: str, *, batch_size: int = 16,
                   model_name: str = _DEFAULT, cache_dir: str | None = "data/translated"):
    buf: list[str] = []
    for t in texts:
        buf.append(t)
        if len(buf) >= batch_size:
            yield from translate_batch(buf, src_lang, tgt_lang, model_name=model_name,
                                       cache_dir=cache_dir)
            buf = []
    if buf:
        yield from translate_batch(buf, src_lang, tgt_lang, model_name=model_name,
                                   cache_dir=cache_dir)


def _cache_key(model_name: str, src: str, tgt: str, text: str) -> str:
    h = hashlib.sha1(f"{model_name}|{src}|{tgt}|{text}".encode("utf8")).hexdigest()
    return h


def _cache_path(cache_dir: str | None, *args) -> Path | None:
    if not cache_dir:
        return None
    return Path(cache_dir) / f"{_cache_key(*args)}.json"


def _cache_get(cache_dir: str | None, *args) -> str | None:
    p = _cache_path(cache_dir, *args)
    if p is None or not p.exists():
        return None
    try:
        return json.loads(p.read_text())["target"]
    except Exception:
        return None


def _cache_put(cache_dir: str | None, model_name: str, src: str, tgt: str,
               source: str, target: str) -> None:
    p = _cache_path(cache_dir, model_name, src, tgt, source)
    if p is None:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"source": source, "target": target,
                             "model": model_name, "src": src, "tgt": tgt}))
