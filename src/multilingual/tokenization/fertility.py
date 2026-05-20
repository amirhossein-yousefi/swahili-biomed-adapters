"""Token fertility (subwords-per-word) for §2.4: XLM-R vs AfroXLMR vs AfriBERTa on Swahili.

Run on a held-out Swahili corpus slice (e.g. OpenWHO Swahili) and report
mean fertility per tokenizer.
"""
from __future__ import annotations

from typing import Iterable

from transformers import AutoTokenizer

DEFAULT_TOKENIZERS = [
    "xlm-roberta-base",
    "xlm-roberta-large",
    "Davlan/afro-xlmr-base",
    "Davlan/afro-xlmr-large",
    "castorini/afriberta_large",
]


def fertility(tokenizer_name: str, sentences: Iterable[str]) -> dict:
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    n_words = 0
    n_subwords = 0
    n_sents = 0
    for s in sentences:
        words = s.split()
        if not words:
            continue
        ids = tok(s, add_special_tokens=False).input_ids
        n_words += len(words)
        n_subwords += len(ids)
        n_sents += 1
    return {
        "tokenizer": tokenizer_name,
        "sentences": n_sents,
        "words": n_words,
        "subwords": n_subwords,
        "fertility": (n_subwords / n_words) if n_words else float("nan"),
    }


def fertility_table(sentences: Iterable[str], tokenizers: list[str] = None) -> list[dict]:
    """Return one row per tokenizer; useful for the §2.4 paper table."""
    sents_list = list(sentences)
    return [fertility(name, sents_list) for name in (tokenizers or DEFAULT_TOKENIZERS)]
