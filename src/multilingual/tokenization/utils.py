"""Tokenization utilities (currently small; expand as needed)."""
from __future__ import annotations

from transformers import PreTrainedTokenizerBase


def encode_for_mlm(tokenizer: PreTrainedTokenizerBase, texts, max_length: int = 512):
    """Encode raw text for MLM pretraining (no special multi-segment handling)."""
    return tokenizer(texts, truncation=True, max_length=max_length, padding=False,
                     return_attention_mask=True)
