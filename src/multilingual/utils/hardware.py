"""BF16 / FlashAttention-2 / DGX Spark detection helpers."""
from __future__ import annotations

import logging
import platform

import torch

log = logging.getLogger(__name__)


def is_bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def is_dgx_spark() -> bool:
    """Best-effort detection: GB10 Grace Blackwell on Linux aarch64."""
    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_name(0).lower()
    return "gb10" in name or ("blackwell" in name and platform.machine() == "aarch64")


def best_dtype(prefer_bf16: bool = True) -> torch.dtype:
    if prefer_bf16 and is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def maybe_enable_flash_attn(model_kwargs: dict, requested: bool) -> dict:
    """Try to set attn_implementation=flash_attention_2 in HF model kwargs."""
    if not requested:
        return model_kwargs
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
        log.info("FlashAttention-2 enabled")
    except ImportError:
        log.warning("flash-attn not installed; falling back to SDPA")
        model_kwargs["attn_implementation"] = "sdpa"
    return model_kwargs
