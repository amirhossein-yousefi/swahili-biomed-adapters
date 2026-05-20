"""Held-out MLM perplexity on OpenWHO Swahili — sanity check that the LA/DA is doing something."""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from ..data.mlm_collator import make_mlm_collator


@torch.no_grad()
def held_out_mlm_perplexity(model, tokenizer, sentences: Iterable[str], *, max_length: int = 512,
                            batch_size: int = 8, device: str | None = None,
                            mlm_prob: float = 0.15, seed: int = 42) -> dict:
    """Mask 15% of tokens, average cross-entropy, exponentiate → 'pseudo-perplexity'.

    Note: encoder MLM perplexity is not classical LM perplexity; it's a fixed-mask
    pseudo-PPL used widely as a probe (Salazar et al. 2020). We only use it as a
    relative diagnostic, not as an absolute number.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    torch.manual_seed(seed)

    encs = [tokenizer(s, truncation=True, max_length=max_length) for s in sentences]
    collator = make_mlm_collator(tokenizer, mlm_prob=mlm_prob)

    losses: list[float] = []
    counts: list[int] = []
    for i in range(0, len(encs), batch_size):
        batch = collator(encs[i:i + batch_size])
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        # HF returns mean loss over masked positions
        n_mask = (batch["labels"] != -100).sum().item()
        if n_mask == 0:
            continue
        losses.append(float(out.loss) * n_mask)
        counts.append(n_mask)
    total = sum(losses) / max(1, sum(counts))
    return {"loss": total, "perplexity": math.exp(total), "n_masked_tokens": sum(counts)}
