"""Significance testing per §7.2.

  - paired_bootstrap (Koehn 2004): 1000 resamples on per-example outputs.
  - almost_stochastic_order (Dror et al. 2019): better for small effect sizes.

Inputs are per-example scalar scores (0/1 for accuracy, F1 by sentence, etc.).
"""
from __future__ import annotations

import numpy as np


def paired_bootstrap(scores_a: np.ndarray, scores_b: np.ndarray, *, n_resamples: int = 1000,
                     seed: int = 42) -> dict:
    """Two-sided paired bootstrap. Returns p-value and CI of mean(a) - mean(b)."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = a - b
    obs = float(diffs.mean())
    boot = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot[i] = diffs[idx].mean()
    centered = boot - obs
    p_two = float(np.mean(np.abs(centered) >= abs(obs)))
    return {
        "mean_diff": obs,
        "ci_lower": float(np.quantile(boot, 0.025)),
        "ci_upper": float(np.quantile(boot, 0.975)),
        "p_value": p_two,
        "n": int(n),
        "n_resamples": int(n_resamples),
    }


def almost_stochastic_order(scores_a: np.ndarray, scores_b: np.ndarray, *,
                            confidence: float = 0.95) -> dict:
    """Wraps deepsig.aso (Dror et al. 2019). Returns the ASO score (lower = a better)."""
    try:
        from deepsig import aso
    except ImportError:
        return {"error": "deepsig not installed; pip install deepsig"}
    score = aso(scores_a, scores_b, confidence_level=confidence)
    return {"aso": float(score), "confidence": float(confidence)}
