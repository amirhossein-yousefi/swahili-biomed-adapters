"""Paired bootstrap recovers known p-value on synthetic data."""
from __future__ import annotations

import numpy as np


def test_paired_bootstrap_clear_difference():
    from multilingual.eval.significance import paired_bootstrap

    rng = np.random.default_rng(0)
    n = 200
    a = rng.binomial(1, 0.75, size=n)
    b = rng.binomial(1, 0.55, size=n)
    out = paired_bootstrap(a, b, n_resamples=1000, seed=42)
    assert out["mean_diff"] > 0
    assert out["p_value"] < 0.05


def test_paired_bootstrap_no_difference():
    from multilingual.eval.significance import paired_bootstrap

    rng = np.random.default_rng(1)
    n = 200
    p = 0.6
    a = rng.binomial(1, p, size=n)
    b = rng.binomial(1, p, size=n)
    out = paired_bootstrap(a, b, n_resamples=500, seed=42)
    assert out["p_value"] > 0.1, f"identical-distribution p_value too low: {out['p_value']}"
