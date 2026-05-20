"""Smoke: BIO span projection runs without raising and returns aligned-length lists.

Skips if neither awesome-align nor LaBSE is installed (transformers will be
attempted via the LaBSE fallback; pure-CPU OK on small inputs).
"""
from __future__ import annotations

import pytest


def test_project_bio_runs_on_toy_pair():
    pytest.importorskip("transformers")
    from multilingual.translate.span_project import project_bio_tags

    en = ["Aspirin", "is", "used", "for", "headache", "."]
    en_tags = ["B-Disease", "O", "O", "O", "B-Disease", "O"]
    swh = "Aspirini hutumika kwa maumivu ya kichwa ."

    swh_toks, swh_tags = project_bio_tags(en, en_tags, swh)
    assert len(swh_toks) == len(swh_tags)
    assert any(t != "O" for t in swh_tags), "no spans projected at all"
