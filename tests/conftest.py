"""Shared pytest fixtures.

We use `xlm-roberta-base` (~280M) as the smoke-test backbone — same architecture
family as AfroXLMR-large but small enough for CPU-only smoke runs.
"""
from __future__ import annotations

import pytest


SMOKE_BACKBONE = "xlm-roberta-base"


@pytest.fixture(scope="session")
def backbone_name() -> str:
    return SMOKE_BACKBONE
