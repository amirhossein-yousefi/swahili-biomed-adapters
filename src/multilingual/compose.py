"""Composition matrix C1–C10 for §5.5.

Single API every eval script calls. No Pfeiffer / Fuse / Parallel / Stack
internals here — those are provided by the `adapters` library. This module is
a config-string → `model.active_adapters` dispatcher.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
from adapters.composition import Fuse, Parallel, Stack

log = logging.getLogger(__name__)


def describe(config_id: str) -> str:
    table = {
        "C1": "base only (no adapters)",
        "C2": "LA_swh only",
        "C3": "DA_eng only",
        "C4": "Stack(LA_swh, DA_eng, TA)  ★ MAD-X analog",
        "C5": "Stack(DA_eng, LA_swh, TA)  reversed order",
        "C6": "Stack(Fuse(LA_swh, DA_eng), TA)  AdapterFusion",
        "C7": "Stack(Parallel(LA_swh, DA_eng), TA)  He et al. 2021 parallel",
        "C8": "Stack(LA_DA_joint, TA)  BAD-X-style joint",
        "C9": "Stack(merged_LA_DA, TA)  TIES weight-merge",
        "C10": "Hyper-X hypernetwork",
    }
    return table.get(config_id, f"unknown({config_id})")


def _load_named(model, name: str, path: str) -> None:
    """Idempotently load an adapter under `name` from `path`.

    Adapter and head weights are typically saved at float32 even when the
    backbone runs at bf16, so we harmonize dtypes after each load to avoid
    `mat1 and mat2 must have the same dtype` errors at forward time.
    """
    if name in getattr(model, "adapters_config", {}).adapters:
        log.info("Adapter %r already loaded; skipping", name)
        return
    loaded = model.load_adapter(path, load_as=name, set_active=False)
    log.info("Loaded adapter %r from %s as %r", loaded, path, name)
    _harmonize_dtype(model)


def _harmonize_dtype(model) -> None:
    """Cast all floating-point parameters to the model's dominant dtype.

    Uses the dtype of a frozen-base parameter as ground truth (since the
    backbone was loaded under that dtype via `load_backbone`). Newly loaded
    adapter / head tensors that arrived as float32 get downcast to bf16.
    """
    base_dtype = None
    for n, p in model.named_parameters():
        if "adapter" not in n.lower() and "heads" not in n.lower() and p.is_floating_point():
            base_dtype = p.dtype
            break
    if base_dtype is None:
        return
    n_cast = 0
    for p in model.parameters():
        if p.is_floating_point() and p.dtype != base_dtype:
            p.data = p.data.to(base_dtype)
            n_cast += 1
    if n_cast:
        log.info("Harmonized %d tensors to %s", n_cast, base_dtype)


def build_composition(model, config_id: str, adapters_paths: dict[str, str], *,
                      fusion_ckpt: Optional[str] = None,
                      merge_method: str = "ties") -> Any:
    """Set up the inference-time adapter stack on `model` per §5.5 C{1..10}.

    Args:
        model:           an `adapters.AutoAdapterModel`-like model.
        config_id:       "C1" .. "C10".
        adapters_paths:  map of role → checkpoint dir, e.g.
                         {"LA_swh": ".../la_swh", "DA_eng": ".../da_eng_bio_on_base",
                          "TA": ".../ta_mcqa/seed42"}.
                         Required keys vary by config — see README of compose/.
        fusion_ckpt:     path to a saved AdapterFusion bundle (C6). If None,
                         attempts to add an untrained fusion module (still
                         meaningful for ablation but expect lower performance).
        merge_method:    "ties" (Yadav et al. 2023) or "linear" (C9).

    Returns:
        the same model with `model.active_adapters` configured.
    """
    cid = config_id.upper()
    log.info("Configuring composition %s — %s", cid, describe(cid))

    if cid == "C1":
        model.active_adapters = None
        return model

    if cid == "C2":
        _load_named(model, "LA_swh", adapters_paths["LA_swh"])
        _load_named(model, "TA", adapters_paths["TA"])
        model.active_adapters = Stack("LA_swh", "TA")
        return model

    if cid == "C3":
        _load_named(model, "DA_eng", adapters_paths["DA_eng"])
        _load_named(model, "TA", adapters_paths["TA"])
        model.active_adapters = Stack("DA_eng", "TA")
        return model

    if cid == "C4":
        _load_named(model, "LA_swh", adapters_paths["LA_swh"])
        _load_named(model, "DA_eng", adapters_paths["DA_eng"])
        _load_named(model, "TA", adapters_paths["TA"])
        model.active_adapters = Stack("LA_swh", "DA_eng", "TA")
        return model

    if cid == "C5":
        _load_named(model, "LA_swh", adapters_paths["LA_swh"])
        _load_named(model, "DA_eng", adapters_paths["DA_eng"])
        _load_named(model, "TA", adapters_paths["TA"])
        model.active_adapters = Stack("DA_eng", "LA_swh", "TA")
        return model

    if cid == "C6":
        _load_named(model, "LA_swh", adapters_paths["LA_swh"])
        _load_named(model, "DA_eng", adapters_paths["DA_eng"])
        _load_named(model, "TA", adapters_paths["TA"])
        if fusion_ckpt and Path(fusion_ckpt).exists():
            model.load_adapter_fusion(fusion_ckpt, set_active=False)
        else:
            model.add_adapter_fusion(["LA_swh", "DA_eng"])
        model.active_adapters = Stack(Fuse("LA_swh", "DA_eng"), "TA")
        return model

    if cid == "C7":
        _load_named(model, "LA_swh", adapters_paths["LA_swh"])
        _load_named(model, "DA_eng", adapters_paths["DA_eng"])
        _load_named(model, "TA", adapters_paths["TA"])
        model.active_adapters = Stack(Parallel("LA_swh", "DA_eng"), "TA")
        return model

    if cid == "C8":
        _load_named(model, "LA_DA_joint", adapters_paths["LA_DA_joint"])
        _load_named(model, "TA", adapters_paths["TA"])
        model.active_adapters = Stack("LA_DA_joint", "TA")
        return model

    if cid == "C9":
        _load_named(model, "LA_swh", adapters_paths["LA_swh"])
        _load_named(model, "DA_eng", adapters_paths["DA_eng"])
        _load_named(model, "TA", adapters_paths["TA"])
        merge_adapters_into(model, sources=["LA_swh", "DA_eng"], target="LA_DA_merged",
                            method=merge_method)
        model.active_adapters = Stack("LA_DA_merged", "TA")
        return model

    if cid == "C10":
        # Hyper-X: load a hypernetwork that emits LA, DA on demand. Optional/last.
        _load_named(model, "HyperX", adapters_paths["HyperX"])
        _load_named(model, "TA", adapters_paths["TA"])
        model.active_adapters = Stack("HyperX", "TA")
        return model

    raise ValueError(f"Unknown composition id: {config_id}")


def merge_adapters_into(model, sources: list[str], target: str, *, method: str = "ties",
                        density: float = 0.2) -> None:
    """TIES (Yadav et al. 2023) or simple linear merge of adapter weights.

    Implementation:
      1. Pull state_dicts of each source adapter.
      2. For TIES: keep top-`density` magnitudes per param, resolve sign by majority,
         then average the surviving values.
      3. For linear: average param-wise.
      4. Add a fresh adapter with the same config under `target`, load merged
         state_dict into it.
    """
    src_states = {s: _adapter_state_dict(model, s) for s in sources}
    keys = next(iter(src_states.values())).keys()

    merged: dict[str, torch.Tensor] = {}
    for k in keys:
        stacked = torch.stack([src_states[s][k].float() for s in sources], dim=0)  # (S, *)
        if method == "linear":
            merged[k] = stacked.mean(dim=0)
        elif method == "ties":
            merged[k] = _ties_merge(stacked, density=density)
        else:
            raise ValueError(f"Unknown merge method: {method}")

    cfg = model.adapters_config.get(sources[0])
    model.add_adapter(target, config=cfg)
    _load_into_adapter(model, target, merged)
    log.info("Merged adapters %s -> %r via %s", sources, target, method)


def _ties_merge(stacked: torch.Tensor, density: float) -> torch.Tensor:
    """TIES: trim → elect sign → disjoint-mean."""
    S = stacked.shape[0]
    flat = stacked.view(S, -1)
    abs_flat = flat.abs()
    k = max(1, int(density * flat.shape[1]))
    thresholds = abs_flat.topk(k, dim=1).values[:, -1:].clamp_min_(1e-12)
    keep = abs_flat >= thresholds
    trimmed = flat * keep
    sign = trimmed.sum(dim=0).sign()
    sign[sign == 0] = 1.0
    aligned = trimmed * (trimmed.sign() == sign).float()
    counts = (aligned != 0).sum(dim=0).clamp_min_(1).float()
    out = aligned.sum(dim=0) / counts
    return out.view(stacked.shape[1:]).to(stacked.dtype)


def _adapter_state_dict(model, name: str) -> dict[str, torch.Tensor]:
    full = model.state_dict()
    needle = f".{name}."
    return {k: v for k, v in full.items() if needle in k}


def _load_into_adapter(model, name: str, weights: dict[str, torch.Tensor]) -> None:
    full = model.state_dict()
    needle = f".{name}."
    target_keys = [k for k in full if needle in k]
    src_by_suffix = {k.split(needle, 1)[1]: v for k, v in weights.items()}
    incoming: dict[str, torch.Tensor] = {}
    for tk in target_keys:
        suffix = tk.split(needle, 1)[1]
        if suffix in src_by_suffix:
            incoming[tk] = src_by_suffix[suffix].to(full[tk].dtype)
    missing = set(target_keys) - set(incoming)
    if missing:
        log.warning("merge: %d target keys had no source match (e.g. %s)", len(missing),
                    next(iter(missing), "—"))
    model.load_state_dict(incoming, strict=False)
