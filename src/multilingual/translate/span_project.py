"""awesome-align span-projection for NER label transfer (English → Swahili).

Projects English BIO tags onto NLLB-translated Swahili tokens by computing
word alignments with awesome-align (mBERT/LaBSE backbone). Falls back to a
LaBSE-cosine alignment if the awesome-align CLI is unavailable.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


def project_bio_tags(en_tokens: list[str], en_tags: list[str], swh_sentence: str,
                     *, aligner_model: str = "bert-base-multilingual-cased") -> tuple[list[str],
                                                                                       list[str]]:
    """Returns (swh_tokens, swh_tags) where tags are BIO labels."""
    swh_tokens = swh_sentence.split()
    align = _align(en_tokens, swh_tokens, model=aligner_model)
    swh_tags = ["O"] * len(swh_tokens)
    for en_i, swh_j in align:
        if en_i >= len(en_tags) or swh_j >= len(swh_tokens):
            continue
        en_tag = en_tags[en_i]
        if en_tag == "O":
            continue
        # B- propagates as B- on first projected swh token; subsequent get I-
        if swh_tags[swh_j] == "O":
            swh_tags[swh_j] = en_tag
        else:
            entity = en_tag.split("-", 1)[1] if "-" in en_tag else en_tag
            swh_tags[swh_j] = f"I-{entity}"
    return swh_tokens, swh_tags


def _align(src: list[str], tgt: list[str], model: str) -> list[tuple[int, int]]:
    """Run awesome-align CLI if available; else fall back to LaBSE cosine."""
    if shutil.which("awesome-align") is not None:
        return _awesome_align(src, tgt, model)
    return _labse_align(src, tgt)


def _awesome_align(src: list[str], tgt: list[str], model: str) -> list[tuple[int, int]]:
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        in_path = tmpd / "pair.txt"
        out_path = tmpd / "aligned.txt"
        in_path.write_text(f"{' '.join(src)} ||| {' '.join(tgt)}\n")
        cmd = ["awesome-align", "--output_file", str(out_path), "--model_name_or_path", model,
               "--data_file", str(in_path), "--extraction", "softmax",
               "--batch_size", "1", "--cache_dir", ".cache/awesome_align"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log.warning("awesome-align failed (%s); falling back to LaBSE", e)
            return _labse_align(src, tgt)
        line = out_path.read_text().strip().splitlines()[0] if out_path.exists() else ""
        out: list[tuple[int, int]] = []
        for pair in line.split():
            try:
                i, j = pair.split("-")
                out.append((int(i), int(j)))
            except ValueError:
                continue
        return out


def _labse_align(src: list[str], tgt: list[str]) -> list[tuple[int, int]]:
    """Fallback: per-source-token argmax cosine to LaBSE-encoded target tokens."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception:
        return [(i, min(i, len(tgt) - 1)) for i in range(len(src))]

    if not src or not tgt:
        return []

    tok = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    model = AutoModel.from_pretrained("sentence-transformers/LaBSE").eval()

    def emb(toks: list[str]) -> "torch.Tensor":
        ids = [tok(t, return_tensors="pt", add_special_tokens=False).input_ids for t in toks]
        out: list[torch.Tensor] = []
        with torch.no_grad():
            for x in ids:
                if x.shape[1] == 0:
                    out.append(torch.zeros(model.config.hidden_size))
                    continue
                h = model(x).last_hidden_state[0]
                out.append(h.mean(dim=0))
        return torch.stack(out)

    e_src = torch.nn.functional.normalize(emb(src), dim=-1)
    e_tgt = torch.nn.functional.normalize(emb(tgt), dim=-1)
    sim = e_src @ e_tgt.T  # (S, T)
    best = sim.argmax(dim=-1).tolist()
    return list(enumerate(best))
