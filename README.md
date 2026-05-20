# swahili-biomed-adapters

MAD-X-style adapter stacking for cross-lingual domain adaptation in Swahili biomedical NLP.

The pipeline trains a Swahili **language adapter (LA)** via MLM, an English
**biomedical domain adapter (DA)** via MLM, and supervised **task adapters
(TA)** for medical MCQA / NER / topic classification, then composes them at
inference: `[LA_swh → DA_eng → TA]` on a frozen AfroXLMR-large backbone using
the HuggingFace `adapters` library (Poth et al. 2023).

See `compass_artifact_*.md` for the full project plan, literature review, and
benchmark gap analysis.

## Quick start

```bash
# 1. Environment — pick ONE
#  (a) uv (recommended; fast, locks Python version):
curl -LsSf https://astral.sh/uv/install.sh | sh   # one-time, installs to ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"                        # add ,flash on a CUDA-toolchain box

#  (b) plain pip:
pip install -e .[flash,dev]

#  (c) conda:
conda env create -f environment.yml

# 2. Secrets (.env is gitignored; auto-loaded on `import multilingual`)
#    Searched in order: ./.env, src/multilingual/.env, $MULTILINGUAL_ENV_FILE
cat > .env <<'EOF'
HF_TOKEN=hf_xxx                 # mirrored to HUGGING_FACE_HUB_TOKEN automatically
WANDB_API_KEY=xxx               # optional
RESULTS_DIR=./results           # optional; defaults match these
CKPT_DIR=./checkpoints
DATA_DIR=./data
EOF

# 3. Sanity tests (must pass before any real training)
make test                       # full suite — downloads xlm-roberta-base (~1.1GB)
pytest -m "not heavy"           # quick subset, no model download

# 3. Data prep
make data                       # download/clean/dedup/filter all corpora

# 4. Train adapters (frozen backbone; only adapter params updated)
make la                         # Swahili LA, MLM, ~1–2 days on DGX Spark
make da                         # English biomedical DA, MLM, ~1–2 days
make ta                         # MCQA task adapter (seed42; NER pending — see STORY.md)

# 5. Compose + evaluate (§5.5 C1–C10)
make eval-mcqa                  # MMLU-ProX-Swahili clinical subsets — ready
make eval-ner                   # BC5CDR-Disease projected to Swahili — roadmap
make eval-forget                # §8 MasakhaNER-Swahili forgetting probe — roadmap

# 6. Baselines (§5.6)
make baseline-a                 # zero-shot
make baseline-d                 # full DAPT (~2 days)
make baseline-e baseline-f      # translate-train, translate-test
make baseline-g                 # joint LoRA

# 7. Significance + tables
make significance
jupyter notebook notebooks/04_results_tables.ipynb
```

All entry points are Hydra-configurable. Example ablation multirun:
```bash
python scripts/train_la.py -m \
    adapter=pfeiffer_rf4,pfeiffer_rf8,pfeiffer_rf16,pfeiffer_rf32,pfeiffer_rf64 \
    +adapter.invertible=true,false
```

## Repo layout

```
configs/    Hydra YAML for backbone, adapter, data, train, compose, baseline, eval, ablation
src/multilingual/
  backbone.py adapter_setup.py compose.py        # core
  data/ translate/ tokenization/                  # I/O + preprocessing
  training/ eval/ utils/                          # trainers + metrics
scripts/    Thin CLI entry points (one per pipeline stage)
tests/      Smoke + correctness tests
benchmark/  Swa-MedBench v0.1 construction + clinician validation rubric
notebooks/  Corpus stats, fertility, translation QA, results tables
```

## Hardware

Designed for a single NVIDIA DGX Spark (GB10 Grace Blackwell, 128GB unified
LPDDR5x, BF16 + FlashAttention-2). The adapter-only training regime fits
comfortably; full DAPT baseline (§5.6 d) is ~2 days. See `compass_artifact_*.md`
§6 for full feasibility analysis.

## Licenses

See [`NOTICES.md`](NOTICES.md) for per-resource attribution. **OpenWHO is
CC BY-NC-SA 3.0 IGO (non-commercial)**; **NLLB-200 is CC-BY-NC 4.0**. Any
release of derived adapter checkpoints must propagate these notices.

## Disclaimer

Research use only. Not for clinical decision-making.
