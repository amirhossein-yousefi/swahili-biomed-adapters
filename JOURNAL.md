# Project Journal

Working record of everything that's been built, run, fixed, and learned for the
Swahili biomedical adapter-stacking study. Read top-to-bottom; the order is
chronological (May 10 → ongoing). Future runs / next steps live at the bottom.

---

## 0. Environment

- **uv**: installed at `~/.local/bin/uv` via the official `astral.sh/uv/install.sh`
  script. Always `export PATH="$HOME/.local/bin:$PATH"` first.
- **Venv**: `.venv/` at repo root, Python 3.10.20 (set by `uv venv --python 3.10`).
- **Install**: `uv pip install -e ".[dev]"` (skip `flash` extra — it requires
  the CUDA toolchain; the pipeline falls back to SDPA cleanly).
- **Secrets**: HF token lives at `src/multilingual/.env`
  (`HF_TOKEN=hf_...`). Auto-loaded on `import multilingual` via
  [src/multilingual/utils/env.py](src/multilingual/utils/env.py); mirrored to
  `HUGGING_FACE_HUB_TOKEN`. `.env` is gitignored.
- **Shorthand**:
  ```bash
  export PATH="$HOME/.local/bin:$PATH"
  source .venv/bin/activate
  export EXPORTS="RESULTS_DIR=$PWD/results CKPT_DIR=$PWD/checkpoints DATA_DIR=$PWD/data"
  ```
  Every `make` target or CLI invocation expects these.

---

## 1. Tests — all pass

```bash
pytest                              # 8 tests, ~70s incl. xlm-roberta-base download
pytest -m "not heavy"               # 5 light tests, ~1s
```

Light suite includes paired-bootstrap recovery, mlm-collator masking ratio,
span-projection, compose-config table, etc. Heavy adds adapter-attach +
C1≡base equivalence + MLM micro-run.

---

## 2. Data prep — status by source

The plan's "best-effort" repo names mostly didn't survive `datasets >= 4.0`
which dropped support for loading scripts. Each source below documents the
actual working repo + any post-fix landed.

| Source | HF repo (actually working) | Volume | Manifest |
|---|---|---|---|
| MMLU-ProX-sw clinical | `li-lab/MMLU-ProX` config `sw`, filter `src` to 4 clinical subjects | **331 questions** (CK 72, CM 48, PM 165, Vir 46) | `data/mmlu_prox_swh_test/` |
| BC5CDR English | `tner/bc5cdr` via direct JSONL download (5-class → Disease-only collapse) | 5228/5330/5865 (train/val/test) | `data/bc5cdr_en/` |
| BC5CDR Swahili (NLLB-translated + span-projected) | built from English test → `facebook/nllb-200-3.3B` → awesome-align fallback to LaBSE | **5865 sentences with BIO tags** | `data/bc5cdr_swh_test.jsonl` |
| MedMCQA | `openlifescienceai/medmcqa` | 182,822 / 4,183 / 6,150 | `data/medmcqa_train/` |
| MasakhaNER Swahili | direct download from `https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main/MasakhaNER2.0/data/swa/{train,dev,test}.txt` | 6593 / 942 / 1883 | `data/masakhaner_swh/` |
| OpenWHO Swahili | `raphaelmerx/openwho` config `sent__swa` (**gated** — must request access at https://huggingface.co/datasets/raphaelmerx/openwho first) | **107 sentences only** (doc's "26,824" was multi-lang total) | `data/openwho_swh_eval.txt` |
| Swahili raw corpus | mC4 (`allenai/c4`, `sw`) + WURA (`llama-lang-adapt/wura`, `sw`) + Wikipedia (`wikimedia/wikipedia`, `20231101.sw`) | **2,042,338 sentences** after sentence-split + dedup + LID@p>0.5 | `data/swh_raw.txt` |
| English biomedical raw | `slinusc/PubMedAbstractsSubset` (parquet, streaming) | streamed up to `max_abstracts` | `data/bio_raw.txt` (**not yet built end-to-end**) |
| MedlinePlus Swahili | scrape `medlineplus.gov/languages/swahili.html` | **not yet run** | — |

### Data prep — fixes that landed (so they don't bite again)

- **Hydra struct mode**: ad-hoc keys (`source=...`, `mode=...`) need `+` prefix
  → Makefile uses `+source=...`. Applies to `prepare_data.py` and
  `translate_corpus.py`.
- **Script-based HF datasets** (`cc100`, `castorini/wura`, `ncbi/pubmed`,
  `tner/bc5cdr`, `masakhane/masakhaner2`, etc.) fail with "Dataset scripts
  are no longer supported". Workarounds in code:
  - Find a parquet/JSONL alternative (WURA via `llama-lang-adapt/wura`).
  - Or fetch the raw data files directly via `hf_hub_download` / `requests`
    (BC5CDR, MasakhaNER, PubMed abstracts).
- **OpenWHO**: gated — request access manually before running.
- **`masakhane/african_corpus`**: doesn't exist; removed the placeholder.
- **FastText LID × NumPy 2.x**: `fasttext-wheel` uses `np.array(..., copy=False)`
  which NumPy ≥2 rejects. Monkey-patched in
  [src/multilingual/data/lid.py](src/multilingual/data/lid.py) (`_patch_fasttext_for_numpy2`).
- **LID threshold**: doc's `p>0.9` retains <2% of Swahili sentences (FastText
  `lid.176` is calibrated low on Swahili). Lowered default to **0.5** in
  [configs/data/la_swh.yaml](configs/data/la_swh.yaml). Documented in the lid
  module.
- **LID input granularity**: must sentence-split paragraphs BEFORE LID,
  otherwise even genuine Swahili paragraphs get rejected. The `split_sentences`
  helper now runs upstream of LID.
- **MMLU-ProX schema**: subset is `sw` (not `swa`); up to 10 options per row
  with trailing `None`s; correct answer in `answer_index`; original MMLU
  subject in `src` field (e.g. `ori_mmlu-clinical_knowledge`).
  [src/multilingual/data/mmlu_prox_swh.py](src/multilingual/data/mmlu_prox_swh.py)
  exposes both `_clinical()` (4 subjects, 331 qs) and `_health()` (full
  health category, 687 qs).
- **Option count for MCQA**: 71% of MMLU-ProX-sw clinical are 10-option,
  47% of answers sit at index ≥4. **Must train and eval with
  `num_choices=10`**, padding shorter MCQs with empty strings. Set on CLI
  (`train.num_choices=10`, `eval.num_choices=10`).

---

## 3. Training — what's verified

### 3.1 Hydra plumbing

All scripts now use `config_name="default"` so they inherit the top-level
`run:`, `backbone:`, `compose:`, `eval:` blocks. Per-script overrides happen
via CLI (`train=ta_mcqa`, `compose=c2_la_only`, etc.).

### 3.2 LA training — smoke (2,000 steps)

Command (run via `train_la.py train.max_steps=2000 ...`).
Saved: `results/runs/2026-05-10_18-40-10/la_swh/LA_swh/`
Symlinked to: `checkpoints/la_swh/`

**MLM eval-loss on OpenWHO held-out (107 sentences)**:
```
step  500: 5.976
step 1000: 4.198    ↓ 1.78
step 1500: 3.401    ↓ 0.80
step 2000: 2.743    ↓ 0.66
```
**54% reduction in 2000 steps** — far above the §9 "≥10%" smoke threshold.
Wall-clock: 33 min on GB10 at BS=16, GA=4, 1.07 s/step.

### 3.3 TA training — full (5 epochs / 57,130 steps)

Command (run via `train_ta.py train=ta_mcqa train.la_path=null train.da_path=null
train.eval_swh=false train.num_choices=10`).
Saved: `results/runs/2026-05-11_00-28-31/ta_mcqa/seed42/TA/`
Symlinked to: `checkpoints/ta_mcqa/seed42/TA/`

**MCQA eval-loss on MedMCQA validation (4183 questions)**:
```
step   500: 2.049   (≈ log 10, random)
step  3000: 1.386   (≈ log 4 — learned to discount empty pads)
step 33000: 1.345   (best — slight learning beyond random-on-real-options)
step 57130: 1.345   (final)
```
The adapter learned to discount the 6 empty pads but barely separates the 4
real MedMCQA options — that's the capacity ceiling of TA-only training on
English with no LA/DA underneath.
Wall-clock: ~11h on GB10.

### 3.4 Composition + eval — verified end-to-end

`scripts/eval_mcqa.py compose=c2_la_only eval.num_choices=10`:

- Loads LA_swh + TA via `compose.build_composition(model, "C2", paths)`.
- Casts FP32 adapter weights down to BF16 to match the model (`_harmonize_dtype`).
- Iterates 331 MMLU-ProX-sw clinical MCQs through per-option scoring.
- Reports accuracy + ECE + Brier + per-subject breakdown.
- Dumps `config.yaml`, `metrics.json`, `predictions.jsonl`, `git_sha.txt`
  to `results/runs/.../eval_mcqa_C2/`.

**C2 (LA_swh-2K-steps + TA-fully-trained) on MMLU-ProX-sw clinical**:
| metric | value |
|---|---|
| accuracy | **0.1118** |
| ECE | 0.058 |
| Brier | 0.887 |
| n | 331 |

Both smoke and full TA land near the ~12.5% random baseline. This is the
honest **C2 baseline number for a 2K-step LA**. To beat it we need a real LA
(full training) and ideally the DA on top.

---

## 4. Footguns / things that bit us (sorted by likelihood of re-occurrence)

1. **`fasttext-wheel` + NumPy 2.x**: handled via monkey-patch in `lid.py`.
   If you upgrade fasttext or numpy, re-verify.
2. **HF script-based datasets**: any new dataset you reach for, expect to
   need a parquet alternative or direct file download. Probe with
   `huggingface_hub.HfApi().dataset_info(...).siblings` before assuming
   `load_dataset(name)` works.
3. **BF16 / FP32 mismatch on adapter load**: HF `adapters` lib saves weights
   in FP32 regardless of the model's dtype.
   [src/multilingual/compose.py:`_harmonize_dtype`](src/multilingual/compose.py)
   handles this — keep calling it after every `load_adapter`.
4. **`load_best_model_at_end` requires matching save/eval strategies** in
   transformers. Already fixed in
   [src/multilingual/training/task_trainer.py](src/multilingual/training/task_trainer.py).
5. **`Dataset.from_generator` + complex closures**: dill chokes when
   fingerprinting a closure that contains FastText / model / dedup state.
   Workaround: write data to disk first (`prepare_data.py`), then read via
   `load_dataset("text", data_files=...)` in the trainer.
6. **MMLU-ProX option count**: `num_choices=10` is non-negotiable for
   Swahili eval. Train and eval must use the same value.
7. **OpenWHO sample size**: 107 sentences is too small for stable held-out
   perplexity numbers. The second `trainer.evaluate()` call (post-save)
   can swing 0.5+ from the in-training number due to random MLM masking.
   Use as a relative/directional probe, not absolute.
8. **`tail -40 | python` buffering**: foreground `python … | tail -N`
   blocks output until pipe close. Always `python -u … > /tmp/log 2>&1`
   instead, and `tail -f /tmp/log` to follow.
9. **Terminal SIGHUP**: an early LA smoke died because the terminal session
   ended. Always `nohup … &` for multi-hour jobs.

---

## 5. Checkpoint / symlink conventions

The trainers write into `results/runs/<timestamp>/.../<name>/` (per the
Hydra run dir). The compose configs expect adapters at
`${ckpt_dir}/<role>` (= `./checkpoints/<role>`). **Symlink after each
training run**:

```bash
# After train_la.py
LATEST=$(ls -td results/runs/*/ | head -1)
ln -sfn "$PWD/$LATEST/la_swh/LA_swh" "$PWD/checkpoints/la_swh"

# After train_ta.py
ln -sfn "$PWD/$LATEST/ta_mcqa/seed42/TA" "$PWD/checkpoints/ta_mcqa/seed42/TA"

# Future: after train_da.py
ln -sfn "$PWD/$LATEST/da_eng_bio_on_base/DA_eng" "$PWD/checkpoints/da_eng_bio_on_base"
```

Don't move/rename run dirs — the symlinks would break.

---

## 6. Currently running / queued

- **Full LA training** (in progress as of 2026-05-12). `nohup` background.
  Default config (3 epochs on `data/swh_raw.txt`, no `max_steps` cap).
  Log at `/tmp/la_full.log`. ~250K steps expected, ~1–2 days on GB10.
  When it finishes:
  ```bash
  LATEST=$(ls -td results/runs/*/ | head -1)
  ln -sfn "$PWD/$LATEST/la_swh/LA_swh" "$PWD/checkpoints/la_swh"
  # Re-measure C2 with the real LA:
  env $EXPORTS python -u scripts/eval_mcqa.py compose=c2_la_only eval.num_choices=10
  ```

---

## 7. Roadmap — what to run next, in order

### 7.1 Verify full LA helps (after current run finishes)
1. Symlink the full LA (see §5).
2. Re-eval C2 (LA_swh + fully-trained TA). Expect accuracy to move above
   ~12% if the LA is doing its job. If it doesn't move much, the bottleneck
   is the TA (likely — English→Swahili cross-lingual transfer without
   translated supervised data is genuinely hard).
3. Optionally also run the §8 forgetting probe to verify the LA hasn't
   broken general-domain Swahili NER:
   ```bash
   env $EXPORTS python -u scripts/eval_forgetting.py compose=c2_la_only
   ```

### 7.2 Build the DA (~1–2 days)
1. Fetch PubMed abstracts (loader is already fixed):
   ```bash
   env $EXPORTS python -u scripts/prepare_data.py +source=bio
   # OR for a quicker first pass:
   env $EXPORTS python -u scripts/prepare_data.py +source=bio +max_abstracts=100000
   ```
   Expect `data/bio_raw.txt` with up to 1M abstracts (~1.5 GB).
2. Train DA on the frozen base (`train.mode=on_base`):
   ```bash
   nohup env $EXPORTS python -u scripts/train_da.py train=da_mlm \
       > /tmp/da_full.log 2>&1 &
   ```
   ~1–2 days.
3. Symlink:
   ```bash
   LATEST=$(ls -td results/runs/*/ | head -1)
   ln -sfn "$PWD/$LATEST/da_eng_bio_on_base/DA_eng" "$PWD/checkpoints/da_eng_bio_on_base"
   ```

### 7.3 Retrain TA on top of the [LA_en → DA] stack (MAD-X canonical recipe)
The current TA was trained on raw base. The MAD-X recipe trains TA on top
of `LA_en` (English source-language adapter) so the TA learns "task given
language-baseline". For full reproduction this needs an `LA_en` checkpoint
too — train a small English MLM adapter via:
```bash
env $EXPORTS python -u scripts/train_la.py data=la_en
```
(la_en.yaml already exists in [configs/data/](configs/data/la_en.yaml).)

Then retrain TA:
```bash
nohup env $EXPORTS python -u scripts/train_ta.py train=ta_mcqa \
    train.la_path=$PWD/checkpoints/la_en train.da_path=$PWD/checkpoints/da_eng_bio_on_base \
    train.num_choices=10 train.eval_swh=false \
    > /tmp/ta_full_madx.log 2>&1 &
```

### 7.4 Run C4 — the headline composition
```bash
env $EXPORTS python -u scripts/eval_mcqa.py compose=c4_la_da_ta eval.num_choices=10
```

C4 is `Stack(LA_swh, DA_eng, TA)`. At inference the English `LA_en` is
swapped for `LA_swh` — that's the MAD-X cross-lingual zero-shot trick.

### 7.5 Composition matrix C1–C9 + ablations
After C4 lands, sweep the rest of §5.5:
```bash
for cid in c1_base c2_la_only c3_da_only c4_la_da_ta c5_da_la_ta c7_parallel; do
    env $EXPORTS python -u scripts/eval_mcqa.py compose=$cid eval.num_choices=10
done
```
C6 (AdapterFusion) needs a separate small fusion-training step; C8 (BAD-X
joint) and C9 (TIES merge) need separate setup. Defer until C1–C5 numbers
are landed.

### 7.6 NER evaluation
```bash
# First, train TA-NER on top of LA_en + DA (currently not done):
env $EXPORTS python -u scripts/train_ta.py train=ta_ner \
    train.la_path=$PWD/checkpoints/la_en train.da_path=$PWD/checkpoints/da_eng_bio_on_base

# Then eval on the NLLB-projected Swahili BC5CDR:
env $EXPORTS python -u scripts/eval_ner.py compose=c4_la_da_ta
```

### 7.7 Forgetting probe + significance + clinician validation
The §9 non-negotiables before any paper:
- `eval_forgetting.py` ΔF1 on MasakhaNER-Swahili.
- 3 seeds (`-m run.seed=42,7,123`) for each primary config.
- Clinician-validated subset of MMLU-ProX-sw clinical and BC5CDR-sw test
  (see [benchmark/clinician_validation/rubric.md](benchmark/clinician_validation/rubric.md)
  and [benchmark/clinician_validation/sampling.py](benchmark/clinician_validation/sampling.py)).

---

## 8. Numbers landed so far (will accumulate)

| Date | Stack | Eval set | n | acc | ECE | Brier | run dir |
|---|---|---|---|---|---|---|---|
| 2026-05-10 | C2 (LA 2K-step + TA 572-step) | MMLU-ProX-sw clinical | 331 | 0.1239 | 0.011 | 0.896 | `results/runs/2026-05-10_22-47-…` |
| 2026-05-12 | C2 (LA 2K-step + TA 57K-step) | MMLU-ProX-sw clinical | 331 | 0.1118 | 0.058 | 0.887 | `results/runs/2026-05-12_09-57-…` |
| 2026-05-17 | C2 (LA 95.7K-step + TA 57K-step) | MMLU-ProX-sw clinical | 331 | **0.1269** | 0.059 | 0.885 | `results/runs/2026-05-17_17-38-04/eval_mcqa_C2/` |

Breakdown by subject (2026-05-17 C2): Clinical Knowledge 9.7 % (n=72),
College Medicine 10.4 % (n=48), Professional Medicine 13.9 % (n=165),
Virology 15.2 % (n=46). Full LA bought only +1.5 absolute points over the
2K-step LA — bottleneck is the TA, not the LA. The TA was trained on English
MedMCQA on the raw frozen base, so it can't make use of `LA_swh`'s
Swahili-shaped representations at inference. The MAD-X canonical recipe
(retrain TA on top of LA_en, then swap LA_en → LA_swh at inference) is what
should bridge this — see roadmap §7.3.

| 2026-05-18 | C2 (LA_swh 95.7K + TA-on-LA_en 57.1K, MAD-X canonical) | MMLU-ProX-sw clinical | 331 | **0.1360** | 0.040 | 0.879 | `results/runs/2026-05-18_22-40-37/eval_mcqa_C2/` |

Breakdown by subject (2026-05-18 C2): Clinical Knowledge 12.5 % (+2.8 vs
2026-05-17), College Medicine 14.6 % (+4.2), Professional Medicine 14.5 %
(+0.6), Virology 10.9 % (−4.3, n=46 — likely noise). MAD-X canonical recipe
fires in the expected direction (lifts on 3 of 4 subjects, biggest on the
two that were below random; ECE drops from 0.059 → 0.040 — better-calibrated
too), but magnitude is modest: +0.9 over the raw-base TA, +2.4 total from
the 2K-LA baseline. The cross-lingual transfer mechanism works; the task
supervision from English MedMCQA alone doesn't bridge the distribution gap
to Swahili clinical MCQA. Next: Path B — add DA on PubMed and evaluate C3 +
C4. LA_en checkpoint at `checkpoints/la_en_run/LA_en/` (10K steps, eval_loss
1.68); MAD-X TA at `results/runs/2026-05-18_05-35-09/ta_mcqa/seed42/TA/`
(symlinked to checkpoints/ta_mcqa/seed42/TA).

Update this table after every eval run.

---

## 9. Files modified vs. the original scaffold

(Each was changed at least once to handle a real-world dataset/library quirk
discovered during runtime. Re-review if you re-clone or branch.)

- [src/multilingual/data/mmlu_prox_swh.py](src/multilingual/data/mmlu_prox_swh.py) — schema rewrite
- [src/multilingual/data/bc5cdr.py](src/multilingual/data/bc5cdr.py) — direct JSONL fetch
- [src/multilingual/data/masakhaner.py](src/multilingual/data/masakhaner.py) — GitHub raw fetch
- [src/multilingual/data/corpora_swh.py](src/multilingual/data/corpora_swh.py) — replaced CC-100/WURA/Masakhane with parquet alts
- [src/multilingual/data/corpora_bio.py](src/multilingual/data/corpora_bio.py) — switched to `slinusc/PubMedAbstractsSubset`
- [src/multilingual/data/openwho.py](src/multilingual/data/openwho.py) — schema column-name probe
- [src/multilingual/data/lid.py](src/multilingual/data/lid.py) — NumPy 2 patch, sentence-split, threshold 0.5
- [src/multilingual/adapter_setup.py](src/multilingual/adapter_setup.py) — modern `adapters` API names (`SeqBnConfig`/`DoubleSeqBnConfig`/`ParBnConfig`)
- [src/multilingual/compose.py](src/multilingual/compose.py) — `_harmonize_dtype` helper
- [src/multilingual/training/task_trainer.py](src/multilingual/training/task_trainer.py) — `save_strategy="steps"` to match eval strategy
- [src/multilingual/utils/env.py](src/multilingual/utils/env.py) — new file; .env loader
- [src/multilingual/__init__.py](src/multilingual/__init__.py) — calls `_load_dotenv()` on import
- All 14 entry-point [scripts/](scripts/) — `config_name="default"` for unified Hydra structure
- [scripts/prepare_data.py](scripts/prepare_data.py) — `lid_threshold` read from config
- [Makefile](Makefile) — `+source=`, `+mode=` for Hydra struct mode
- [configs/data/la_swh.yaml](configs/data/la_swh.yaml) — added `train_data_path`, lowered `lid_threshold`

---

## 10. Useful one-liners

```bash
# What's running right now?
ps -ef | grep -E 'train_(la|da|ta)|prepare_data|eval_' | grep -v grep

# Latest run dir
ls -td results/runs/*/ | head -1

# Latest eval metrics (excluding the per-example dump)
LATEST=$(ls -td results/runs/*/ | head -1)
cat $LATEST/eval_*/metrics.json 2>/dev/null | head -40

# Tail any background log
tail -f /tmp/la_full.log
tail -f /tmp/da_full.log
tail -f /tmp/ta_full.log

# GPU is alive?
nvidia-smi | head -16
```

---

## 11. Crash-resilient training (added 2026-05-15)

### What happened

The workstation hard-rebooted twice during LA training:

| # | Training start | Last sysstat | Survived |
|---|---|---|---|
| 1 | 2026-05-12 13:04 | 14:10:06 (sa12) | 66 min |
| 2 | 2026-05-14 00:52 | 01:00:00 (sa14, restart at 01:16:41) | ~24 min |

Both events have the textbook signature of **abrupt hardware power-off**: no
kernel panic, no shutdown markers in `journalctl`, `wtmp` marks the session as
`crash`, sysstat's daily binary stops being written mid-cycle. The OS was idle
(CPU 4–6 %, mem 45 %) at the time of death — this rules out OOM and points at
either a workstation power-cap trip or a firmware-level thermal shutoff that
bypasses Linux entirely. Cause is hardware-level; cannot be fixed in software.

Both crashes happened before the first scheduled save at step 5000 (config had
`save_steps=10_000`), so all training progress was lost both times.

### The fix (software-side mitigations)

Three changes in the training stack so progress survives across power events:

1. **Frequent checkpoints**: dropped `save_steps` from 10 000 → 1 000 in
   [configs/train/la_mlm.yaml](configs/train/la_mlm.yaml) (and `da_mlm.yaml`).
   Each crash now costs at most ~17 min, not ~17 h.
2. **Stable output dir** for the trainer:
   - LA writes to `${ckpt_dir}/la_swh_run/` (no timestamp).
   - DA writes to `${ckpt_dir}/da_eng_bio_<mode>_run/` (no timestamp).
   - The Hydra `results/runs/<ts>/` dir is now used **only** for the artifact
     dump (config/metrics snapshot per launch); training itself doesn't write
     checkpoints there. This is what lets resume work — the stable dir is the
     same across launches.
3. **Auto-resume**: trainers now pass `resume_from_checkpoint=<latest_ckpt>` to
   `Trainer.train(...)` when a `checkpoint-N/` directory exists. First launch
   starts fresh; later launches pick up from the latest saved step.
   - Plumbed through
     [src/multilingual/training/mlm_trainer.py](src/multilingual/training/mlm_trainer.py)
     `build_mlm_run(..., resume_from_checkpoint=...)`.
   - Resume guard in
     [scripts/train_la.py](scripts/train_la.py) /
     [scripts/train_da.py](scripts/train_da.py): only sets `resume=<path>` if
     a `checkpoint-*` dir is actually present, so the first launch doesn't
     fail with HF's "no checkpoint found" error.
   - **Adapters-lib resume gotcha** (verified 2026-05-16): on resume, the
     `adapters` library calls `load_adapter` which **replaces** the adapter's
     `nn.Module` objects with fresh ones. The new parameter IDs no longer
     match what's in the saved `optimizer.pt`, so HF Trainer raises
     `ValueError: loaded state dict contains a parameter group that doesn't
     match the size of optimizer's group` and the watchdog burns through
     all 20 restart attempts. **Fix**: train_la.py / train_da.py now delete
     `optimizer.pt` and `scheduler.pt` from the resume checkpoint right
     before calling `train()`. HF Trainer silently skips optimizer-state
     restoration when the file is missing; it still restores the step
     counter and global state from `trainer_state.json`. Verified resume
     works: checkpoint-50 → step 60 → step 80, loss trajectory continued
     cleanly. The only cost is losing Adam momentum, which warms back in
     a few hundred steps.

### Supervisor

[scripts/run_with_watchdog.py](scripts/run_with_watchdog.py) — small Python
supervisor (~150 lines) that:

- Launches the underlying trainer as a subprocess (`la`, `da`, or `ta`).
- Restarts the child on any non-zero exit (e.g. after a reboot the user
  reruns the watchdog manually; the *intra-session* relaunches handle
  software-level crashes).
- Concurrently samples `nvidia-smi` temperature + power + utilization +
  memory every 5 s and writes to `/tmp/watchdog/<target>_gpu.csv`. This is
  the **only** thermal forensics we can collect — the kernel dies too fast
  to log anything itself.
- Uses `os.setsid` so Ctrl-C cleanly kills the whole training subtree.

### Production usage (LA training, post-mitigation)

```bash
# Foreground (visible logs):
env $EXPORTS python scripts/run_with_watchdog.py --target la

# Background, terminal-detached, survives logout:
nohup env $EXPORTS python scripts/run_with_watchdog.py --target la \
    > /tmp/watchdog_la.log 2>&1 &

# Override anything via Hydra (after `--`):
env $EXPORTS python scripts/run_with_watchdog.py --target la \
    -- train.batch_size=8 train.grad_accum=8 train.warmup_steps=500
```

After **every reboot**, just re-run the same watchdog command. The trainer
will see the latest `${ckpt_dir}/la_swh_run/checkpoint-N/` and resume from
step N+1. The watchdog process itself is bound to the user session (it does
not survive reboots; it would need to be a systemd user service to do that —
out of scope).

### Verification (smoke + first prod run)

```bash
# 1. Fresh tiny smoke (50 steps, save every 20):
env $EXPORTS python scripts/run_with_watchdog.py --target la \
    -- train.max_steps=50 train.save_steps=20 train.eval_steps=20 \
       train.warmup_steps=5

# Expect: checkpoint-20/ and checkpoint-40/ appear in checkpoints/la_swh_run/.

# 2. Simulate a crash and confirm resume:
env $EXPORTS python scripts/run_with_watchdog.py --target la --max-restarts 3 \
    -- train.max_steps=100 train.save_steps=20 train.eval_steps=20 \
       train.warmup_steps=5 &
sleep 90 && pkill -f train_la.py
# Watchdog logs the restart; next launch picks up at step 20, ends at 100.

# 3. GPU forensics after any session:
tail -20 /tmp/watchdog/la_gpu.csv
awk -F, 'NR==1 || $2+0>80 {print}' /tmp/watchdog/la_gpu.csv
#         ^ rows where temperature.gpu > 80 °C
```

### Hardware-side ideas (not implemented, but worth trying if crashes persist)

- **GPU power cap**: `sudo nvidia-smi -pl 180` (or lower) to bound continuous
  draw. Run `nvidia-smi -q -d POWER | grep 'Power Limit'` to see what's
  permitted on this SKU. Lower power cap trades training speed for stability.
- **Dedicated 20 A circuit** for the workstation (vs. sharing with other
  appliances). The DGX Spark spec sheet calls for ≥1500 W headroom.
- **Decent ventilation** — keep ambient temp <25 °C and ensure the
  workstation has open clearance for intake/exhaust.
- **UPS / line-conditioning**: a small UPS will catch brown-outs that
  trip an unprotected PSU.

If thermals are the cause, the GPU CSV from `run_with_watchdog.py` will
show temperature climbing into the 80s before each crash — that's the
post-mortem signal to look for after the next event.
