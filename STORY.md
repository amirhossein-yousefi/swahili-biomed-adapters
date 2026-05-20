# The Story So Far

*Started 2026-05-10. Last updated 2026-05-17.*

A narrative companion to the other docs. For installation: see
[`README.md`](README.md). For the chronological run log and raw numbers:
[`JOURNAL.md`](JOURNAL.md). For the full literature review and 6–10 week
project plan: [`compass_artifact_wf-…md`](compass_artifact_wf-50ffc1c7-6245-4b10-9964-ecf6331d4d25_text_markdown.md).
For per-source licenses: [`NOTICES.md`](NOTICES.md).

---

## TL;DR

We are building an openly-licensed Swahili medical NLP pipeline by composing
three small adapter modules on top of a frozen multilingual encoder
(AfroXLMR-large), following the MAD-X recipe. We do this because Swahili —
~80 million speakers, the East African lingua franca — has no openly
licensed clinical benchmark and no biomedical pretraining corpus at scale,
while frontier LLMs that do speak Swahili are closed, contamination-prone,
and impossible to deploy offline. After a week of running this end-to-end we
have landed a first measured baseline: 12.7 % on the 331-question
MMLU-ProX-Swahili clinical split — essentially random, which is informative
because it tells us the bottleneck is the task adapter, not the language
adapter, and points cleanly at the next experiment (proper MAD-X
TA-on-LA_en training). The full numbers table lives in `JOURNAL.md` §8.

---

## 1. The problem

Swahili is widely spoken across East Africa and is a common medium of
healthcare conversation throughout Tanzania, Kenya, Uganda, and beyond, yet
the language is essentially absent from biomedical NLP research. Every
established benchmark in the field is English: BC5CDR for disease NER,
MedMCQA and MedQA for medical question answering, MIMIC for clinical notes.
AfriMed-QA — the most relevant pan-African resource published in 2024 — is
also English-only, by construction; its authors call out non-English
extension as future work. The single Swahili-language medical resource we
could find as of mid-2026 is the Swahili split of MMLU-ProX, which after
filtering to the four clinical subjects the original project plan calls out
(Clinical Knowledge, College Medicine, Professional Medicine, Virology)
leaves 331 multiple-choice questions. That is the entire openly-licensed
Swahili clinical evaluation set in existence.

The pretraining side is no better. PubMed and PMC Open Access are vast
(~5 B and ~12 B tokens respectively) but English. MIMIC has clinical notes
but is paywalled by PhysioNet credentialing and cannot be redistributed,
which makes any model trained on it impossible to release. Swahili-side raw
corpora exist — mC4, WURA, Wikipedia — but none of them is biomedical, so
we have no in-domain Swahili text to pretrain on. The full landscape and
gap analysis live in `compass_artifact_*.md` §1 and §4.

You can of course ask Gemini or GPT-4 questions in Swahili and they will
answer competently. But those models are closed, expensive at scale,
contamination-prone with respect to public benchmarks, impossible to audit
or deploy in a Tanzanian hospital with intermittent connectivity, and not
suitable as research artifacts. We want an open, reproducible,
offline-deployable pipeline. That is the gap.

---

## 2. The approach

The technical bet is **modular adapters**, specifically the MAD-X recipe
(Pfeiffer, Vulić, Gurevych, Ruder — EMNLP 2020). Rather than fine-tune
hundreds of millions of base parameters on every task × language
combination, we freeze a single multilingual encoder (AfroXLMR-large,
560 M parameters) and attach small bottleneck adapters at each transformer
layer. Each adapter is ~7 M trainable parameters, roughly 1 % of the base.
Three of them, composed at inference, do the heavy lifting.

A **language adapter** (LA_swh) is trained via masked-language-modelling
on raw Swahili text — mC4, WURA, Wikipedia — so it learns to project
Swahili morphology and vocabulary into the same representational space the
frozen base uses for the high-resource languages it was originally trained
on. A **domain adapter** (DA_eng) is trained similarly via MLM, but on
English biomedical text — PubMed abstracts and PMC OA — so it specializes
the representations toward medical vocabulary and discourse. A **task
adapter** (TA) is trained supervised on the actual downstream task in
English (MedMCQA for medical question answering, BC5CDR for disease NER),
on top of the LA and DA which are kept frozen.

At inference, the three are stacked. The headline composition is
**C4 = Stack(LA_swh, DA_eng, TA)**: a Swahili medical question flows up
through the Swahili language adapter, then through the biomedical domain
adapter, then through the task adapter, all of which were trained
independently and now compose by virtue of all sitting on top of the same
frozen encoder. The plan also exposes seven other compositions
(C1 base-only, C2 LA-only, C3 DA-only, C5 reversed order, C6
AdapterFusion, C7 He-et-al parallel, C8 BAD-X-style joint, C9 TIES weight
merge), so we can attribute exactly which component is doing the lifting.

The cost story is what makes this attractive. Training a 7 M adapter on
1.5 B tokens of Swahili is roughly 80× cheaper than full continued
pretraining of the 560 M base on the same corpus, the base weights are
preserved exactly so there is no risk of catastrophic forgetting, and the
modular structure means a single language adapter is reusable across
arbitrarily many domains and tasks. We get N + M adapters instead of N × M.

---

## 3. What we built

The codebase scaffolds out the full MAD-X pipeline as a Hydra-configurable
research repo. Backbone, adapter shape, data sources, training
hyperparameters, composition strategy, and evaluation set are all driven by
YAML configs so the §5.5 composition matrix and the §5.7 ablation grid in
the project plan are expressible as `python script.py -m
adapter=pfeiffer_rf{4,8,16,32,64}` multiruns rather than code branches.

The data layer covers seven sources working end-to-end and one stub, with
the full status table in `JOURNAL.md` §2: MMLU-ProX Swahili clinical (331
questions, our primary eval), BC5CDR-Disease English plus an NLLB-translated
and span-projected Swahili test split (5,865 sentences), MedMCQA (182 K
English questions, our TA training set), MasakhaNER Swahili (the §8
forgetting probe), OpenWHO Swahili (held-out MLM perplexity), and Swahili
raw corpora from mC4, WURA, and Wikipedia (2 M sentences after
sentence-splitting, dedup, and language-ID filtering). Half of those
sources required workarounds when `datasets >= 4.0` dropped support for
loading scripts and broke the canonical HF repos — see `JOURNAL.md` §2 for
the per-source fix list.

The HuggingFace `adapters` library (Poth et al. 2023) handles the
primitives natively: Pfeiffer bottleneck placement, Houlsby placement,
invertible NICE coupling on the embedding layer, parallel and stacked
composition, AdapterFusion, LoRA. What we wrote is the orchestration around
it: a backbone loader that freezes the base and sets up dtype harmonization,
a single `compose.build_composition(model, "Cn", paths)` API that
implements all ten composition strategies as config-string dispatches, a
shared MLM trainer for LA and DA, a task trainer for MCQA / NER /
classification, and a watchdog supervisor for crash-resilient long runs.

Hardware: a single NVIDIA DGX Spark workstation (GB10 Grace Blackwell SoC,
128 GB unified LPDDR5x). All training so far has been BF16 on the
integrated GPU with SDPA attention (FlashAttention-2 not installed). The
pipeline does fit; the constraint has been the workstation's sustained
power envelope, more on that below.

---

## 4. What we ran, 2026-05-10 through 2026-05-17

The first week was pipeline shakedown. A 2,000-step LA smoke run on
2 million Swahili sentences from the disk-cached corpus dropped MLM
held-out loss by 54 % in 33 minutes — well above the §9 non-negotiable
"10 % drop minimum" sanity check, confirming that the adapter
infrastructure was correctly wired end-to-end (frozen base, trainable
adapter + invertible head, BF16, AdapterTrainer). Then a 572-step task
adapter smoke on English MedMCQA, composed at inference with the smoke LA,
landed C2 at 12.4 % accuracy on the 331-question Swahili clinical eval.
That confirmed the composition + eval path worked: a number came out, the
artifact dump landed, the dtype-harmonization fix that strips FP32 adapter
weights down to BF16 fired correctly.

The second week added scale on the task adapter. A full 57,000-step run
over five epochs of MedMCQA at BS=8, GA=2 took 11 hours. Final MedMCQA-val
eval loss was 1.345 — just below `log(4) = 1.386`, which means the
multiple-choice head learned to discount the six empty padded options
(MMLU-ProX is up to 10-option, MedMCQA is 4, so we train with
num_choices=10 and pad) but barely distinguishes among the four real
options. That capacity ceiling is expected: pure TA with no language or
domain adapter underneath has nothing English-specific to lean on, and the
MedMCQA → MMLU-ProX-Swahili-clinical transfer is genuinely hard. C2 with
this full TA landed at 11.2 %.

The third week brought the operational lesson. The workstation crashed
twice during a planned multi-day LA training run, both with the unmistakable
signature of an abrupt firmware-level power-off: no kernel logs, no
shutdown markers, `sysstat` truncated mid-cycle, `wtmp` flagging the user
session as a "crash". The OS was idle (CPU 6 %, mem 45 %) at the time of
both deaths — this is not OOM, not a software fault. It looks like either
the workstation hit its sustained-power envelope and the PSU tripped, or
the SoC firmware issued a hard thermal-protection power-off that bypasses
Linux. We could not pin the exact cause from logs alone.

The software response was to make training survive these events:
checkpoints every 1,000 steps instead of every 10,000, a stable
non-timestamped output directory so resume picks up the latest checkpoint
across reboots, auto-resume with `optimizer.pt` and `scheduler.pt`
explicitly stripped before each restart to dodge a parameter-group
mismatch the `adapters` library introduces when it replaces module objects
on `load_adapter`. The watchdog supervisor at
`scripts/run_with_watchdog.py` relaunches the trainer on any non-zero
exit, caps total restarts, and concurrently writes a 5-second-cadence
nvidia-smi CSV so we have GPU thermal forensics — the only data the kernel
won't preserve through these crashes. `JOURNAL.md` §11 documents the
mechanism in full.

With that infrastructure in place, the **full LA training ran for 25
hours 12 minutes uninterrupted**, hitting step 95,733 (three full epochs)
with zero watchdog restarts needed. Peak GPU temperature across the run
was 87 °C, sustained mean 77 °C — exactly the thermal envelope where the
firmware shutoffs likely originate, but we didn't trip it this time.

Re-evaluating C2 with this fully-trained LA produced the latest headline
number: **12.69 % on MMLU-ProX-Swahili clinical, n=331** (Clinical
Knowledge 9.7 %, College Medicine 10.4 %, Professional Medicine 13.9 %,
Virology 15.2 %). That is only +1.5 percentage points over the same eval
with the 2K-step smoke LA, despite 48× more LA training. The conclusion
the JOURNAL §7.1 prediction called out a week earlier is now confirmed
empirically: **the bottleneck is the task adapter, not the language
adapter**. The TA was trained on raw English MedMCQA with no language
adapter underneath, so it learned representations conditioned on an
English-baseline view of the base model; at inference under LA_swh, the
representations shift to Swahili-shaped and the TA cannot use them.

---

## 5. What we learned

The most important lesson is that **the MAD-X recipe is not optional**.
The original Pfeiffer et al. 2020 paper trains the task adapter on top of
a source-language adapter (LA_en) and then swaps LA_en for the
target-language adapter (LA_swh) at inference time — that swap is the
mechanism by which the TA's task knowledge cross-lingually transfers. Our
first attempt skipped the LA_en step and trained the TA on the raw frozen
base, on the theory that "the base already understands English well
enough". The empirical answer to that theory is the 12.69 % C2 number
above. `compass_artifact_*.md` §5.4 was right and we re-learned it the
expensive way.

A secondary lesson, more operational: **sustained-power thermals are the
binding constraint on workstation-class hardware**. Three days of training
on the GB10 brought the SoC to its sustained-load envelope twice. The
software-side fix (frequent checkpoints + resume + watchdog) means a
power-off costs ~17 minutes of progress instead of 17 hours, which is a
huge improvement, but the underlying problem is hardware. The right
medium-term fix is a GPU power cap via `nvidia-smi -pl`, a UPS to catch
brown-outs, better ambient cooling, and a dedicated electrical circuit. The
GPU CSV emitted by the watchdog will tell us — if any future crash is
preceded by temperatures climbing into the 90s, thermals are the cause; if
temperatures look normal and the box dies anyway, it's the AC side.

There were two ML-infrastructure lessons that probably generalize. First,
`datasets >= 4.0` dropped support for loading scripts, which broke roughly
half the dataset repos our project plan named: CC-100, WURA (the canonical
`castorini/wura` repo, not the parquet `llama-lang-adapt/wura`),
BC5CDR (`tner/bc5cdr`), MasakhaNER, NCBI PubMed. The recovery in each case
was either to find a parquet-only alternative or to fetch the raw data
files directly via `hf_hub_download`/`requests` and parse them locally.
The full per-source fix list is in `JOURNAL.md` §2; the pattern is
generalizable to any low-resource HF dataset that hasn't been
parquet-migrated.

Second, the FastText `lid.176` language-ID model that the project plan
specifies "at p > 0.9" is empirically calibrated very low on Swahili.
Even on the cleanest source we have (WURA, already filtered by Common
Crawl quality classifiers), fewer than 2 % of true-Swahili sentences clear
that confidence threshold. After sentence-splitting before the LID stage
and lowering the threshold to 0.5, retention rose to ~85 % of real
Swahili while still rejecting the English / Esperanto / Indonesian false
positives that contaminate mC4-Swahili. The threshold-of-record in
`configs/data/la_swh.yaml` is now 0.5, with a comment explaining why.

---

## 6. Where we are now, and the next experiment

The number to beat is **C2 = 12.69 %** on MMLU-ProX-Swahili clinical
(2026-05-17). Random for that question mix is roughly 12.5 % (a weighted
average over the 4–10 option distribution), so we are essentially at
chance. The full landed-numbers table lives in `JOURNAL.md` §8 and is
updated after every evaluation.

The next experiment is the canonical MAD-X recipe, labelled **Path A** in
`JOURNAL.md` §7.3. It has three steps. First, train a small English
language adapter LA_en via MLM on English Wikipedia — a few hours of
training, since English is easy and the LA just needs to anchor the
English-baseline representation. Second, retrain the task adapter on top
of [LA_en → frozen base], which is the configuration MAD-X actually
prescribes — the TA learns "task given English-language baseline" rather
than "task given raw base". Third, re-evaluate C2 with the new TA composed
on top of LA_swh, and check whether the cross-lingual transfer that
LA_en → LA_swh swap unlocks moves the number above random. If it does,
the next experiment is **Path B**: build a domain adapter on PubMed
abstracts and run the headline composition C4 = Stack(LA_swh, DA_eng,
TA). If it doesn't, we have a publishable negative result that echoes
Stickland et al. 2021's finding on NMT — naive language × domain adapter
composition does not always work — but now in the encoder regime and on a
low-resource African language. Both outcomes are worth the run.

After C4 there is a long roadmap of ablations (composition order,
bottleneck dimension, invertible adapter on/off, three random seeds,
AdapterFusion vs Parallel composition), an NER track using the
NLLB-projected Swahili BC5CDR, the §8 catastrophic-forgetting probe on
MasakhaNER, and clinician validation of a stratified ~500-question
subset of the synthetic Swa-MedBench (the rubric is at
`benchmark/clinician_validation/rubric.md`). All of that is in
`JOURNAL.md` §7.4–§7.7.

---

## 7. The bigger picture

If the empirical results land — even partially — the project produces two
durable artifacts. The first is **Swa-MedBench v0.1**, the first openly
licensed Swahili clinical NLP evaluation suite, assembled from
MMLU-ProX-Swahili, NLLB-translated MedMCQA and BC5CDR, MedlinePlus Swahili
scrapes, and OpenWHO Swahili held-out. Even if every adapter result we
land is below frontier LLMs, the benchmark itself is useful to every
subsequent group working on Swahili medical NLP. The second artifact is a
reproducible adapter-composition recipe and full open codebase: backbone,
adapters, configs, training scripts, evaluation harness, and watchdog
supervisor, all under permissive licenses (modulo the CC-BY-NC-SA on
OpenWHO and NLLB which makes any derived benchmark non-commercial — see
`NOTICES.md`).

If we further land a positive C4 result, that becomes a clean empirical
contribution to the modular-deep-learning literature: the first
systematic study of language × biomedical-domain adapter composition on a
low-resource African language. If C4 lands negative, that is also a
contribution — Stickland et al. 2021 reported catastrophic-forgetting on
NMT seq2seq; we would be reporting it (or refuting it) on encoder NLU,
which is a different regime. Either outcome is the kind of clean,
reproducible, hardware-modest empirical study that earns a place on
arXiv and reads well in a portfolio. The framing argument lives in
`compass_artifact_*.md` §10.

That is the story so far.
