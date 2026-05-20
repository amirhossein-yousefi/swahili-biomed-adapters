# Project Plan: Adapter Stacking for Cross-Lingual Domain Adaptation in Swahili Biomedical NLP

## TL;DR
- **The opportunity is real and the gap is genuine.** As of late 2025/early 2026, there is no published systematic study of MAD-X-style language × *domain* (rather than language × *task*) adapter composition for Swahili biomedical NLP; AfriMed-QA exists but is English-only, OpenWHO provides health parallel text including Swahili, MasakhaNER covers general Swahili NER, and there is no Swahili clinical NER, ICD coding, or medical QA benchmark in Swahili. The strongest novelty pillar is therefore **(a) a careful empirical study + (d) a synthetic Swahili biomedical benchmark constructed from OpenWHO/MedlinePlus/AfriMed-QA translations**, with a secondary contribution from a new composition method (e.g., a learned routing or post-hoc language-arithmetic adjustment between LA and DA).
- **The recommended setup fits a single DGX Spark cleanly.** AfroXLMR-large (≈560M params) as the backbone, Pfeiffer bottleneck adapters (reduction factor 16, ~7M params per adapter) with invertible adapters on the embedding layer for the Swahili LA, MLM-trained on 1–3B Swahili tokens (CC-100 + MC4 + WURA + Masakhane corpora) and 1–2B English biomedical tokens (PubMed abstracts + PMC OA), composed by sequential stacking [LA → DA → TA] at inference. Wall-clock on DGX Spark for each adapter MLM run is ~1–3 days at FP16/BF16; full pipeline including five baselines, three seeds, and ablations is ~6–10 weeks part-time.
- **The publishable story is "the first systematic study of language × domain adapter composition for low-resource clinical NLP, with a reproducible Swahili medical benchmark suite."** Differentiation from MAD-X (task vs. domain), BAD-X (bilingual LA), Hyper-X (hypernetwork), AdaMergeX (LoRA arithmetic for cross-lingual *task* transfer), Stickland et al. 2021 (NMT-only language × domain), and UDApter (monolingual UDA) is clean. The compute budget is realistic; the main scientific risks are eval set noise (mitigated by translation QA + clinician spot-checks) and catastrophic forgetting of Swahili competence under the English domain adapter (mitigated by Pfeiffer-style residuals, AdapterFusion, and an explicit "language-readjustment" probe on MasakhaNER/SIB-200).

---

## Key Findings

1. **MAD-X is the right backbone framework, but it has never been systematically extended to language × biomedical-domain composition.** The original MAD-X (Pfeiffer et al., EMNLP 2020) composes language adapters (LA, MLM-trained on raw target text) with task adapters (TA, supervised on labeled source-language data) and adds invertible embedding adapters that learn a token-level injection/inversion function to bridge tokenizer mismatches. The closest published work to your proposed setting is Stickland et al. 2021 (NMT, decoder-only domain adapters with backtranslation) and the line on "Don't Stop Pretraining" (Gururangan et al. 2020) which establishes DAPT/TAPT but in a monolingual setting. Recent 2024–2026 work (AdaMergeX, Language Arithmetic, FLARE, target-language-ready task adapters by Parović et al. 2023) operates at the language × task axis, not language × domain.

2. **The Swahili biomedical evaluation gap is the dominant project risk and the dominant novelty lever.** AfriMed-QA (Olatunji et al. 2024) is English-only across 16 African countries — Swahili NLP can borrow it for cross-lingual training but not as a Swahili eval. MMLU-ProX, Global MMLU, and the Bridging-the-Gap benchmark have Swahili splits including College Medicine / Clinical Knowledge / Virology subsets — these are usable as Swahili medical MCQA evaluations. OpenWHO (Merx et al., WMT 2025) provides 26,824 expert-translated parallel health sentences in 20 languages and explicitly covers Swahili (translated WHO e-learning courses, including a Swahili Ebola/COVID course confirmed by WHO Tanzania documentation). MedlinePlus has hundreds of Swahili patient-information PDFs. There is no Swahili clinical NER dataset published as of early 2026.

3. **DGX Spark (GB10 Grace Blackwell, 128GB unified LPDDR5x, ~273 GB/s memory bandwidth, 1 PFLOP sparse FP4) is well-suited for encoder-MLM adapter training in this regime, but bandwidth-bound for 70B+ inference.** NVIDIA's published training benchmarks: Llama-3.2-3B full FT at 82,739 tok/s; Llama-3.1-8B LoRA at 53,657 tok/s; Llama-3.3-70B QLoRA at 5,079 tok/s. For a 560M-param XLM-R-large with frozen base + ~7M adapter params (reduction factor 16), expected training throughput will exceed the 8B LoRA number; an MLM pass over ~1B Swahili tokens at sequence length 512 fits comfortably in <48 hours wall-clock.

---

## Details

### 1. Literature review — adapter methods and cross-lingual domain adaptation

**1.1 MAD-X (Pfeiffer, Vulić, Gurevych, Ruder, EMNLP 2020).** Builds on Houlsby/Pfeiffer adapters (Houlsby et al. ICML 2019; Pfeiffer et al., AdapterFusion 2021). The architecture freezes the multilingual transformer (mBERT or XLM-R) and inserts three modular adapter types: (a) **language adapters (LA)** trained with MLM on raw target-language text (Wikipedia/CC-100), one per language; (b) **task adapters (TA)** trained on supervised data in a *source* language, stacked on top of the source LA during training; (c) **invertible adapters (Inv)**, a NICE/GLOW-style coupling-layer transformation applied to the input embeddings (and inverted at the output projection to share parameters with the masked-LM head). At inference, the source LA is *replaced* by the target LA while the same TA remains stacked, enabling zero-shot cross-lingual transfer. The bottleneck adapter follows the **Pfeiffer placement** (single down-up bottleneck after the FFN sub-layer; reduction factor 16, GeLU non-linearity, residual connection around the bottleneck plus the original FFN residual). On NER (MasakhaNER predecessor / WikiAnn) and XCOPA, MAD-X improves over zero-shot XLM-R particularly on languages outside the base model's pretraining set. Code/checkpoints are integrated into AdapterHub (now `adapters` library, Poth et al. 2023).

**1.2 MAD-G (Ansell, Ponti, Pfeiffer, Ruder, Glavaš, Vulić, Korhonen — Findings of EMNLP 2021).** Replaces individual LA training with **Contextual Parameter Generation (CPG)**: a hypernetwork takes URIEL typological vectors (~289 typological features) of a target language as input and emits the LA's bottleneck weights θ. Reduces fine-tuning cost ~50× per language, supports zero-shot generation of LAs for languages without monolingual data. Particularly strong on low-resource African POS/NER. For Swahili — which is well-resourced enough to have its own LA — MAD-G is mainly a baseline / efficiency comparison, not the primary method. However, MAD-G suggests an interesting ablation: can Swahili MAD-G LA (typology-conditioned) match a directly trained Swahili LA on biomedical tasks?

**1.3 BAD-X (Parović, Glavaš, Vulić, Korhonen — NAACL 2022).** Replaces two monolingual LAs with a single bilingual LA trained jointly on both source (English) and target (Swahili) raw text. Improves zero-shot transfer at the cost of modularity (one bilingual adapter per source-target pair). The natural extension for your project is a **trilingual or "bilingual+domain" BAD-X**: train a single adapter on a balanced mixture of Swahili general + English biomedical raw text and compare to sequential LA-then-DA stacking. Parović et al. 2023 ("Cross-Lingual Transfer with Target-Language-Ready Task Adapters") and WAD-X (Wang et al., TALLIP 2023) are further refinements that add word-alignment objectives.

**1.4 Hyper-X (Üstün, Bisazza, Bouma, van Noord, Ruder — EMNLP 2022).** A single hypernetwork conditioned on (language embedding, task embedding, layer ID) generates adapter weights for any (language, task) pair, including unseen combinations. Closest competitor for the "(language, domain)" generalization you want: a Hyper-X variant that conditions on (language ID, domain ID) is a viable extension.

**1.5 LoRA, AdapterFusion, (IA)³, prefix tuning, Compacter — comparison.**
| Method | Trainable params | Composition story | Reported strength |
|---|---|---|---|
| Houlsby adapter (ICML 2019) | ~3.6% | Sequential stacking | Original; 2× params of Pfeiffer |
| Pfeiffer adapter (EMNLP 2020) | ~0.9% (rf=16) | Sequential, MAD-X compatible | Best fit for MAD-X recipe |
| AdapterFusion (Pfeiffer et al., EACL 2021) | adapter + ~1% fusion | Attention over multiple adapters | Non-destructive multi-adapter composition |
| LoRA (Hu et al., ICLR 2022) | 0.1–0.3% | Additive in weight space; merges into base | Strong on FT; weaker per "Comprehensive Analysis of Adapter Efficiency" (Bansal et al.) on cross-lingual tasks |
| Compacter (Mahabadi et al., NeurIPS 2021) | ~0.05% (Kronecker × rank-1) | Sequential | Extremely small but underperforms Pfeiffer on cross-lingual |
| (IA)³ (Liu et al., 2022) | ~0.01% (rescale vectors) | Multiplicative on K, V, FFN activations | Lowest latency; best when memory is the binding constraint |
| Prefix tuning (Li & Liang, 2021) | ~0.1% (virtual tokens) | Prepended to keys/values | Strong on generation; weaker on encoder-only |

For an arXiv paper that wants to maximize cross-lingual transfer credibility, **Pfeiffer bottleneck + invertible adapters is the right primary configuration**, with LoRA as a baseline for a fair PEFT-method ablation. The "Comprehensive Analysis of Adapter Efficiency" survey (Open Review submission) explicitly recommends Houlsby/Pfeiffer over LoRA/Compacter for cross-lingual tasks.

**1.6 Recent 2023–2026 work on adapter composition, stacking, and routing.**
- **Modular Deep Learning** (Pfeiffer, Ruder, Vulić, Ponti, TMLR 2023, arXiv 2302.11529) is the canonical survey to anchor your framing; it formalizes module/routing/aggregation/training as the four design dimensions.
- **AdaMergeX** (Zhao et al., NAACL 2024 / arXiv 2402.18913) — adaptive merging of language and task LoRA adapters via vector arithmetic; key insight is that merging method must match adapter type (LoRA-style merging fails for (IA)³ and vice versa).
- **Language Arithmetic / "No Train but Gain"** (arXiv 2404.15737) — task-arithmetic-style adjustment of MAD-X language adapters at inference, training-free.
- **Language and Task Arithmetic with PEFT Layers** (Klimaszewski et al., MRL 2024) — composing LoRA task and language adapters in PaLM-2 by addition/subtraction.
- **Target-language-ready Task Adapters** (Parović, Ansell, Vulić, Korhonen — arXiv 2306.02767) — TA fine-tuned with target-language exposure to remove TA/LA inference-time mismatch.
- **FLARE** (Borchert et al. 2025) — fusion at output level rather than parameter level.
- **The Impact of Language Adapters in Cross-Lingual Transfer for NLU** (Kunz & Holmström, MOOMIN 2024) — *important caveat*: shows that target-language adapters have inconsistent effects on NLU tasks; sometimes retaining the source LA at inference outperforms swapping in the target LA. This is a result you must replicate as a sanity check and discuss explicitly in the paper.
- **FUN with Fisher** (Foroutan et al. 2023) — scheduled unfreezing of language adapters during task FT to improve cross-lingual generalization.
- **Cascading Adaptors** (arXiv 2112.09866) — stacked LA+TA for QA across low-resource languages; closest existing pipeline to yours but on QA, not biomedical.
- **Stickland et al. 2021** (arXiv 2110.09574, "Multilingual Domain Adaptation for NMT: Decoupling Language and Domain Information with Adapters") — the *only* prior work that explicitly composes language × domain adapters; finds catastrophic forgetting in naive composition and uses encoder-only/decoder-only domain adapters + back-translation. This is your single most important differentiation point: they work on NMT seq2seq, you work on encoder-only NLU/clinical-extraction tasks; their finding that "naive combination of domain-specific and language-specific adapters often results in catastrophic forgetting" is a direct hypothesis to test in your domain.
- **UDApter** (Malik, Kashyap, Kan, Poria — EACL 2023) — UDA decomposed into domain adapter + task adapter; monolingual; shows invertible bottleneck adapters are particularly effective for MLM during UDA. Confirms the design choice to keep invertible adapters on the LA, but tests on MNLI, not biomedicine.

**1.7 Why adapter stacking can outperform full continued pretraining in low-resource settings.** Three theoretical reasons: (a) **catastrophic forgetting** — full DAPT on a small Swahili-medical corpus risks erasing English world knowledge; adapters keep base parameters frozen so that knowledge is preserved exactly. (b) **modularity / compositionality** — once an LA exists, it is reusable across all biomedical tasks (NER, classification, QA) and vice versa; a single backbone serves N×M (language, domain) pairs at O(N+M) parameter cost rather than O(NM). (c) **regularization through the bottleneck** — the bottleneck dimension acts as an inductive bias preventing overfitting on small data; this matches the empirical finding (Houlsby 2019, He et al. 2021) that adapters generalize better than full FT in low-data regimes. (d) **compute** — training a 7M-param adapter on 1B tokens is ~80× cheaper than full FT of XLM-R-large on the same data. The trade-off (Hu et al. ICLR 2022 LoRA paper, Bansal et al. analysis) is a small per-task accuracy gap that vanishes at moderate adapter capacity.

### 2. Swahili NLP resources and benchmarks

**2.1 Raw text corpora.** CC-100 Swahili (~275M tokens, Conneau et al. 2019); OSCAR Swahili (multiple snapshots, ~250–400M tokens depending on filtering); mC4 Swahili (~3B tokens, Xue et al. 2020); WURA (Oladipo et al. 2023) — cleaned African-language portion of mC4 used in AfriTeVa V2; Helsinki-NLP Swahili corpus / OPUS subsets including JW300, Tatoeba, Tanzil; Masakhane raw corpora; the Mendeley Swahili corpus by Shikali et al. 2024 (PMC11372376, "In the heart of Swahili") with 1.69M sentences / 39.6M words / 168MB across 12 categories including a **Health (AFYA) subset** — a non-trivial Swahili health-specific raw text starter. Swahili Wikipedia (~70M tokens) is small but high-quality. Targeting a 1–3B-token Swahili LA training set is realistic by combining mC4-Swahili + CC-100 + WURA filtered + Masakhane.

**2.2 Swahili-aware models.**
- **AfriBERTa** (Ogueji et al. CoNLL 2021) — 97/111/126M-param BERT trained from scratch on 108.8M tokens across 11 African languages including Swahili.
- **AfroXLMR** (Alabi et al., COLING 2022; HuggingFace `Davlan/afro-xlmr-{base,large}`) — XLM-R adapted via MLM on 17 African languages + Arabic/French/English, ~117M (base) and ~560M (large) params.
- **AfroXLMR-Social** (Belay et al. 2025, arXiv 2503.18247) — DAPT of XLM-R on AfriSocial corpus; relevant precedent for domain-adaptive pretraining on African languages.
- **SERENGETI** (Adebara et al., Findings of ACL 2023; arXiv 2212.10785) — 517 African languages, 8.6B tokens, 277M params (E250 variant).
- **InkubaLM** (Tonja, Dossou et al., arXiv 2408.17024) — 0.4B decoder-only autoregressive LM trained from scratch on 5 African languages (Swahili, isiXhosa, isiZulu, Yoruba, Hausa) plus English/French.
- **AfroLlama 8B** — Llama-3 8B continued-pretrained on 6 African languages.
- **AfriTeVa V2** (Oladipo et al. 2023) — T5-style encoder-decoder for African languages.
- **AfroLM** (Dossou et al., arXiv 2211.03263) — self-active-learning multilingual model.

**2.3 Swahili NLU/NLP benchmarks.** MasakhaNER 1.0 / 2.0 / X (Adelani et al. TACL 2021, EMNLP 2022) — PER/ORG/LOC/DATE for 10/20/20 African languages including Swahili; AfriSenti (Muhammad et al. 2023) — sentiment in 14 African languages; MasakhaNEWS — topic classification; SIB-200 (Adelani et al. 2024) — topic classification for 200 languages; AfriQA — QA; AfriMMLU and AfriXNLI in IrokoBench (Adelani et al. 2025); Global MMLU and MMLU-ProX have Swahili splits — important since Swahili reasoning/medical MCQA is harder to source than NER.

**2.4 Token fertility on Swahili.** XLM-R uses a 250K SentencePiece vocabulary; Swahili was in the XLM-R pretraining mix (CC-100 swh, ~12 dumps for non-English vs. 1 for English) and is reasonably well-tokenized: empirical studies report XLM-R Swahili fertility around 1.5–2.0 tokens/word, much better than mBERT (110K WordPiece, fertility >2) and meaningfully worse than monolingual Swahili tokenizers. AfroXLMR keeps the XLM-R vocabulary so fertility is unchanged; AfriBERTa uses a smaller WordPiece vocab (70K) trained on African languages and has lower Swahili fertility (~1.3) at the cost of weaker English capacity. For Llama-3-8B (128K BPE), Swahili fertility is reported around 2.5–3.5 in the African-LLM survey (arXiv 2506.02280), making decoder LLMs ~2× more expensive than they look on Swahili compared to English. The "Script Tax" study (Dixit & Dixit 2026) and Petrov et al. 2023 are the standard references for this. **Implication for your project**: with XLM-R/AfroXLMR-large, Swahili medical text expands ~1.6–2.0× in tokens vs. English biomedical text of equal information content — budget accordingly.

**2.5 Swahili linguistic features relevant to modeling.** Swahili (Bantu, Niger-Congo) is **agglutinative** with extensive prefix/suffix morphology and a **noun class system** of ~15–18 classes (Proto-Bantu had 19; Swahili uses 1–18 with gaps). Verbs encode subject agreement, tense/aspect/mood, object agreement, derivation (causative, applicative, passive), and class concord — a single Swahili word can encode a full English clause (e.g., *atakayekuja* = "the one who will come"). Implications: (a) **subword tokenizers fragment morphologically rich words inefficiently**, motivating language adapters that can re-cluster morphological variants in representation space; (b) **noun-class agreement creates long-range dependencies** that adapter MLM training over 512-token contexts can capture; (c) Swahili has minimal tone and uses Latin script, so the script-tax penalty is small but the morphological-fragmentation penalty is real. For biomedical terminology specifically, Swahili medical lexicon is sparse and heavily borrows from Arabic and English (the only standard is *Kamusi ya Tiba*, TUKI 2001) — a major reason a domain adapter trained on English biomedical text and bridged via the LA may underperform on rare medical concepts not covered in any Swahili corpus.

### 3. Biomedical NLP resources

**3.1 English biomedical corpora.** PubMed abstracts (~36M abstracts, ~5B tokens, free download via NCBI); PMC Open Access Subset (~6M full-text articles, ~12B tokens); MIMIC-III/IV (~2M de-identified ICU notes, requires PhysioNet credentialing — **not redistributable**, so unsuitable for releasing adapter weights derived from it); MedQuAD (~47K consumer health Q&A); MedDialog; MedNLI (NLI on clinical notes, MIMIC-derived).

**3.2 English biomedical pretrained models.**
- **BioBERT** (Lee et al., Bioinformatics 2019) — BERT continued-pretrained on PubMed + PMC; 110M/340M.
- **PubMedBERT / BiomedNLP-PubMedBERT** (Gu et al. 2021, "Domain-Specific Language Model Pretraining for Biomedical NLP") — trained from scratch on PubMed; sets BLURB benchmark.
- **BioLinkBERT** (Yasunaga, Leskovec, Liang — ACL 2022) — trained on PubMed with citation links via document-relation prediction; SOTA on BLURB and MedQA-USMLE in its parameter class; 110M / 340M.
- **BioGPT** (Luo et al. 2022) — generative biomedical model.
- **BioMistral** (Labrak et al., Findings of ACL 2024, arXiv 2402.10373) — Mistral-7B-Instruct continued-pretrained on PMC; SOTA on 10 medical QA tasks at the 7B scale; multilingual (8 languages, none of them African).
- **Meditron-7B/70B** (Chen et al. 2023) — Llama-2 continued-pretrained on PubMed + Cochrane + medical guidelines.
- **MedAlpaca, ClinicalBERT, SciBERT** — additional reference points.

**3.3 Multilingual biomedical resources.** WMT Biomedical Translation tracks have parallel scientific abstracts in en↔{pt,es,fr,de,zh,it,ru} (no African languages); UFAL Medical Corpus (en↔{es,de,fr,ro}); MeSpEn (en↔es); the MedlinePlus website publishes consumer health pages in dozens of languages including Swahili (vaccine information statements, pregnancy guides, immunization schedules — manually translated by NIH/CDC, freely available). The **largest multilingual medical resource that includes Swahili** is OpenWHO (see §4).

**3.4 Biomedical terminology in Swahili.** UMLS (NLM) is the umbrella resource — ~900K concepts, 200+ source vocabularies (SNOMED CT, ICD-10, MeSH, RxNorm, LOINC). **UMLS coverage of Swahili is essentially nil** in the standard release — translations exist for major European languages but not African languages. WHO ICD-11 has some Swahili entries through national health authority contributions but not full coverage. SNOMED CT has no Swahili translation. MeSH Swahili: no. Practical implication: any UMLS-based normalization for Swahili must be **bootstrapped via translation** (e.g., translate the English UMLS concept names to Swahili using NLLB-200 or Gemini and have a clinician spot-check) — this is itself a small contribution worth describing in the paper.

### 4. Swahili biomedical resources — gap analysis

This is the part of the plan that requires the most explicit acknowledgment of what does *not* exist.

**4.1 What exists (and is usable).**
- **OpenWHO** (Merx, Suominen, Cohn, Vylomova — WMT 2025, arXiv 2508.16048) — 2,978 documents / 26,824 parallel sentences from WHO's e-learning platform across 20+ languages including Swahili. Expert-authored, professionally translated, and **shielded from web crawls** before December 2024 — i.e., not contaminated in most LLM pretraining. WHO Tanzania has confirmed Swahili courses on occupational health, COVID-19, Ebola. **This is the single most important Swahili biomedical resource for your project.**
- **MedlinePlus Swahili** (medlineplus.gov/languages/swahili.html) — hundreds of consumer-health PDFs (vaccine info, pregnancy/postpartum, immunizations, COVID-19, electricity-during-emergency) in en↔swh parallel form, NIH-published. Estimated ~5–15K parallel sentences after cleaning.
- **Mendeley "In the heart of Swahili" Swahili Corpus** (Shikali et al. 2024, PMC11372376) — has an explicit AFYA (Health) category subset.
- **Kamusi ya Tiba** (TUKI 2001) — bilingual English-Swahili dictionary of medicine. Not freely scrapable, but useful as a gold reference.
- **WHO-5 Well-Being Index Swahili** (Mwangala et al. 2018, BMC) and other psychometric instrument translations — small but high quality.
- **AfriMed-QA** (Olatunji et al., NAACL/ACL 2025, arXiv 2411.15640) — 15K English MCQs/SAQs/CQs from 16 African countries. **Not Swahili text, but highly relevant**: (a) the consortium has publicly stated a roadmap to expand to non-English languages including Swahili; (b) Google's MedGemma uses AfriMed-QA + MedQA in training; (c) AfriMed-QA can be used as the *source-language* labeled task corpus for training the task adapter, which is then composed with Swahili LA at inference for Swahili medical QA.
- **MMLU-Pro / Global MMLU / MMLU-ProX Swahili splits** — translated MCQs covering College Medicine, Clinical Knowledge, Professional Medicine, Virology subjects. Used in the MKG-Rank paper (Wang et al., arXiv 2503.16131) explicitly for Swahili medical QA evaluation; this is the most directly usable existing **Swahili medical evaluation set**.
- **Bridging-the-Gap clinical translations** (arXiv 2412.12417) — Winogrande + 3 MMLU clinical sections (College Medicine, Clinical Knowledge, Virology) translated into 8 African languages. Swahili is not in their 8 but isiZulu, Sesotho, Sepedi, Tsonga, Setswana, Igbo, Bambara, Amharic, and Shona are; Winogrande-MMLU-Clinical-ZA covers 3 South African languages.
- **NLLB-200 / FLORES-200** — provides Swahili↔English MT capacity sufficient to generate translated training/eval data on demand; FLORES-200 has a small high-quality Swahili dev/test set (~1K parallel sentences) usable as a translation-quality probe.

**4.2 What does not exist (the gap).**
- No Swahili clinical NER dataset (no Swahili equivalent of n2c2 or BC5CDR or NCBI-Disease).
- No Swahili ICD-10/11 coding dataset.
- No Swahili medical relation extraction.
- No Swahili-native medical QA dataset (everything is translated).
- No Swahili clinical note corpus (the analog of MIMIC-III); strict regulatory and language barriers in Tanzanian/Kenyan health systems.
- No Swahili biomedical terminology gazetteer aligned to UMLS.
- The Notre Dame paper (Oketch, Lalor, Abbasi — arXiv 2508.14051) collects 2,170 Swahili free-text responses to **psychometric** health instruments — closest thing to clinical free text in Swahili, but not annotated for clinical concepts.

**4.3 Realistic downstream evaluation tasks (ranked by feasibility on this compute and timeline).**
1. **Swahili medical MCQA** using MMLU-ProX-Swahili clinical subsets (Clinical Knowledge, College Medicine, Professional Medicine, Virology), AfriMMLU clinical filter, and machine-translated subsets of MedQA-USMLE (with quality filtering). Metric: accuracy, with stratification by subject. Highest signal-to-noise.
2. **Cross-lingual medical NER** by *projecting* English biomedical NER labels (BC5CDR-Disease, NCBI-Disease, BC2GM-Gene, JNLPBA, n2c2 if obtainable) onto NLLB-translated Swahili text using awesome-align or fast_align, with a small (200–500 sentence) human-validated test set. Metric: span-level F1.
3. **Swahili medical text classification** (symptom vs. anatomy vs. medication vs. procedure topic classification) constructed from MedlinePlus Swahili categories. Metric: macro-F1.
4. **English→Swahili medical machine translation** on OpenWHO test split (held out). Metric: ChrF++, BLEU, MetricX, with sentence-level human MQM on a 200-sentence subset. (Note: this requires a seq2seq backbone, e.g., NLLB or mT5; less natural for an encoder-only adapter pipeline.)
5. **Swahili medical NLI** by translating MedNLI to Swahili with NLLB. Metric: accuracy.
6. **AfriMed-QA cross-lingual** — train TA on English AfriMed-QA, evaluate on a translated 500-question Swahili subset.

**4.4 Synthetic benchmark construction (concrete recipe).** Build **Swa-MedBench v0.1** as a project deliverable:
1. Translate MedQA (USMLE-style, 4-option MCQs, ~12K) and MedMCQA (~190K) via NLLB-200-3.3B into Swahili. Filter via round-trip BLEU and OpenWHO-trained translator perplexity. Have 2 native-Swahili clinicians validate a 1,000-question stratified sample (post-edit, accept, or reject).
2. Translate BC5CDR-Disease test set (~500 abstracts) via NLLB; project entity spans via awesome-align with mBERT/LaBSE alignments; have a clinician verify 200 abstracts.
3. Use OpenWHO Swahili held-out test sentences (~2,000) as a held-out perplexity / domain-MLM probe.
4. Use the MMLU-ProX clinical splits as-is (no translation needed; already part of the benchmark).
5. Use MedlinePlus Swahili for a topic classification task (8–12 health topic classes).
6. Release the assembled benchmark with construction code, NLLB version, alignment scripts, and clinician-validation rubric.

This synthetic benchmark + the empirical adapter-composition study is a defensible, novel arXiv contribution.

### 5. Concrete experimental design

**5.1 Base model recommendation.** Three reasonable backbones; pick **AfroXLMR-large** as primary.

| Model | Params | Pros | Cons |
|---|---|---|---|
| **XLM-R-large** | 560M | Standard MAD-X backbone; reproducibility | Swahili in pretraining mix but not specialized; weakest Swahili of the three |
| **AfroXLMR-large** ★ | 560M | DAPT'd on Swahili + 16 African langs already; same architecture as XLM-R; tokenizer identical | Some "Swahili adapter" effect already baked in — must be addressed in baselines (a "no LA" baseline using AfroXLMR alone will be strong) |
| **mT5-large** | 1.2B | Encoder-decoder enables MT eval | 2× memory; slower; less standard for adapter literature |
| **Llama-3-8B / AfroLlama-8B** | 8B | Generative QA, instruction-tuneable, current relevance | Worst Swahili token fertility (2.5–3.5×); no published MAD-X-style results; QLoRA needed; less directly comparable to MAD-X line |

The strongest paper has AfroXLMR-large as primary (best Swahili capacity in the encoder regime with frozen weights, fits comfortably on DGX Spark in BF16 with adapters), XLM-R-large as a secondary ablation (to isolate "DAPT-then-adapter" vs. "adapter-only"), and either Llama-3-8B with QLoRA + LoRA-as-adapter or mT5-base for the MT task as a third backbone for breadth.

**5.2 Language adapter (LA) training.**
- **Corpus**: ~1.5B Swahili tokens deduplicated from mC4-Swahili + CC-100 + WURA + Masakhane raw + Swahili Wikipedia. Aggressive paragraph-level dedup via MinHash (Lee et al. 2022 dedup recipe). Filter via FastText language-ID p>0.9 to remove code-mixed lines.
- **Objective**: standard MLM with 15% masking probability (80% [MASK] / 10% random / 10% unchanged), span-masking optional (Joshi et al. 2020 SpanBERT-style 3-token spans). Sequence length 512.
- **Architecture**: Pfeiffer adapter, reduction factor 16 (bottleneck dim 64 for XLM-R-large d=1024), GeLU, residual around bottleneck, layer norm before; **invertible adapters on the embedding layer** (NICE coupling, reduction factor 2) — these are the part of MAD-X that handle vocabulary mismatch and are particularly important for an under-tokenized morphologically rich language. Add adapter only after the FFN sub-layer (Pfeiffer placement, ~7M params for XLM-R-large), not Houlsby's dual placement.
- **Hyperparameters**: AdamW, lr 1e-4 (adapter weights only) with linear warmup over 2K steps then linear decay, batch size 64–128 sequences (gradient accumulation as needed), 100K–250K steps depending on tokens. BF16 mixed precision, FlashAttention-2, weight decay 0.01, gradient clipping 1.0. Save checkpoint every 10K steps; report dev MLM perplexity on held-out OpenWHO Swahili.
- **DGX Spark wall-clock estimate**: with frozen 560M XLM-R-large in BF16 (~1.1GB weights) + adapter forward/backward + activations at SL=512 BS=64, expected throughput >50K tok/s. 1.5B tokens × 1 epoch = ~30K seconds = ~8 hours per epoch; 3–5 epochs = 24–40 hours. Comfortably feasible.

**5.3 Domain adapter (DA) training.**
- **Corpus**: 1–2B English biomedical tokens from PubMed abstracts (free, ~5B tokens available — subsample to a curated 1B clean subset) + a curated PMC OA subset (~500M tokens). Avoid MIMIC because of redistribution constraints.
- **Objective**: MLM, identical hyperparameters and architecture to LA. Adapter is trained with **English LA stacked underneath** (or with no LA — both configurations should be in the ablation matrix). Pfeiffer adapter, reduction factor 16, **no invertible adapters** on the DA (vocabulary is English-aligned, no script-mismatch problem).
- **Critical design choice**: train the DA either (a) **on top of a frozen English LA** so it specializes to "English biomedical *given* English language baseline", or (b) **directly on the frozen base model** so it specializes to "biomedical regardless of language". (a) is the cleaner MAD-X analog; (b) makes composition with Swahili LA more semantically coherent. Run both as an ablation.
- **DGX Spark wall-clock**: comparable to LA, 24–40 hours for 1.5B English biomedical tokens.

**5.4 Task adapter (TA) training.**
- **Corpus**: depends on downstream task. For medical NER: BC5CDR-Disease (~500 train abstracts) + NCBI-Disease + JNLPBA (small, fast). For medical MCQA: MedMCQA (190K English questions) and AfriMed-QA English MCQs. For text classification: MedlinePlus topic-labeled English pages.
- **Architecture**: Pfeiffer adapter trained on top of the **stack [LA_en → DA → TA]** with all earlier adapters frozen, following MAD-X exactly. Reduction factor 16 (or 8 for richer task adaptation).
- **Hyperparameters**: AdamW lr 1e-4, 5–10 epochs depending on data size, early stopping on dev loss. 3 seeds (42, 7, 123).

**5.5 Composition strategies (the experimental matrix).** Inference-time stack on Swahili medical input:

| # | Stack | Rationale |
|---|---|---|
| C1 | Base only (no adapters) | Zero-shot baseline |
| C2 | LA_swh only | Pure language adaptation |
| C3 | DA_eng only | Pure domain adaptation |
| C4 | LA_swh → DA_eng → TA | Sequential MAD-X analog ★ |
| C5 | DA_eng → LA_swh → TA | Reversed order (does ordering matter?) |
| C6 | AdapterFusion(LA_swh, DA_eng) → TA | Learned attention over the two adapters |
| C7 | Parallel(LA_swh ⊕ DA_eng) → TA | He et al. 2021 parallel composition |
| C8 | BAD-X-style joint LA+DA | Single adapter trained on swh + en-bio mixture |
| C9 | TIES-merged LA_swh + DA_eng → TA | Weight-arithmetic merge (Yadav et al. 2023) |
| C10 | Hyper-X style (lang_id, domain_id) hypernetwork | Generates LA, DA on demand |

C4 is the primary method; C5–C9 are the composition ablations.

**5.6 Strong baselines required for credibility.**
- **(a) Zero-shot transfer of base model** (XLM-R-large and AfroXLMR-large) — fine-tune TA on English data only, evaluate on Swahili.
- **(b) Language adapter only** — LA_swh + TA, no domain adapter (this isolates the "domain transfer" claim).
- **(c) Domain adapter only** — DA_eng + TA, no language adapter (isolates the "language transfer" claim).
- **(d) Full continued pretraining on Swahili biomedical** — build a Swa-Med corpus from OpenWHO + MedlinePlus + machine-translated PubMed abstracts (NLLB) and full-FT XLM-R-large or AfroXLMR-large on it. This is the most expensive baseline (~3–5 days on DGX Spark) but the most important one for the "adapters > full DAPT" claim.
- **(e) Translate-train** — translate English biomedical training data to Swahili with NLLB-3.3B, fine-tune AfroXLMR + TA on translated Swahili data. Standard cross-lingual baseline.
- **(f) Translate-test** — translate Swahili test inputs to English at inference, run English biomedical model, translate back. Common but often a strong baseline.
- **(g) Joint LoRA fine-tuning** — single LoRA module fine-tuned on the union of Swahili raw text + English biomedical raw text + task-supervised data. Apples-to-apples PEFT comparison.
- **(h) BioMistral / Meditron / Llama-3 + AfroLlama few-shot** — modern LLM zero/few-shot baselines (no training needed; use lm-evaluation-harness).
- **(i) GPT-4o / Gemini 2.5 Flash / Claude 3.7 zero-shot prompting** — closed-LLM ceiling reference (per OpenWHO paper, Gemini 2.5 Flash is +4.79 ChrF over NLLB-54B for low-resource health MT — your encoder pipeline will not beat it on free-form QA but should be competitive on NER and classification).

**5.7 Ablations.**
- Bottleneck reduction factor: {4, 8, 16, 32, 64} → quantify capacity-vs-overfitting trade-off.
- Adapter placement: Pfeiffer (FFN only) vs. Houlsby (attention + FFN) vs. parallel (He et al.) vs. Adapter+ (Steitz & Roth 2024).
- Composition order: LA→DA vs. DA→LA vs. fusion vs. parallel.
- Invertible adapters on LA: yes vs. no (this is a known important ablation per MAD-X paper and UDApter).
- Adapter training data volume: {100M, 500M, 1B, 1.5B, 3B} tokens for LA → estimate the data-efficiency curve.
- DA trained on top of LA_en vs. base only.
- Effect of seed: 3 seeds for primary configs, report mean ± std and significance via paired bootstrap (Koehn 2004) or paired permutation test on per-example predictions.

### 6. DGX Spark feasibility analysis

**6.1 Memory footprint.**
- **AfroXLMR-large** (560M params, BF16 = 1.1GB weights). Frozen base + 7M-param Pfeiffer adapter (28MB BF16) + Adam optimizer states for adapter only (Adam uses 2×params for moments, in FP32 = 56MB) + activations at BS=64 SL=512 with FlashAttention-2 ≈ 8–12GB. Fits in <20GB unified memory; comfortable.
- **XLM-R-large**: identical numbers.
- **Llama-3-8B in BF16** (16GB weights) + LoRA-as-adapter (rank 16, ~50M params) + activations + Adam states ≈ 32–48GB. Fits with QLoRA (8GB 4-bit weights + LoRA + activations ≈ 24GB). DGX Spark's 128GB unified memory makes both straightforward; per NVIDIA's published numbers (LMSYS review, NVIDIA blog): Llama-3.1-8B LoRA reaches 53.6K tok/s, gpt-oss-120b inference at 47–57 tok/s.
- **Llama-3-70B / 3.3-70B**: only via QLoRA; throughput ~5K tok/s per NVIDIA. Pretraining a 70B language adapter in 6–10 weeks is borderline; not recommended as primary.

**6.2 Training time estimates.**
| Workload | Tokens | Throughput | Wall-clock |
|---|---|---|---|
| AfroXLMR-large LA MLM (1.5B Swh tokens) | 1.5B | ~50K tok/s | ~8h × 3–5 epochs ≈ 1–2 days |
| AfroXLMR-large DA MLM (1.5B en-bio) | 1.5B | ~50K tok/s | ~1–2 days |
| AfroXLMR-large full DAPT on Swa-Med (300M) baseline | 300M tokens, 3 epochs, full FT | ~15K tok/s | ~2 days |
| AfroXLMR-large TA on MedMCQA (190K examples × 5 epochs) | small | n/a | <6 hours |
| Llama-3-8B continued pretraining via LoRA on Swa raw text | 1.5B | ~53K tok/s | ~8 hours |
| Synthetic benchmark generation via NLLB-3.3B | ~50M tok of MCQ translation | NLLB inference ~15K tok/s on Spark | ~1 day |
| End-to-end (all baselines + ablations + 3 seeds) | — | — | **6–10 weeks part-time** |

**6.3 Encoder regime is the right choice.** Stay in the 100M–560M encoder regime (XLM-R-base/large, AfroXLMR-base/large) for the primary contribution. Use Llama-3-8B + AfroLlama as a **secondary** decoder-LLM baseline for MCQA only, since (a) MAD-X literature and adapter composition results are most established on encoders, (b) the comparison to MedQA-translate-train is cleanest at the encoder level, (c) DGX Spark is bandwidth-bound for >20B inference and unified memory's main advantage (large model loading) is wasted on a primarily training workload.

**6.4 Mixed precision, FP4, FlashAttention.** BF16 for primary training (most stable for MLM continued pretraining; avoid FP16 to prevent loss-scaling instabilities). FlashAttention-2 is fully supported on Blackwell and gives ~2× attention speedup at SL=512 and is essential at SL≥1024. **FP4 (NVFP4) is for inference, not training** — DGX Spark's headline 1 PFLOP is FP4 sparse, dense BF16 throughput is ~31 TFLOPS FP32 / ~250 TFLOPS BF16 dense. For training, expect to operate in the BF16 regime with FlashAttention-2; FP4 becomes relevant for the final inference-time evaluation pass and for serving the model after the project. The DGX Spark unified-memory architecture (LPDDR5x at ~273 GB/s) is the critical bottleneck for inference of large models; for training small adapters atop frozen weights it is a non-issue because activations dominate and fit in fast on-chip caches.

### 7. Evaluation methodology

**7.1 Primary metrics by task.**
- Medical NER: span-level F1 (CoNLL eval), with separate scores per entity type and a strict + relaxed match.
- Medical MCQA: accuracy, with stratification by subject and question difficulty; CoT and 5-shot vs. 0-shot variants for LLM baselines.
- Text classification: macro-F1 (accounts for class imbalance) + weighted-F1.
- MT (if included): ChrF++ as primary, BLEU as secondary, MetricX-23-XL or COMET-22 as learned metric, AutoMQM error analysis on a 200-sentence sample.
- Domain MLM perplexity on OpenWHO held-out: report as a sanity check.

**7.2 Significance testing.** Three random seeds (42, 7, 123) per configuration. Report mean ± standard deviation. Use **paired bootstrap resampling** (Koehn 2004) on per-example outputs with 1,000 resamples; p<0.05 threshold. For NER, also report the AlmostStochasticOrder test (Dror et al. 2019) which is more appropriate for small effect sizes typical of adapter ablations.

**7.3 Cross-lingual transfer evaluation protocols.** Standard zero-shot setup: TA trained on English, evaluated on Swahili. Also report few-shot (5, 50, 200 examples in-target-language) to characterize the data-efficiency curve. Per Hu et al. 2020 (XTREME) and Adelani et al. 2022 (MasakhaNER 2.0), report by-language and aggregate scores; for your single-target case, report by **subject** (clinical knowledge vs. virology vs. college medicine) instead.

**7.4 Fairness/calibration considerations in medical NLP.**
- Calibration: report ECE (expected calibration error) and Brier score on MCQA. Medical models tend to be over-confident.
- Demographic / specialty bias: AfriMed-QA paper (Olatunji et al. 2024) shows large performance variation across specialties and West vs. East African geographies — replicate this stratification on your Swahili eval and discuss.
- Hallucination on free-form medical answers: do not deploy a generative QA model without a hallucination evaluation; for the encoder pipeline this is less of a concern but worth a sentence.
- Translation-induced bias: any translated benchmark inherits the translation system's biases. Always include a clinician-validated subset and explicitly bound results by translation quality.
- Document data licenses; flag that MedlinePlus is public domain (USG work), OpenWHO is CC BY-NC-SA 3.0 IGO, AfriMed-QA has its own license; Masakhane resources are CC-BY-4.0-NC; PubMed abstracts are public-access, full-text PMC OA subset has author-set licenses (mostly CC BY).

### 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Swahili biomedical eval set too small/noisy | High | Triangulate ≥3 eval sources (MMLU-ProX, BC5CDR-translated, OpenWHO-test-set MT, MedlinePlus topic-classification). Have a clinician validate a stratified ~500-question subset and report results both on full and clinician-validated subsets. |
| English DA does not compose with Swahili LA due to tokenizer mismatch | Medium | Invertible adapters on the embedding layer (the original MAD-X mitigation); also try training DA on top of LA_en (bridging via the shared "language baseline" abstraction); and as fallback, train DA on **NLLB-translated Swahili biomedical text** in addition to English. |
| Catastrophic forgetting of Swahili linguistic competence under DA | Medium | Pfeiffer placement preserves residuals; AdapterFusion as composition fallback. Diagnostic: evaluate the LA→DA stack on **MasakhaNER-Swahili** (general-domain Swahili NER). If F1 drops by >2 points vs. LA-only, forgetting is occurring; mitigate via lower DA learning rate or DA with smaller bottleneck. |
| Naive composition fails (Stickland et al. 2021's finding) | Medium-high | Test multiple composition strategies (C4–C10 in §5.5); in worst case, fall back to AdapterFusion or a small language-arithmetic correction (Klimaszewski et al. 2024). |
| LLM baselines (Gemini 2.5, GPT-4o) crush adapter pipeline on free-form tasks | High for QA | Frame the contribution as parameter-efficient + fully open + reproducible + offline-deployable, not as SOTA absolute accuracy. Compare adapter pipeline to BioMistral/Meditron/AfroLlama in the open-model regime, not against frontier closed models. Position closed LLMs as a reference upper bound. |
| Translation contamination of MedQA/MMLU translations into pretraining of LLM baselines | Medium | Use OpenWHO held-out (shielded by login wall pre-2024) and clinician-newly-written items (synthesize ~200 with native speakers if budget allows) as the contamination-robust slice. |
| Reproducibility | Low (manageable) | Pin all dataset versions, random seeds, NLLB checkpoint version, AdapterHub `adapters` library version. Release adapter checkpoints, training code, eval code, and the synthetic benchmark on HuggingFace + GitHub. |
| Data licensing for medical text | Medium | Avoid MIMIC (PhysioNet credentialing prevents weight redistribution). Use only PubMed abstracts (public access), PMC OA (mostly CC BY), MedlinePlus (USG public domain), OpenWHO (CC BY-NC-SA 3.0 IGO — non-commercial, attribution required: clearly mark in releases). Document each source's license in the repo. |
| Clinical safety | Medium | Add a "do not use for clinical decision-making" disclaimer; do not release a generative QA model; if NER is released, mark as research-only. |

### 9. 6–10 week project timeline (part-time on DGX Spark)

| Week | Milestone | Output |
|---|---|---|
| 1 | Data collection & cleaning: Swahili raw (mC4 + CC-100 + WURA + Wiki + Masakhane), English biomedical (PubMed abstracts + PMC OA filtered), OpenWHO Swahili extracted, MedlinePlus Swahili scraped, MMLU-ProX Swahili extracted | Cleaned, deduplicated, FastText-LID-filtered Swahili and English-biomedical training corpora; raw eval splits assembled |
| 2 | Synthetic benchmark construction v0: NLLB-200-3.3B translation of MedQA, MedMCQA, BC5CDR-Disease into Swahili; round-trip-quality filter; alignment-projection of NER spans | Swa-MedBench v0 (uncurated); translation quality numbers (ChrF, perplexity) |
| 3 | Train AfroXLMR-large LA_swh (Pfeiffer + invertible) on Swahili corpus; train DA_en on PubMed; sanity-check via held-out perplexity drops | LA_swh and DA_en checkpoints; perplexity diagnostic plots |
| 4 | Train task adapters: TA_NER (BC5CDR-Disease English), TA_MCQA (MedMCQA English), TA_topic (MedlinePlus English); evaluate composition C1–C5 (zero-shot, LA only, DA only, LA→DA→TA, DA→LA→TA) | First end-to-end results; preliminary main table |
| 5 | Strong baselines (a)–(g): zero-shot, full DAPT on Swa-Med (~2 days training), translate-train, translate-test, joint LoRA, BioMistral / AfroLlama few-shot | Baseline numbers, main result table v1 |
| 6 | Composition ablations C6–C10 (AdapterFusion, parallel, BAD-X joint, TIES-merge, Hyper-X-style hypernetwork) + bottleneck-size and adapter-placement ablations | Ablation tables; composition-method comparison |
| 7 | Catastrophic-forgetting probe on MasakhaNER-Swahili; Swahili-vs-English DA-trained-on-LA ablation; data-volume curve (LA at 100M / 500M / 1B / 1.5B tokens) | Forgetting diagnostic; data-efficiency curves |
| 8 | Clinician validation pass on 500-question Swa-MedBench subset (~2–3 paid clinician hours via Intron Health network or Masakhane volunteers); update results on validated subset | Clinician-validated benchmark v1; revised main table |
| 9 | LLM-baseline comparison run (Gemini 2.5, GPT-4o, Claude 3.7 zero-shot via API; budget ~$200) + writing draft | Complete results, paper draft v1 |
| 10 | Polish, error analysis, reviewer-anticipation ablations, arXiv submission, code/data release | arXiv v1, GitHub repo, HuggingFace adapter checkpoints |

**What can be cut if time runs short.**
- Drop Llama-3-8B / AfroLlama LoRA continued-pretraining experiments (week 4–5 alternative).
- Drop Hyper-X-style hypernetwork composition (C10).
- Drop BAD-X joint training (C8) — only run sequential and AdapterFusion.
- Drop the MT eval task — restrict to NER + MCQA + classification.
- Reduce ablation matrix to bottleneck size {8, 16, 32} only.
- Reduce seeds to 2 instead of 3.
- Use only NLLB-translated MedQA + MMLU-ProX-Swahili for eval; skip BC5CDR translation/alignment.

**What is not negotiable.**
- The four primary baselines (a, b, c, d): zero-shot, LA-only, DA-only, full DAPT — without these the paper has no story.
- At least 2 seeds + significance testing.
- A clinician-validated eval subset, even if only 100 questions — without medical-expert validation the paper will be challenged on safety grounds.
- Catastrophic-forgetting probe on at least one general-Swahili task (MasakhaNER-Swahili).

### 10. Novelty positioning for arXiv

**10.1 What is the contribution?**
The paper's contributions, in order of strength:

1. **First systematic empirical study of MAD-X-style language × *domain* adapter composition for low-resource clinical NLP.** Stickland et al. 2021 did language × domain for NMT; UDApter did monolingual UDA; AdaMergeX/Language Arithmetic did language × task at the LoRA level; nobody has cleanly done language × domain stacking for an encoder pipeline on a low-resource African language in the medical domain. (This is the headline contribution.)

2. **Swa-MedBench: a reproducible Swahili medical evaluation suite** built from OpenWHO + MedlinePlus + NLLB-translated MedQA/MMLU/BC5CDR with clinician validation, released under permissive licenses. This addresses an explicit gap noted by AfriMed-QA's authors and the Africa-NLP scoping review (PMC11923465). (This is the most durable artifact of the paper — even if the empirical adapter results are bested by future work, the benchmark remains useful.)

3. **An empirical answer to "does the Stickland 2021 catastrophic-forgetting result generalize to encoder-only NLU and to the African low-resource regime?"** With a clean diagnostic via MasakhaNER-Swahili degradation analysis.

4. **A new composition method (optional, secondary).** Two candidates: (a) a language-readjustment "domain-aware language arithmetic" that subtracts the English-LA component from a DA trained on top of LA_en before composing with LA_swh; (b) a small fusion adapter trained on a Swahili+biomedical mixture that is only a few hundred K parameters and corrects compositional mismatch. Frame this as "we tried X and Y; X works, Y does not, here is when each helps."

**10.2 Recent (2024–2026) closely-related papers to differentiate from.**

| Paper | Year | Setting | Differentiation |
|---|---|---|---|
| Stickland et al. (arXiv 2110.09574) | 2021 | NMT seq2seq, language × domain adapters | Different: encoder-only NLU, focus on extraction/QA/classification rather than MT; modern compositions (AdapterFusion, TIES) |
| UDApter (Malik et al. EACL) | 2023 | Monolingual UDA | Different: cross-lingual, low-resource, biomedical |
| AdaMergeX (Zhao et al. NAACL) | 2024 | LoRA, language × task | Different: bottleneck adapters (not LoRA arithmetic), domain not task |
| Language Arithmetic (arXiv 2404.15737) | 2024 | MAD-X language adapters | Different: domain composition, full empirical study, new benchmark |
| Language and Task Arithmetic with PEFT Layers (Klimaszewski et al. MRL) | 2024 | PaLM-2 LoRA, language × task | Different: encoder regime, domain, low-resource |
| The Impact of Language Adapters (Kunz & Holmström MOOMIN) | 2024 | NLU, language adapter ablations | Replicate their negative finding ("LA effect is inconsistent") in the medical domain — turn their caveat into your motivation |
| AfriMed-QA (Olatunji et al. NAACL) | 2024 | English Pan-African medical QA | Complementary: provide the Swahili-language counterpart they explicitly call for |
| OpenWHO (Merx et al. WMT) | 2025 | Low-resource health MT | Use as data source; their paper is about LLMs vs. NMT for MT, not adapter composition |
| Bridging the Gap (arXiv 2412.12417) | 2024 | African-language clinical reasoning benchmarks | Use Winogrande + clinical MMLU translation methodology; extend to Swahili |
| AfroXLMR-Social (arXiv 2503.18247) | 2025 | DAPT for African social media | Adjacent: domain-adaptive pretraining for African languages; you do adapter composition vs. their full DAPT — direct comparison |
| Small Models, Big Impact (arXiv 2502.10140) | 2025 | Adapters for African low-resource | Closest empirical ancestor: bottleneck/inv/LoRA on mBERT, XLM-R, Llama-3 with ConceptNet+GlotCC; you extend to clinical domain and add domain adapters |
| FLARE (Borchert et al.) | 2025 | Adapter aggregation | Different aggregation level (output vs. parameter); cite as related |
| Typologically Informed Parameter Aggregation (arXiv 2601.16629) | 2026 | Typological adapter aggregation | Cite; orthogonal to this work |

**10.3 Framing for impact.** Working title: **"Stack and Bridge: Language × Domain Adapter Composition for Swahili Biomedical NLP."** Subtitle/positioning: "We present the first systematic study of MAD-X-style adapter composition where the second adapter is a *domain* adapter rather than a *task* adapter, and apply it to clinical NLP in Swahili — the highest-impact African language with no existing biomedical benchmark." Three-sentence elevator: (1) Cross-lingual domain transfer for Swahili medical NLP is blocked by the absence of Swahili biomedical pretraining data and of clinical evaluation benchmarks. (2) We show that stacking a Swahili language adapter on a frozen multilingual base, then composing it with an English biomedical domain adapter at inference, recovers most of the gap to full domain-adaptive pretraining at <2% of the trainable parameters and <10% of the wall-clock cost, and we identify when the composition fails (echoing Stickland 2021's catastrophic-forgetting warning). (3) We release Swa-MedBench, the first openly licensed Swahili medical evaluation suite, plus all adapter checkpoints.

The combination of (a) a clean empirical study with strong baselines, (b) a real and persistent benchmark gap addressed, (c) explicit replication of a previously published negative result (Kunz & Holmström 2024, Stickland 2021) in a new setting, and (d) full open release on commodity hardware (DGX Spark, ~$4,700) is exactly the profile that makes a strong arXiv preprint and a credible portfolio piece for a senior ML engineering interview at a FAANG / Apple-tier organization. The writing should foreground the engineering rigor (compute estimates, ablation completeness, reproducibility, license documentation, clinician validation) as much as the empirical findings.

---

## Caveats

- **The "first systematic study" claim must be hedged carefully against unpublished concurrent work.** Late 2025 / early 2026 has seen a flurry of adapter-composition papers; before submitting, run a final arXiv full-text search for "language adapter" + "domain adapter" + "Swahili" + "clinical" + "biomedical" within the last 60 days. The closest existing concurrent work appears to be the "Small Models, Big Impact" paper (Kargaran et al., arXiv 2502.10140, Feb 2025) which does adapter-based adaptation for African low-resource languages but focuses on knowledge graphs (ConceptNet) rather than the biomedical domain, and the "Multilingual Domain Adaptation for NMT" line which is seq2seq-only.
- **Some 2026 results in the search above (e.g., arXiv 2601.16629, 2604.22723, 2604.02881, 2602.11174, 2606.06820) may be speculative or unverified previews returned by search.** When citing, verify the paper exists and the contents match before referencing in the final paper. Several of these timestamps appear to be malformed or future-dated; treat them as unconfirmed and substitute with verified 2024–2025 alternatives if needed.
- **OpenWHO and AfriMed-QA are evolving.** The AfriMed-QA consortium has publicly committed to non-English language extension; if Swahili AfriMed-QA appears during the project window, it supersedes the synthetic benchmark for MCQA — switch immediately.
- **DGX Spark performance numbers are NVIDIA-published.** Independent reviews (LMSYS Oct 2025, ServeTheHome, IntuitionLabs) show that real-world throughput especially on inference is bandwidth-limited and below the headline 1 PFLOP figure, which is theoretical FP4 sparse. For training adapters on a ≤560M frozen encoder, this caveat is mild; for any plan involving 70B+ training, it is severe.
- **Translation-based benchmarks have known biases.** Any result on NLLB-translated MedQA-Swahili must be reported with a confidence interval that reflects translation quality, ideally bounded by the clinician-validated subset. Do not over-claim absolute accuracies on translated benchmarks.
- **MIMIC-derived models cannot be redistributed.** If you use BioBERT/PubMedBERT/BioLinkBERT as DA initialization, those are fine (PubMed-only). If you pretrain DA on MIMIC, you cannot release the adapter weights. Stick to PubMed + PMC OA.
- **Clinician validation requires IRB-style awareness.** Even for a research artifact, paying clinicians to spot-check translated medical questions implies a small ethics workflow. Document who validated, what credentials they held, and the agreement protocol.
- **The "FAANG/Apple-level interview piece" framing implies engineering rigor matters as much as scientific novelty.** Reviewers (and interviewers) will look at the repo's reproducibility (single-command training, environment.yml, deterministic seeds), the clarity of the ablation matrix, the discipline of the baseline set, and the honest reporting of negative results — possibly more than at the headline numbers. Plan time accordingly: weeks 9–10 are not "extra polish," they are the difference between a credible paper and a forgettable one.