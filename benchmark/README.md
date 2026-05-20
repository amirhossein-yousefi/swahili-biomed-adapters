# Swa-MedBench v0.1

Synthetic Swahili medical evaluation suite per §4.4 of the project plan.

## Construction recipe

1. **MMLU-ProX-Swahili clinical subsets**: Clinical Knowledge, College Medicine,
   Professional Medicine, Virology — used as-is, no translation needed.
2. **Translated MedQA / MedMCQA**: NLLB-200-3.3B translation; round-trip
   ChrF/perplexity filter.
3. **BC5CDR-Disease Swahili NER**: NLLB translation + awesome-align span
   projection of English BIO tags.
4. **OpenWHO Swahili held-out** (~2k sentences): MLM perplexity probe + small
   parallel-MT slice for translation-quality bounds.
5. **MedlinePlus Swahili**: 8–12 health-topic categories scraped from
   medlineplus.gov; topic-classification eval set.

## Construction artifacts

- `construct_swa_medbench.py` — orchestrates the full pipeline.
- `clinician_validation/rubric.md` — accept / post-edit / reject protocol.
- `clinician_validation/sampling.py` — stratified 1k-question sampler.

## Versioning

Pin and record:
- NLLB checkpoint hash (default: `facebook/nllb-200-3.3B`)
- mBERT / LaBSE alignment-model hash
- HF dataset versions for MedQA, MedMCQA, BC5CDR, MMLU-ProX
- Clinician credentials, count, agreement protocol

## Per-source license labels (also see top-level NOTICES.md)

| Source            | License                  |
|-------------------|--------------------------|
| OpenWHO           | CC BY-NC-SA 3.0 IGO      |
| MedlinePlus       | Public domain (US Gov.)  |
| MMLU-ProX         | per HF dataset card      |
| MedQA             | per repo                 |
| MedMCQA           | per HF dataset card      |
| BC5CDR-Disease    | per BioCreative          |
| NLLB-200-3.3B     | CC-BY-NC 4.0 (NC!)       |

The CC-BY-NC license on NLLB makes Swa-MedBench v0.1 **non-commercial**.
For a commercial-friendly successor, replace NLLB with a permissively licensed
translator or post-edit human translations.

## Disclaimer

Translated medical content has known biases and errors. Always report
results separately on the clinician-validated subset; never as a sole metric.
