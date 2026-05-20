# Third-party data and model attributions

Training and evaluation in this repository depend on external resources released
under their own licenses. Document and respect each before redistributing
checkpoints or derived datasets.

| Resource | License | Notes |
|---|---|---|
| **OpenWHO parallel sentences** (Merx et al., WMT 2025; arXiv 2508.16048) | CC BY-NC-SA 3.0 IGO | Non-commercial; attribution required. Mark in any release. |
| **MedlinePlus Swahili** (NIH/NLM) | Public domain (US Gov. work) | Attribution recommended. |
| **AfriMed-QA** (Olatunji et al., NAACL/ACL 2025) | per repository terms | English-only; consult upstream license. |
| **PubMed abstracts** | NCBI public access (no copyright on US-Gov.-authored portions) | Bulk download via NCBI E-utilities. |
| **PMC OA Subset** | Author-set (mostly CC BY) | Per-article license check required. |
| **MMLU-ProX / Global MMLU Swahili** | per HuggingFace dataset card | translated MCQs; clinical subjects. |
| **MedQA** | per repository | USMLE-style 4-option MCQs. |
| **MedMCQA** | per HuggingFace card | English MCQA. |
| **BC5CDR-Disease** | per BioCreative | English NER source. |
| **MasakhaNER 2.0** | CC BY 4.0 NC | Swahili general-domain NER (forgetting probe). |
| **mC4 Swahili** | ODC-By 1.0 | Common Crawl derivative. |
| **CC-100 Swahili** (Conneau et al. 2019) | per CC source | Common Crawl derivative. |
| **WURA** (Oladipo et al. 2023) | per repository | African-language portion of mC4. |
| **Masakhane raw corpora** | CC BY 4.0 NC | African NLP community. |
| **NLLB-200-3.3B** (`facebook/nllb-200-3.3B`) | CC-BY-NC 4.0 | Translation; non-commercial. |
| **AfroXLMR** (`Davlan/afro-xlmr-{base,large}`) | MIT | Backbone. |
| **XLM-R** | MIT | Backbone. |

**MIMIC**: deliberately not used. PhysioNet credentialing prevents redistribution
of weights derived from MIMIC; this codebase relies on PubMed + PMC OA only for
the biomedical domain adapter (§3.1, §8 risk table).

**Clinical use**: research only. Do NOT use any artifact produced by this
codebase for clinical decision-making.
