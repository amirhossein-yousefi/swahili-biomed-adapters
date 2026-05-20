# Clinician validation rubric — Swa-MedBench v0.1

This rubric governs the §9 non-negotiable clinician-validation pass on a
~500–1000 question stratified subset of Swa-MedBench. Per the §7.4 / §8 risk
table, every paper that uses translated medical content **must report results
separately on the clinician-validated subset** alongside the full set.

## Decision per item

For each translated MCQ / NER sentence / topic-classification document,
record one of:

| Decision     | Definition                                                                   |
|--------------|------------------------------------------------------------------------------|
| **Accept**   | Translation is faithful and the medical content is correct.                  |
| **Post-edit**| Translation needs minor edits; clinician supplies an edited version.         |
| **Reject**   | Item is medically wrong, ambiguous, or untranslatable; drop from the eval.   |

For MCQ items, also record per-question:
- whether the **stem** is acceptable
- whether each **option** is acceptable
- whether the **labelled answer** is medically correct in Swahili context

## Stratification

Stratify the validation sample across:
- subject (Clinical Knowledge, College Medicine, Professional Medicine,
  Virology, MedMCQA general medicine)
- difficulty (using the source dataset's difficulty if present)
- option-content type (numeric values, drug names, symptoms, etc.)

## Credentials and process

Document for each clinician validator:
- name (or anonymous ID), country, specialty, years of experience
- whether they are a native Swahili speaker
- the date(s) of their validation session(s)

For inter-annotator agreement: if more than one clinician validates the
same item, report Cohen's κ (or Krippendorff's α for >2 raters) at the
accept/edit/reject level.

## Compensation and ethics

- Document the compensation rate per hour.
- If formal IRB review is not pursued, follow informed-consent best practices:
  validators should know they may stop at any time, and that no patient data
  is involved (the items are public translated MCQs / public-domain text).
- Do not include any PHI or MIMIC-derived content in the validation pool.

## Outputs

Two artifacts:
1. `validated_decisions.jsonl` — per-item record with decision + edits +
   validator ID + timestamp.
2. `validated_subset_indices.json` — the list of indices accepted (post-edit
   counts as accepted), used as `eval.clinician_validated_subset` in
   `configs/eval/ner.yaml` and `configs/eval/mcqa.yaml`.

## Versioning

- Pin the NLLB checkpoint hash that produced the translations.
- Pin the source-dataset version (HF revision) for each component.
- Tag the validated subset with a release label, e.g. `v0.1-clin500-2026q2`.
