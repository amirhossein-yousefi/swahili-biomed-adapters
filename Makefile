SHELL := /bin/bash
PY    := python
EXPORT_DIRS := RESULTS_DIR=$(PWD)/results CKPT_DIR=$(PWD)/checkpoints DATA_DIR=$(PWD)/data

.PHONY: help install test \
        data data-swh data-bio data-openwho data-medlineplus data-mmlu data-bc5cdr data-medmcqa data-masakhaner \
        la la-en da da-on-base da-on-la \
        ta ta-mcqa ta-ner ta-topic \
        baseline-a baseline-d baseline-e baseline-f baseline-g \
        compose eval-mcqa eval-ner eval-cls eval-ppl eval-forget \
        fertility benchmark significance verify-reproduce clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-22s %s\n", $$1, $$2}'

install: ## install package + dev deps
	pip install -e .[flash,dev]

test: ## run smoke tests
	pytest -q tests/

# ---- data prep ----
data: data-swh data-bio data-openwho data-medlineplus data-mmlu data-bc5cdr data-medmcqa data-masakhaner ## prepare all corpora
data-swh:        ; $(EXPORT_DIRS) $(PY) scripts/prepare_data.py +source=swh
data-bio:        ; $(EXPORT_DIRS) $(PY) scripts/prepare_data.py +source=bio
data-openwho:    ; $(EXPORT_DIRS) $(PY) scripts/prepare_data.py +source=openwho
data-medlineplus:; $(EXPORT_DIRS) $(PY) scripts/prepare_data.py +source=medlineplus
data-mmlu:       ; $(EXPORT_DIRS) $(PY) scripts/prepare_data.py +source=mmlu_prox_swh
data-bc5cdr:     ; $(EXPORT_DIRS) $(PY) scripts/prepare_data.py +source=bc5cdr
data-medmcqa:    ; $(EXPORT_DIRS) $(PY) scripts/prepare_data.py +source=medmcqa
data-masakhaner: ; $(EXPORT_DIRS) $(PY) scripts/prepare_data.py +source=masakhaner

# ---- adapter training ----
la:             ; $(EXPORT_DIRS) $(PY) scripts/train_la.py adapter=pfeiffer_rf16_inv
la-en:          ; $(EXPORT_DIRS) $(PY) scripts/train_la.py adapter=pfeiffer_rf16_inv data=la_en
da: da-on-base
da-on-base:     ; $(EXPORT_DIRS) $(PY) scripts/train_da.py train=da_mlm train.mode=on_base
da-on-la:       ; $(EXPORT_DIRS) $(PY) scripts/train_da.py train=da_mlm train.mode=on_la

ta: ta-mcqa ta-ner
ta-mcqa:        ; $(EXPORT_DIRS) $(PY) scripts/train_ta.py train=ta_mcqa
ta-ner:         ; $(EXPORT_DIRS) $(PY) scripts/train_ta.py train=ta_ner
ta-topic:       ; $(EXPORT_DIRS) $(PY) scripts/train_ta.py train=ta_topic

# ---- baselines (§5.6) ----
baseline-a:     ; $(EXPORT_DIRS) $(PY) scripts/train_ta.py baseline=a_zeroshot
baseline-d:     ; $(EXPORT_DIRS) $(PY) scripts/train_full_dapt.py
baseline-e:     ; $(EXPORT_DIRS) $(PY) scripts/translate_corpus.py +mode=train
baseline-f:     ; $(EXPORT_DIRS) $(PY) scripts/translate_corpus.py +mode=test
baseline-g:     ; $(EXPORT_DIRS) $(PY) scripts/train_joint_lora.py

# ---- compose + eval (§5.5, §7) ----
compose:        ; $(EXPORT_DIRS) $(PY) scripts/compose_and_save.py
eval-mcqa:      ; $(EXPORT_DIRS) $(PY) scripts/eval_mcqa.py
eval-ner:       ; $(EXPORT_DIRS) $(PY) scripts/eval_ner.py
eval-cls:       ; $(EXPORT_DIRS) $(PY) scripts/eval_classification.py
eval-ppl:       ; $(EXPORT_DIRS) $(PY) scripts/eval_perplexity.py
eval-forget:    ; $(EXPORT_DIRS) $(PY) scripts/eval_forgetting.py  ## §8 MasakhaNER probe

# ---- utilities ----
fertility:      ; $(EXPORT_DIRS) $(PY) scripts/compute_fertility.py
benchmark:      ; $(EXPORT_DIRS) $(PY) benchmark/construct_swa_medbench.py
significance:   ; $(EXPORT_DIRS) $(PY) scripts/run_significance.py

verify-reproduce: ## re-run a recent run on 1% data subset
	@echo "TODO: pick latest run_hash from results/runs and re-execute on a 1% slice"

clean:
	rm -rf results/runs/* checkpoints/*
