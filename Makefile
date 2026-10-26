# Convenience targets. On Windows without make, run the command on each line directly in PowerShell.
PY ?= python

.PHONY: setup etl rules spotcheck tune train index test lint up down audit profile

setup:            ## install runtime + dev dependencies
	$(PY) -m pip install -r requirements-dev.txt

etl:              ## raw UCI workbook -> cleaned transactions -> gold customer table (needs data/raw/online_retail_II.xlsx)
	$(PY) src/build_cleaning_pipeline.py
	$(PY) src/build_gold_features.py

rules:            ## mine association rules (Head B) and evaluate against the popularity baseline
	$(PY) src/build_head_b_rules.py
	$(PY) src/evaluate_head_b.py

spotcheck:        ## 9 model families + 2 baselines under identical folds (Head A)
	$(PY) src/spot_check_head_a.py

tune:             ## Optuna tuning (written, NOT executed in the original build: needs optuna and mlflow installed)
	$(PY) src/tune_head_a_optuna.py

train:            ## calibration, threshold, held-out test, then the model artifact + metadata sidecar
	$(PY) src/calibrate_and_threshold.py
	$(PY) src/build_model_artifact.py

index:            ## the assistant's retrieval index is built at API start-up from app/data/policies.json (nothing to run)
	@echo "Assistant index is built in memory at API start-up (EMBEDDING_PROVIDER=hashing by default)."

test:             ## full test suite
	$(PY) -m pytest -q

lint:
	ruff check app scripts tests ui
	mypy

up:               ## whole system: API :8000, UI :8501, PostgreSQL
	docker compose up --build

down:
	docker compose down

audit:            ## D5 evidence: fairness, importance, latency, scalability probes
	$(PY) scripts/fairness_audit.py
	$(PY) scripts/importance.py
	$(PY) scripts/perf_profile.py
	$(PY) scripts/scale_probe.py
