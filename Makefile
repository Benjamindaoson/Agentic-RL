.PHONY: install install-train test lint toy prepare-spider preflight matrix-plan train evaluate leakage-audit evidence report

install:
	python -m pip install -e '.[dev]'

install-train:
	python -m pip install -e '.[dev,train]'

test:
	python -m pytest -q
	python -m compileall -q src agent scripts
	bash -n scripts/*.sh

lint:
	ruff check src agent scripts tests

toy:
	python scripts/build_toy_data.py --output-dir data/toy

preflight:
	python scripts/preflight.py --require-gpus 1

prepare-spider:
	python scripts/prepare_spider.py --spider-root "${SPIDER_ROOT}" --output-dir "${DATA_ROOT}/spider"

matrix-plan:
	python scripts/run_experiment_matrix.py --stage main

train:
	bash scripts/run_local_training.sh

evaluate:
	@test -n "${POLICY_CHECKPOINT}" || (echo "Set POLICY_CHECKPOINT" >&2; exit 2)
	@test -n "${POLICY_MANIFEST}" || (echo "Set POLICY_MANIFEST" >&2; exit 2)
	python scripts/run_rollouts.py \
		--dataset "${DATA_ROOT}/spider/test_ctx4096_turn1.parquet" \
		--output "${RUN_ROOT}/evaluation/trajectories.jsonl" \
		--policy-checkpoint "${POLICY_CHECKPOINT}" \
		--policy-manifest "${POLICY_MANIFEST}"

leakage-audit:
	python scripts/audit_leakage.py --output artifacts/leakage_audit.json

evidence:
	python scripts/verify_evidence.py --training-dir "${TRAIN_DIR}" \
		--comparison-dir "${COMPARISON_DIR}" \
		--output "${COMPARISON_DIR}/evidence_verification.json"

report:
	python scripts/build_report.py \
		--comparison-dir "${COMPARISON_DIR}" \
		--evidence-verification "${COMPARISON_DIR}/evidence_verification.json" \
		--output "${COMPARISON_DIR}/REPORT.md"
