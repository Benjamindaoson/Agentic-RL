.PHONY: install install-train test lint toy prepare-spider train evaluate report preflight

install:
	python -m pip install -e '.[dev]'

install-train:
	python -m pip install -e '.[dev,train]'

test:
	pytest
	python -m compileall -q src agent scripts
	bash -n scripts/*.sh

lint:
	ruff check src agent scripts tests

toy:
	python scripts/build_toy_data.py --output-dir data/toy

preflight:
	python scripts/preflight.py --require-gpus 1

prepare-spider:
	python scripts/prepare_spider.py --spider-root "$${SPIDER_ROOT}" --output-dir "$${DATA_ROOT}/spider"

train:
	bash scripts/run_local_training.sh

evaluate:
	python scripts/run_rollouts.py --dataset "$${DATA_ROOT}/spider/val_ctx4096_turn3.parquet" --output "$${RUN_ROOT}/eval.jsonl"

report:
	python scripts/build_report.py --inputs "$${RUN_ROOT}" --output "$${RUN_ROOT}/REPORT.md"
