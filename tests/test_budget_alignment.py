"""Ensure Gold classification, training and evaluation agree on row caps."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_all_spider_pipeline_defaults_agree_on_5000_rows():
    prepare = (ROOT / "scripts/prepare_all_datasets.sh").read_text(encoding="utf-8")
    train = (ROOT / "scripts/run_local_training.sh").read_text(encoding="utf-8")
    eval_py = (ROOT / "scripts/run_rollouts.py").read_text(encoding="utf-8")
    filter_py = (ROOT / "scripts/filter_gold_sql.py").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/data-validation.yml").read_text(encoding="utf-8")
    assert 'SQL_MAX_ROWS' in prepare and ':-5000' in prepare
    assert 'SQL_MAX_ROWS' in train and ':-5000' in train
    assert 'ap.add_argument("--max-rows", type=int, default=5000)' in eval_py
    assert 'parser.add_argument("--max-rows", type=int, default=5000)' in filter_py
    assert "--max-rows 5000" in workflow
