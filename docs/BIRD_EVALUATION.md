# BIRD Mini-Dev evaluation: internal subset versus official full-500

## Why two protocols?

The official BIRD Mini-Dev SQLite file contains **500 SELECT-only test questions**.
On GitHub's CPU runner, 498/500 reference SQL queries were scorable under the
project's declared 30s SQLite timeout; two required more time and were excluded
from the project's **internal** Gold-eligible evaluation. This is a property of
this local evaluation protocol, not a license to remove those two questions from
the official benchmark.

| Protocol | Question count | Gold used by policy? | Scorer |
|---|---:|---|---|
| Internal Gold-eligible EX | 498 / 500 under recorded 30-second CPU audit | Never | Project posthoc SQLite result equivalence |
| Full official Mini-Dev EX | **500 / 500** | Never | Upstream bird-bench/mini_dev evaluation_ex.py |

**Neither path has a genuine model accuracy result before actual model inference.**
The CI tests use deterministic stubs to validate message formats and exports.

## Step 1: Prepare the original 500 and audited subset (CPU)

Safely extract the source ZIP (see \`.github/workflows/bird-mini-dev.yml\`
for the source URL and validated ZIP extraction).

\`\`\`bash
python scripts/prepare_bird_minidev.py \
  --source-dir /path/to/extracted/minidev \
  --output-dir data/bird_minidev \
  --full-gold-audit --gold-timeout 30 --max-rows 100000
\`\`\`

This creates:

- \`bird_mini_dev_select.parquet\` — **original 500**, same ID order
- \`bird_mini_dev_eligible.parquet\` — local Gold-scorable subset
- \`bird_mini_dev_manifest.json\` — source hashes, counts, coverage
- \`bird_mini_dev_eligible.gold_eligibility.json\` — detailed exclusions
- \`bird_mini_dev_validation.json\` — real SQLite Gold verification

Do **not** train on either Mini-Dev file.

## Step 2A: Internal EX on the Gold-eligible subset (requires model endpoint)

Start your real Base or trained policy with \`scripts/start_policy_server.sh\`,
then run:

\`\`\`bash
python scripts/evaluate_bird.py \
  --prepared-parquet data/bird_minidev/bird_mini_dev_eligible.parquet \
  --policy-manifest /path/to/actual_hashed_policy_manifest.json \
  --policy-checkpoint ACTUAL_IMMUTABLE_MODEL_REVISION \
  --model sql-policy --tokenizer Qwen/Qwen2.5-Coder-3B-Instruct \
  --base-url http://127.0.0.1:8000/v1 \
  --sql-timeout 30 --max-rows 100000 \
  --output-dir runs/bird_eligible_grpo
\`\`\`

The JSONL, metrics and protocol identify the denominator and the exact
excluded IDs. This output is **NOT** the complete official Mini-Dev metric.

## Step 2B: Official EX on ALL 500 (requires model endpoint)

Run the policy on the original 500 tasks **without posthoc Gold checking**.
The inference loop itself has no \`gold_sql\`, \`reward\`, or \`success\` and no
oracle-driven stop condition.

\`\`\`bash
python scripts/run_blind_predictions.py \
  --dataset data/bird_minidev/bird_mini_dev_select.parquet \
  --output runs/bird_full500/blind_predictions.jsonl \
  --model sql-policy \
  --tokenizer Qwen/Qwen2.5-Coder-3B-Instruct \
  --policy-manifest /path/to/actual_hashed_policy_manifest.json \
  --policy-checkpoint ACTUAL_IMMUTABLE_MODEL_REVISION \
  --context-limit 4096 --max-turns 1 \
  --sql-timeout 30 --max-rows 100000 \
  --require-official-mini-dev

python scripts/export_bird_official.py \
  --trajectories runs/bird_full500/blind_predictions.jsonl \
  --minidev-manifest data/bird_minidev/bird_mini_dev_manifest.json \
  --output-dir runs/bird_full500/official_export
\`\`\`

The exporter guarantees the original canonical task ID order and database
identities for every sample; it refuses mixed/foreign IDs, leaked Gold and
missing predictions. Crucially, **full-500 export does not depend on the local
Gold eligibility filter**.

## Step 3: Run the actual upstream official EX scorer

Clone and pin [bird-bench/mini_dev](https://github.com/bird-bench/mini_dev)
at a recorded commit, and supply the official difficulty JSONL and Gold SQL
files corresponding to the SQLite Mini-Dev release. Upstream accepts:

- \`--predicted_sql_path\`, \`--ground_truth_path\`
- \`--db_root_path\`, \`--diff_json_path\`
- \`--sql_dialect SQLite\`, \`--meta_time_out\`, \`--num_cpus\`
- \`--output_log_path\`

The integration wrapper:

\`\`\`bash
python scripts/export_bird_official.py \
  --trajectories runs/bird_full500/blind_predictions.jsonl \
  --minidev-manifest data/bird_minidev/bird_mini_dev_manifest.json \
  --output-dir runs/bird_full500/official_export \
  --run-official \
  --official-code /path/to/pinned/bird-bench/mini_dev \
  --db-root /path/to/minidev/dev_databases \
  --gold-file /path/to/mini_dev_sqlite_gold.sql \
  --difficulty-file /path/to/mini_dev_sqlite.jsonl \
  --num-cpus 4 --sql-timeout 30
\`\`\`

The \`official_bird_ex_manifest.json\` report becomes authoritative only if
the actual upstream process exits cleanly and produces a valid EX log. Never
substitute the 498-query internal score for official 500-query EX. Never report
a synthetic CPU unit-test prediction as measured model accuracy.
