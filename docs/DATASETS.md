# Datasets: real SQLite and Gold eligibility

## 1. Spider 1.0: preserved official source + disclosed subset

Run the complete, CPU-only preparation:

```bash
bash scripts/prepare_all_datasets.sh
```

The canonical Spider archive is SHA-256 locked to `00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b`. The downloader uses an immutable Hugging Face mirror (`HAL-9001/spider-databases`) or the original Yale Google Drive fallback; neither is trusted without the exact same digest. It stores the ZIP provenance.

The first preparation stage creates `data/spider/{train,val,test}_ctx*_turn*.parquet` with six matched ablations. Train / internal validation / final official Dev are database-disjoint.

**Official reference SQL occasionally fails in the published SQLite release.** `scripts/filter_gold_sql.py` executes every original Gold query with a declared timeout and row cap. It then produces matched `data/spider_eligible/{train,val,test}_ctx*_turn*.parquet` files and preserves:

- `gold_eligibility_audit.json`: exact excluded task IDs, errors, sample counts and denominator coverage
- `manifest.json`: the Gold-eligible input files, split DB IDs and hashes
- `gold_execution_audit.json`: final verification of every retained query

Training and final held-out blind evaluation must use `data/spider_eligible/`. Training and evaluation must use the same `SQL_MAX_ROWS=100000`. An eligible subset's accuracy must **never** be described as an official full Spider Dev score if any test questions were excluded.

This preprocessing evaluates Gold only once; the policy never sees Gold or an oracle correctness signal during rollout.

## 2. BIRD filtered-train metadata

`scripts/download_bird.py` resolves the actual Hugging Face dataset revision for `birdsql/bird23-train-filtered` and stores source and output SHA-256. `scripts/validate_bird_metadata.py` checks all 6,601 records in CI. **This does not download the huge BIRD training database files**, which remain separately required if BIRD training is undertaken.

## 3. Official BIRD Mini-Dev SQLite

The CPU GitHub workflow `.github/workflows/bird-mini-dev.yml` downloads and verifies the public BIRD Mini-Dev ZIP, extracts it with size and path safety protections, prepares the **500 SELECT-only** subset and verifies actual database content. `scripts/filter_bird_gold.py` records any timeout or invalid-Gold exclusions, with true denominator and coverage.

To run locally after downloading and safely extracting `minidev.zip`:

```bash
python scripts/prepare_bird_minidev.py \
  --source-dir /path/to/extracted/minidev \
  --output-dir data/bird_minidev \
  --full-gold-audit --gold-timeout 30 --max-rows 100000
```

Reports distinguish:

- Official Mini-Dev SELECT-only question count
- Gold-eligible count and excluded SQL with reasons
- Gold validation timeout, result cap and input records SHA-256
- Model evaluation (requires real inference; **not completed by CPU preparation**)

Official BIRD leaderboard EX/R-VES require running the original evaluation scripts on the unfiltered official protocol. Internal SQL result-equivalence on a filtered subset is a separately named metric.

## 4. Experimental restrictions

Never train on Spider official Dev or BIRD Mini-Dev. Keep Gold SQL exclusively in post-rollout evaluation. Hash datasets, held-out task IDs and model checkpoints. Log all attempted tasks including runner failures. Do not publish any previously supplied example figures as new results.

## Official BIRD Mini-Dev EX adapter

A separate exporter at `scripts/export_bird_official.py` can align frozen blind trajectories with the upstream BIRD evaluator by original task index and database ID. It refuses official full-set scoring when Gold eligibility has excluded any instance. The current 500-case official Mini-Dev SQLite CPU audit identified 498 Gold-eligible cases and two documented exclusions. Use `scripts/evaluate_bird.py` on that explicit subset for cross-domain tests; it is NOT the same metric or denominator as the unfiltered official leaderboard. After a genuine model rollout and if a future canonical release yields full Gold coverage, the exporter can invoke upstream evaluation_ex.py with strict index, difficulty and source-hash checks.
