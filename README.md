<div align="center">

# Agentic-RL

### Verifiable GRPO for self-correcting Text-to-SQL agents

**Agent Lightning · veRL · GRPO · RLVR · Qwen2.5-Coder · Spider · BIRD**

从单纯模仿 SQL 答案，升级到利用真实数据库执行信号进行强化学习。核心目标不是堆叠 Agent 编排，而是用严谨的实验回答：**策略更新是否在相同观察和计算预算下真正提高了任务成功率？**

[![CI](https://github.com/Benjamindaoson/Agentic-RL/actions/workflows/ci.yml/badge.svg)](https://github.com/Benjamindaoson/Agentic-RL/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

</div>

> **Evidence status — ENGINEERING IMPLEMENTED; GPU RESULTS NOT YET VERIFIED.**
> 现有 ` reports/benchmark_snapshot.* ` 是给定的参考实验数字，不是本仓库本次代码运行得到的 GPU 实测结果。没有训练日志、权重哈希、原始 Rollout、严格对照和证据校验通过之前，请勿将 80.4% 等数值当成本项目实测成果写进简历、论文或对外宣传。

## 1. 为什么是强化学习，而不是多写 Prompt？

SFT 可以学习问题和标准 SQL 的映射；但面对新数据库、执行错误和语义偏差时，重要的是策略能否把环境反馈转化成更好的决策。这里定义一个真实、可审计的强化学习环境：

- **Observation**：问题、Schema、额外证据、真实 SQL 执行结果和错误；无标准答案信息。
- **Action**：单条只读 SQL，加一个模型自选的 `final / inspect` 决策。
- **Environment**：Spider / BIRD SQLite，安全解析、只读执行、超时和结果预览。
- **Reward**：整条轨迹冻结后，使用独立 Gold SQL 结果计算执行等价奖励。
- **Policy update**：Agent Lightning 收集真实模型请求与奖励，veRL 执行 GRPO 更新并保存 FSDP Checkpoint。
- **Evaluation**：相同样本、相同 Prompt/环境/轮次/Token 预算，以盲策略方式比较 Base、GRPO、No-update 和 Reward Ablation。

**特别重要：Gold SQL 不再决定是否重试、何时停止、是否进入 checker，也不进入后续消息。** Gold 只在整条轨迹结束后进入 ` SqlEvaluator `，计算 reward 和测试指标。

## 2. 方法架构

~~~mermaid
flowchart TD
    DATA[Spider / BIRD tasks] --> PRIV[Private SqlTask: question + gold SQL]
    PRIV -->|Allowlisted fields only| POLICY[PolicyTask: question, schema and budget]
    POLICY --> AGENT[Blind SQL Agent]
    AGENT --> SQL[Candidate SQL + decision]
    SQL --> GUARD[SQL safety guard]
    GUARD --> DB[(Read-only SQLite)]
    DB --> ENV[Observed result or execution error]
    ENV -->|Only if policy asks / query fails| AGENT
    ENV --> TRACE[Frozen raw trajectory]
    PRIV -->|Gold only after completion| EVAL[Post-hoc SqlEvaluator]
    TRACE --> EVAL
    EVAL --> REWARD[Execution-based RLVR reward]
    REWARD --> AGL[Agent Lightning events]
    AGL --> VERL[veRL GRPO / clipped update / KL]
    VERL --> CKPT[Weight checkpoints + step metrics]
    CKPT --> EXPORT[Export FSDP to Hugging Face]
    EXPORT --> TEST[Blind held-out rollout]
    TEST --> STATS[Paired bootstrap, McNemar, evidence gate]
~~~

Policy 与 Oracle 的真实边界在代码里是类型隔离的：

- ` SqlTask ` 带 `gold_sql`；仅用于数据加载和评分。
- ` SqlTask.policy_view() ` 生成白名单公开字段。
- ` SqlAgentRunner.run(PolicyTask) ` **拒绝**传入私有 ` SqlTask `。
- ` SqlEvaluator.evaluate(SqlTask, frozen_trajectory) ` 只在最后评分。
- ` scripts/audit_leakage.py ` 交换 Gold SQL，验证策略的 Prompt、SQL、轮次和终止决定不变；评分应该改变。

完整协议：[docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md)。

## 3. 功能和状态

| 工程能力 | 已实现 | 实测证据要求 |
|---|:---:|---|
| SQLite 查询、结果等价与只读门禁 | ✓ | CI 单元测试 |
| Policy 与 Gold Evaluator 隔离 | ✓ | Gold-mutation metamorphic audit |
| RLVR 配置驱动奖励与 validity-only 消融 | ✓ | Reward / leak tests |
| GRPO 优势、裁剪目标及 KL 核心测试 | ✓ | 算法单元测试 |
| Agent Lightning + veRL 真实训练入口 | ✓ | 必须实际完成 GPU run |
| Prompt Token 上限、可追踪 Schema 裁剪 | ✓ | 实际 prompt token count |
| FSDP Offload / gradient checkpointing / vLLM 设置 | ✓ | GPU 峰值与吞吐指标 |
| Checkpoint 保存、SHA-256、FSDP 合并 | ✓（工具） | 必须生成并加载真实权重 |
| Base / GRPO / No-update / validity-only 控制 | ✓（工具） | 需要同任务实测 |
| Paired Bootstrap / McNemar / 报告门禁 | ✓ | 需要原始评测轨迹 |
| BIRD 数据接入 | 部分 | 需要独立外部分布评测 |

**✓（工具）表示代码功能存在，不表示训练结果已完成。** CPU 验收现在还包括真实 Agent Lightning 1.0.2 / veRL 0.8.0 YAML 契约、OpenAI SDK HTTP 回路、官方 Spider 数据的 Gold 可评分性审计，以及官方 BIRD Mini-Dev SQLite 的数据准备。数据集抽样/过滤结果必须保留覆盖率与异常任务清单。

## 4. 快速开始（不需要 GPU）

要求 Python 3.12。GPU 训练额外要求 CUDA 适配的 PyTorch、Ray、vLLM、Agent Lightning 1.0.x、veRL 0.7.1–0.8.x。

~~~bash
git clone https://github.com/Benjamindaoson/Agentic-RL.git
cd Agentic-RL
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python scripts/build_toy_data.py --output-dir data/toy
python -m pytest -q
python scripts/audit_leakage.py --output artifacts/leakage_audit.json
python scripts/run_experiment_matrix.py --stage smoke   # only prints proposed GPU commands
python scripts/verify_framework_contract.py --output artifacts/framework_contract.json
python scripts/offline_readiness.py --output artifacts/offline_readiness.json
~~~

**CPU 测试不证明 GRPO 训练成功。** 测试覆盖安全门禁、奖励、算法、可观察反馈、Gold 隔离、Token 预算、统计比较与证据校验器。

## 5. 真实数据

Spider 主训练和同分布 held-out 验证：

~~~bash
# CPU: prepare official Spider + audited Gold-eligible train/val/test ablations
bash scripts/prepare_all_datasets.sh

# Details, including exact Gold exclusions and test coverage:
cat data/spider_eligible/gold_eligibility_audit.json
cat data/spider_eligible/gold_execution_audit.json

# Explicitly: data/spider is the untouched raw prepared dataset;
# data/spider_eligible is the Gold-scorable subset used by GRPO.
~~~

数据准备会先发现并记录官方 Gold SQL 无法执行的任务，然后生成**严格一致、可追溯且不会被误称官方全量指标**的可评分数据；每个条件生成三份数据库级隔离的 Parquet：`train_*`（Spider Train 中用于策略更新）、`val_*`（从 Spider Train 按数据库划分出的内部验证集）、`test_*`（**Spider 官方 Dev**，只用于最终盲评测）。不得把训练过程的验证准确率当作最终测试成绩。

| 条件 | Context | Turns | Verifier |
|---|---:|---:|---|
| ctx2048_turn1 | 2048 | 1 | 否 |
| ctx2048_turn3 | 2048 | 3 | 否 |
| ctx2048_turn3_check | 2048 | 3 | 是 |
| ctx4096_turn1 | 4096 | 1 | 否 |
| ctx4096_turn3 | 4096 | 3 | 否 |

BIRD 用于外部泛化，不应与 Spider 混报为同一分布：

~~~bash
# BIRD filtered-train metadata: fetched and hash-verified by prepare_all_datasets.sh
cat data/raw/bird/metadata_audit.json

# Official BIRD Mini-Dev SQLite data: GitHub CPU workflow downloads minidev.zip,
# prepares 500 SELECT-only tasks, classifies unscorable Gold tasks and reports coverage.
# Offline example after safely extracting official minidev.zip:
python scripts/prepare_bird_minidev.py \
  --source-dir /path/to/extracted/minidev \
  --output-dir data/bird_minidev \
  --full-gold-audit --gold-timeout 30 --max-rows 100000
cat data/bird_minidev/bird_mini_dev_manifest.json
~~~

BIRD 数据文件可能需要遵循官方使用与下载流程。未运行 BIRD 外部评测前，不得声称“跨数据集泛化已验证”。参见 [docs/DATASETS.md](docs/DATASETS.md)。

## 6. GPU 最小闭环（先单轮再扩展）

建议先按官方版本兼容说明安装 GPU 依赖，再安装项目训练扩展：

~~~bash
pip install -e '.[train]'
python scripts/preflight.py --require-gpus 1

export MODEL=Qwen/Qwen2.5-Coder-3B-Instruct
export MODEL_REVISION=YOUR_40_CHARACTER_HF_COMMIT_SHA
export TRAIN_FILE="$PWD/data/spider_eligible/train_ctx4096_turn1.parquet"
export VAL_FILE="$PWD/data/spider_eligible/val_ctx4096_turn1.parquet"
export SQL_MAX_ROWS=100000
export RUN_NAME=grpo_ctx4096_turn1_seed42
export CONTEXT_LENGTH=4096
export MAX_TURNS=1
export ROLLOUT_MAX_TOKENS=1024
export GPU_MEMORY_UTILIZATION=0.65
bash scripts/run_local_training.sh \
  --seed 42 --epochs 1 --save-freq 1
~~~

训练脚本自动：

1. 检查本地 GPU 与 Agent Lightning / Ray；
2. 启动 Ray、AGL Server/Controller 和实时 GPU 埋点；
3. 运行基于真实 SQLite 的 veRL GRPO；
4. 保存 Train/Val 文件 SHA-256、配置、Gold-free 评测协议；
5. 捕获原始 `training.log` 并提取 `training_metrics.jsonl`；
6. 校验必须生成 Checkpoint 文件并计算 SHA-256；
7. 执行 Gold-mutation 泄漏测试；
8. 任一步失败则退出非零，不制造假的训练成功状态。

FSDP 参数/优化器 Offload、Gradient Checkpointing、Rollout micro-batch 与 vLLM 显存占比都可以调节。显存是否足够必须依据实际 GPU 和具体版本运行判断。

### 实验矩阵

~~~bash
# 默认 dry-run；不实际占用 GPU
python scripts/run_experiment_matrix.py --stage main
python scripts/run_experiment_matrix.py --stage controls
python scripts/run_experiment_matrix.py --stage ablations

# 显式执行；先 main，再 controls，最后 ablations
python scripts/run_experiment_matrix.py --stage main --execute
python scripts/run_experiment_matrix.py --stage controls --execute
python scripts/run_experiment_matrix.py --stage ablations --execute
~~~

包括 No-update (lr=0) 与 validity-only reward 对照；不能把“上下文更多”“重试更多”直接算成 GRPO 策略学习贡献。

## 7. Checkpoint 导出与盲评测

veRL FSDP 权重不是直接供 vLLM 加载的 Hugging Face 文件；先导出：

~~~bash
python scripts/export_policy.py \
  --checkpoint-root runs/grpo_ctx4096_turn1_seed42/checkpoints \
  --target-dir runs/grpo_ctx4096_turn1_seed42/hf-policy
~~~

然后分别启动**完全相同**配置的 Base 与 GRPO Policy Server，在保留的 **Spider 官方 Dev `test_*`** 上评价，内部 `val_*` 只用于训练期模型选择：

~~~bash
# Base policy
export POLICY_MODEL=Qwen/Qwen2.5-Coder-3B-Instruct
export PROMPT_TOKEN_BUDGET=4096 MAX_RESPONSE_LENGTH=1024
bash scripts/start_policy_server.sh

python scripts/run_rollouts.py \
  --dataset data/spider_eligible/test_ctx4096_turn1.parquet \
  --model sql-policy --tokenizer Qwen/Qwen2.5-Coder-3B-Instruct \
  --policy-checkpoint base-pinned-revision \
  --policy-manifest runs/grpo_ctx4096_turn1_seed42/base_model_identity.json \
  --context-limit 4096 --max-turns 1 --max-rows 100000 --seed 42 --temperature 0 \
  --output runs/eval_base/base_trajectories.jsonl
bash scripts/stop_policy_server.sh

# GRPO policy — exported HF weights
export POLICY_MODEL="$PWD/runs/grpo_ctx4096_turn1_seed42/hf-policy"
bash scripts/start_policy_server.sh

python scripts/run_rollouts.py \
  --dataset data/spider_eligible/test_ctx4096_turn1.parquet \
  --model sql-policy --tokenizer Qwen/Qwen2.5-Coder-3B-Instruct \
  --policy-checkpoint trained-export-sha256 \
  --policy-manifest runs/grpo_ctx4096_turn1_seed42/hf-policy/export_manifest.json \
  --context-limit 4096 --max-turns 1 --max-rows 100000 --seed 42 --temperature 0 \
  --output runs/eval_grpo/grpo_trajectories.jsonl
bash scripts/stop_policy_server.sh
~~~

***注意：*** 上面 `base-pinned-revision`、`trained-export-sha256` 是用户必须填写的真实标识符，不能原样当作实验凭证。

生成严格配对报告（可附加 `--no-update`、`--reward-ablation`）：

~~~bash
python scripts/compare_experiments.py \
  --base runs/eval_base/base_trajectories.jsonl \
  --grpo runs/eval_grpo/grpo_trajectories.jsonl \
  --leakage-audit runs/grpo_ctx4096_turn1_seed42/leakage_audit.json \
  --output-dir runs/comparison
~~~

比较器拒绝样本 ID、数据哈希、Context、轮次、温度、Seed、Tokenizer、SQL 执行预算不一致的结果；输出准确率差值、配对 Bootstrap 95% CI 与 McNemar 精确检验。

## 8. 证据目录

一个完整训练和评测运行应产生：

~~~text
runs/
  grpo_ctx4096_turn1_seed42/
    grpo_run_manifest.json
    training_config.json
    training.log
    training_metrics.jsonl
    gpu_telemetry.jsonl
    resource_metrics.json
    base_model_identity.json
    checkpoints/
    policy_checkpoint_manifest.json
    leakage_audit.json
    hf-policy/
      config.json
      *.safetensors
      export_manifest.json
  eval_base/
    base_trajectories.jsonl
    base_trajectories_metrics.json
    base_trajectories_protocol.json
  eval_grpo/
    grpo_trajectories.jsonl
    grpo_trajectories_metrics.json
    grpo_trajectories_protocol.json
  comparison/
    comparison.csv
    comparison.md
    comparison.json
    evaluation_protocol.json
    leakage_audit.json
    weight_change_audit.json
    evidence_verification.json
    REPORT.md
~~~

训练权重、数据集与原始轨迹体积较大，默认被 `.gitignore` 排除。请保存在受控对象存储或模型仓库，并确保 URI、Hash、权限与留存策略明确。

首先对 Base、GRPO 和 No-update 的真实 HF Tensor 计算数值差异：

~~~bash
python scripts/check_weight_change.py \
  --base-hf-dir /path/to/pinned-base-hf \
  --grpo-hf-dir runs/grpo_ctx4096_turn1_seed42/hf-policy \
  --no-update-hf-dir /path/to/no-update-exported-hf \
  --output runs/comparison/weight_change_audit.json
~~~

输入必须是真实模型权重：不仅检查 GRPO 权重发生非零变化，还检查 No-update 权重应保持不变。

完整验收：

~~~bash
python scripts/verify_evidence.py \
  --training-dir runs/grpo_ctx4096_turn1_seed42 \
  --comparison-dir runs/comparison \
  --output runs/comparison/evidence_verification.json

python scripts/build_report.py \
  --comparison-dir runs/comparison \
  --evidence-verification runs/comparison/evidence_verification.json \
  --output runs/comparison/REPORT.md
~~~

**强制规则：** 无训练实测、无 No-update 控制、缺少权重哈希或未锁定 Base 版本时，证据门禁必须 FAIL；README 不允许自行填入最终准确率。若任何 Gold SQL 被排除，必须披露官方集覆盖率；不能将 filtered subset 的准确率称为官方完整 Spider / BIRD 分数。

## 9. 关键实验局限

- **Gold eligibility。** 训练与测试必须使用 `data/spider_eligible/`，并记录 `gold_eligibility_audit.json`；Gold 失效或执行超时只通过预处理和审计清单处理，不在真实 Rollout 期间挑选样本。
- **三路隔离。** `train_*`/`val_*` 均来自官方 Train，严格按照数据库划分；最终 `test_*` 来自 Spider 官方 Dev，不能用于挑选超参数或 Checkpoint。
- **执行等价不等于在所有数据库上语义等价。** 两条 SQL 可能因测试数据偶然一致；重复样本、复杂 NULL/ORDER BY 与重排可能需要补充额外判定。
- **训练奖励使用 Gold 是允许的；部署期纠错不允许。** 本项目把 Oracle 放在轨迹结束之后。
- **No-update 不能代替所有因果实验。** 还需要多 Seed、固定样本预算、无效 Reward 对照，并报告统计不确定性。
- **单轮与三轮是不同推理预算。** 只有在相同轮次、相同 Token 和可观察反馈下比较 Base/GRPO，才能把差值归因到学习候选机制。
- **BIRD 外部分布评测未完成前不能宣称泛化。** 项目自实现的 execution match 也不能直接当作官方排行榜分数。

详见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)、[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)、[docs/WORK_PLAN.md](docs/WORK_PLAN.md)。

## 10. Frameworks & licenses

- [Agent Lightning](https://github.com/microsoft/agent-lightning) — rollout gateway and Agent training integration (MIT)
- [veRL](https://github.com/verl-project/verl) — GRPO policy updates, vLLM and FSDP (Apache-2.0)
- [Spider](https://yale-lily.github.io/spider) — Text-to-SQL benchmark
- [BIRD](https://bird-bench.github.io/) — cross-domain realistic SQL benchmark
- [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct) — base policy

Repository code: [Apache-2.0](LICENSE). Data and pretrained-model usage follow their respective licenses.
