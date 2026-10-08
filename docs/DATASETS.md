# 数据集与严格切分

## Spider 1.0 — 三路数据库级隔离

Spider 1.0 官方 Train 与 Dev 含不同数据库。禁止再将官方 Dev 作为 veRL 训练期间的验证集。
`scripts/prepare_spider.py` 从 Spider 官方 Train 中按固定随机种子和 **数据库 ID** 划分：

| 文件 | 来源 | 用途 |
|---|---|---|
| `train_ctx*_turn*.parquet` | 官方 Train DB 的训练部分 | GRPO 参数更新 |
| `val_ctx*_turn*.parquet` | 官方 Train DB 中按数据库划出的内部验证集 | 训练期间模型选择 |
| `test_ctx*_turn*.parquet` | **官方 Dev（独立数据库）** | **最终 Base/GRPO/Controls 盲评测** |

三个集合数据库 ID 两两不重叠；如有重叠，准备脚本将报错。每个设置写入 `manifest.json`，记录切分种子、样本数、数据库数与数据库 ID 哈希。

~~~bash
python scripts/download_spider.py --output-dir data/raw/spider
python scripts/prepare_spider.py \
  --spider-root data/raw/spider --output-dir data/spider \
  --split-seed 42 --internal-val-fraction 0.10
~~~

支持上下文 2048/4096、最大轮次 1/3 和三轮显式自检消融。最终评测必须读取与对应实验预算一致的 `test_*` Parquet，不使用训练时的 `val_*`。

## BIRD — 仅在独立 held-out 数据上验证泛化

`scripts/download_bird.py` 下载公开记录元数据及 Mini-Dev 工具仓库。数据库和 held-out 开发集数据需要遵循 BIRD 官方流程手动取得；不得以 BIRD Train 伪装外部泛化测试。

~~~bash
python scripts/evaluate_bird.py \
  --records /path/to/official-bird-dev.json \
  --db-root /path/to/bird-dev-databases \
  --output-dir runs/bird_external \
  --policy-checkpoint ACTUAL_MODEL_IDENTITY \
  --policy-manifest /path/to/export_manifest.json
~~~

评测脚本将因数据库缺失或不支持的记录被过滤而拒绝报告完整集分数。其结果是**项目内部 SQLite 执行匹配 EX**，不是 BIRD 官方 R-VES 排行榜分数，后者须按官方独立工具另行验证。

## 数据治理

- 不将原始 Spider/BIRD SQLite 数据库、Parquet、模型权重或私有任务运行轨迹提交 Git。
- 每次运行保存数据集 SHA-256、任务 ID 集合 SHA-256、切分种子和模型身份。
- SQL 环境使用 SQLite 只读 URI、query_only、SQL 安全检查、结果上限与超时。
- 只有在 Policy 轨迹完全结束后，Gold SQL 才用于后置评分。

## Official Spider Gold SQL compatibility

The raw official Spider 1.0 annotations include some queries that are not executable under the bundled SQLite DBs, and some whose results exceed the runtime 5,000-row limit. Do not drop these silently or claim full-set accuracy. Run `scripts/filter_gold_sql.py --input-manifest data/spider/manifest.json --output-dir data/spider_eligible --max-rows 5000` to create a **documented Gold-eligible subset** with exact per-split exclusion IDs and coverage. Use `data/spider_eligible/` consistently for training, internal validation and held-out results, while retaining the source data and audit manifests. A result measured on this subset is *not* the official full Spider Dev score.
