# 开源数据集接入

## Spider 1.0：主训练与同分布评测

- 10,181 个自然语言问题；
- 5,693 个唯一复杂 SQL；
- 200 个多表数据库；
- 138 个领域；
- 训练与测试使用不同数据库 Schema，适合验证跨 Schema 泛化。

仓库通过 `scripts/download_spider.py` 下载官方数据，并由 `prepare_spider.py` 生成 2048/4096 上下文、1/3 轮交互和显式检查等实验变体。

## BIRD-SQL：外部泛化评测

- `birdsql/bird23-train-filtered`：6,601 条质量筛选后的训练样本；
- BIRD 数据库强调真实数据库内容、外部知识、脏数据和 SQL 效率；
- BIRD Mini-Dev 可用于独立于 Spider 的外部评测。

BIRD 数据库体积较大，仓库自动下载公开元数据；数据库文件需按 BIRD 官方说明下载，并通过 `--db-root` 接入。

## 数据安全

- 只运行 SELECT/WITH；
- SQLite 以只读 URI 打开；
- 每个任务只保存数据库路径，不复制数据库到 Git；
- `data/`、数据库、训练轨迹和模型权重均被 `.gitignore` 排除。
