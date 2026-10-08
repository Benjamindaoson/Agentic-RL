# RLVR 奖励函数与 Reward Hacking 防护

## Blind policy 与 Oracle 双路径

`SqlAgentRunner.run(PolicyTask)` 只接收 Schema、任务问题、公开证据和真实 SQLite 执行反馈。模型看不到 `gold_sql`、不知道结果是否匹配参考答案，也不能根据标准答案决定是否重试/停止。

策略**先完成整条轨迹**，然后 `SqlEvaluator.evaluate(SqlTask, trajectory)` 才能读取 Gold SQL 并给出执行匹配结果和 Reward。训练期使用 Gold 计算环境奖励是合理的，但推理/评测期不能把 Gold 正确性反馈泄漏给模型。

## 奖励权重由代码读取 YAML

权重来自 `configs/reward.yaml`，并由运行时 `load_reward_config()` 实际加载：

| 成分 | 权重 |
|---|---:|
| Gold Execution Match | +0.90 |
| SQL 合法 | +0.03 |
| SQL 可执行 | +0.07 |
| 危险 SQL | -1.00 |
| 非法 SQL | -0.20 |
| SQL 超时 | -0.15 |
| 每次额外重试 | -0.02 |

总 Reward 截断在 [-1, 1]。只有正确执行等价才有可能拿到超过 0.5 的奖励；可执行但错误的查询最高只有 0.10。

**Reward Ablation**：将 `REWARD_MODE=validity_only` 传入训练，移除 Gold match 的 +0.90 信号，保留合法性、可执行性及重试成本。该消融区分“学会正确回答”和“学会输出语法合法的 SQL”。

## 风险与限制

- **Oracle 泄漏**：`scripts/audit_leakage.py` 将 Gold 换成另一条 SQL，验证策略 Prompt、决策、重试和停止保持不变。
- **危险 SQL**：仅允许单条读查询、数据库只读 URI 和 SQLite query_only。
- **结果截断**：输出超过上限不计执行正确。
- **Gold 失败**：Gold SQL 不能执行或结果截断，则拒绝任务而不是给模型无依据的失败标签。
- **错误等价**：只在单个 SQLite DB 上得到相同结果，不能保证 SQL 逻辑在任意数据上语义等价。
- **重试成本**：限定轮次、Token 和 SQL 超时，报告有效查询率、P95 延迟与轮次。
- **伪造增益**：固定测试集、推理预算、模型身份和运行条件，使用 No-update/validity-only 对照以及权重张量差异审计。

`reports/benchmark_snapshot.*` 是给定参考数字，不属于已经复现的训练实测；严禁以其替代真实运行证据。
