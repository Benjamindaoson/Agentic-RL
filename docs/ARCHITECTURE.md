# 架构设计

```text
Open-source SQL tasks
  ├─ Spider 1.0
  └─ BIRD-SQL filtered train / Mini-Dev
          ↓
Prepared Parquet
  input.task_json / db_path / max_turns / dataset
          ↓
Agent Lightning Controller
          ↓
Real SQL Agent Harness
  question + schema + evidence
  → generate SQL
  → read-only SQLite execution
  → verifier feedback
  → optional rewrite / explicit check
          ↓
RLVR Reward
  execution-equivalent result = task success
  + small validity shaping
  - unsafe / invalid / timeout / retry cost
          ↓
veRL GRPO
  grouped rollouts
  → relative advantages
  → clipped policy update
  → reference KL control
          ↓
Held-out evaluation
  task accuracy / first-turn accuracy / invalid SQL / turns / latency
```

## 运行与训练分离

运行侧只负责与真实数据库交互、采集轨迹和返回奖励。训练侧负责并行 Rollout、组内优势估计与模型参数更新。数据库环境不参与梯度计算，模型训练框架也不承担 SQL 执行逻辑。

## 算法证明对象

本项目不以 Agent 编排复杂度为贡献点，重点验证：

1. 可验证环境奖励是否能提升任务成功率；
2. GRPO 在线更新是否优于静态模型；
3. 上下文长度、交互轮次和显式检查如何影响收益与训练成本；
4. 奖励设计如何避免“能执行但答案错误”的 Reward Hacking。
