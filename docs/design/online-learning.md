# Gavis 在线学习（Online Learning）设计

> 版本: v0.1 | 状态: 已实现（Phase 1 捕获 + Phase 2 经验对手模型/门禁/发布）| 日期: 2026-08-23
>
> 相关: `docs/design/architecture.md` §2（Layer 4 OnlineLearning）与 §7 路线图。

---

## 1. 目标

把**真实人机对局**沉淀为可复用经验，周期性更新求解器策略，且——

- 更新前经过**不回归门禁**（候选 vs 当前模型短赛，固定种子、换边）；
- 发布有**版本与回滚**；
- 全程**可见可配置**（平台 API + 前端页面 + CLI）；
- 数据与用户可见历史**严格分离**（隐藏信息不外泄）。

MVP 落点是德州扑克 Hybrid 求解器的**经验对手模型**：AI 从人类真实行动
（按信息集计数）中学习对手倾向，`opponent_model` 由 `uniform` 升级为
`empirical`。捕获与管线按游戏通用，后续求解器（MCTS 价值学习、PPO
离线摄入、CFR 周期重训）直接复用协议与数据通道。

**默认启用游戏：`texas_holdem`**（`LearningManager.enabled` 门控默认仅此
一款，其余游戏需经 `/api/learning/config` 显式开启；`profile.json` 的
`learning_enabled` 字段当前未被后端消费，是前端展示字段）。

---

## 2. 分层职责

严格维持「Layer N 只依赖 Layer N-1；L4 不 import L3」。

| 层 | 模块 | 职责 |
|----|------|------|
| L4 | `layer4_interface/online_learning/recorder.py` | 逐决策捕获：`TrajectoryRecorder`（人类决策在 `GameSession.step` 处记录；AI 决策经 `RecordingHandle` 包装求解器句柄记录——不改任何 `GameSpec` 闭包，多动作循环/多座位麻将同样覆盖）；`jsonable` 序列化兜底 |
| L4 | `.../store.py` | `LearningStore`：`data/online_learning/<game>/trajectories.jsonl`，整局原子追加（决策行 + 终局行），损坏行跳过，`trim` 保最新 N 局 |
| L4 | `.../signals.py`、`feedback_collector.py` | 轨迹 → `OnlineLearningSignal`（stable 数据类，含 `user_rating`/`solver_suggestions` 留白字段）；内存采集器 |
| L4 | `.../models.py` | `OnlineModelStore`：已发布经验表（纯数据 `{info_key: {action_key: count}}`），版本化 + 上一版回滚，持久化在 `data/online_learning/models/<game>.json` |
| L4 | `.../manager.py` | `LearningManager`：实现 `LearningHooks`（`wrap_handle`/`on_finished`，被 `PlayManager` 调用）；apply 管线（建表 → 门禁 → 发布）、`start_auto` 后台线程轮询、状态 API |
| L3 | `layer3_solvers/hybrid/opponent_model.py` | `EmpiricalModel`（Laplace 平滑 + 未见信息集回退 uniform）；`HybridSolver.learn_online()` 增量合并（duck-typed 信号，L3 不 import L4） |
| L3 | `.../hybrid/solver.py` | `HybridConfig.opponent_model="empirical"` + `empirical_table` / `empirical_table_path` 注入 |
| 应用层 | `train-cli/games.py`（经 `train_cli` 桥） | `DefaultSolverProvider` 挂 `OnlineModelStore`：创建德州 Hybrid 时读取已发布表注入（显式 `empirical_table` kwarg 优先——门禁用）；`OnlineModelStore` 来自 L4（应用层可自由 import L4） |
| L4 前端 | `frontend/platform/server.py`、`platform-frontend/` | `/api/learning/status|apply|config` 路由 + `LearningPage` |

---

## 3. 数据流

```
GameSession (人机对局)
  ├─ 人类决策 → PlayManager.move → GameSession.step → recorder.record_human
  ├─ AI 决策   → RecordingHandle.select_action （不改 run_ai 闭包）
  └─ 终局      → PlayManager.move → learning.on_finished → recorder.finish
                        │
        data/online_learning/<game>/trajectories.jsonl   （整局原子块）
                        │
LearningManager.apply(game_id)   （API 手动 / --learning-interval 后台 / CLI）
  ① build_empirical_table：人类决策按 info_key 计数
  ② gate：候选 vs 当前模型（无则 uniform）固定种子短赛 K=20，双方换边
  ③ 发布条件：样本 ≥ min_samples(30) 且 覆盖率 > 0 且 候选胜率 ≥ 基准胜率 - 容差(0.03)
  ④ 通过 → OnlineModelStore.publish（版本 +1，上一版留作回滚）；失败 → 保留旧版
                        │
新对局 start() → DefaultSolverProvider → HybridConfig(opponent_model="empirical", 表=当前版)
```

门禁对比双方都是 Hybrid 对手模型搜索（`imperfect_information=True`，
`mcts_budget=300`），由 `LearningManager._play_one` 驱动（镜像
`BenchmarkRunner` 的换边逻辑），因此与真实对局同一条决策路径。

---

## 4. 数据格式

决策行（每行一个 JSON）：

```json
{
  "match_id": "8f3a2c1d", "game_id": "texas_holdem",
  "step": 1, "actor": "human", "player": "p_sb",
  "state": {"env": {...}, "_arrays": {...}},
  "action": {"template_id": "...", "type": "act", "params": {...}, "canonical_key": "act:raise:4"},
  "info_key": "5070f99e...", "legal": ["act:fold:0", "act:call:2", ...]
}
```

终局行：

```json
{
  "match_id": "8f3a2c1d", "game_id": "texas_holdem", "terminal": true,
  "winner": "p_bb", "utilities": {"p_sb": -12.0, "p_bb": 12.0},
  "human_pid": "p_sb", "ai_pid": "p_bb", "difficulty": "easy",
  "started_at": "...", "finished_at": "..."
}
```

经验表（发布产物，`data/online_learning/models/<game>.json`）：

```json
{ "game_id": "texas_holdem", "version": 2, "samples": 87, "coverage": 19,
  "published_at": "...", "gate": { "episodes": 20, "candidate_win_rate": 0.55, ... },
  "table": { "<info_key>": {"act:fold:0": 23, "act:call:2": 11, ...} } }
```

---

## 5. 门禁与失败安全

- **样本门槛**：总人类决策 ≥ `min_samples`（默认 30）且覆盖率 > 0，否则
  `reason="insufficient"`，status 显示 `pending`。
- **低样本信息集**：`EmpiricalModel` 用 Laplace 先验（`prior_alpha=1.0`）
  平滑，单次观察不会让某动作近乎确定；未见信息集回退 uniform。
- **回归拒绝**：候选胜率 < 基准胜率 − 容差 → `reason="rejected"`，保留旧版
  并记录门禁数据（可回滚 `revert()`）。
- **数据安全**：学习数据只在 `data/online_learning/`（已 gitignore）；
  绝不写入用户可见的 `data/matches/`；`store.py`/`models.py` 都有
  路径遍历防护（game_id 白名单）；`jsonable` 兜底非 JSON 值。
- **隐私红线**：`TrajectoryRecorder._decision` 只存
  `engine.project_observation(state, player)`（决策者自己的信息集投影），
  不落盘含对手底牌的全量 god-view 状态。
- **线程/进程**：store/models/manager 均用 `threading.Lock`（与平台其余
  部分一致）；单进程假设明确——多实例部署需外部存储（文档注明，后续）。
- **失败隔离**：`apply()` 捕获一切异常 → `reason="error"`；后台
  `start_auto` 每轮独立 try/except，不中断循环、不影响 HTTP 服务。
- **中断对局**：`TrajectoryRecorder` 先在内存缓冲，`finish()` 才整局落盘；
  被驱逐/服务停止的未完成对局不会留下半局数据。

---

## 6. API / UI / CLI

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/learning/status` | GET | 每游戏：启用、对局数、决策数（人/AI）、模型版本/样本/覆盖率/门禁 |
| `/api/learning/apply` | POST | `{game_id}` 单游戏（缺省=全部启用游戏）同步执行建表+门禁+发布 |
| `/api/learning/config` | POST | `{game_id, enabled}` 启停（MVP 内存态，重启复位） |

前端：`platform-frontend/src/pages/LearningPage.tsx`（状态表 + 「立即学习」+
启停开关，5s 轮询）。

CLI：`python train-cli/train.py --game texas_holdem --solver hybrid` 训练 Hybrid
（含经验表注入）；平台侧 `--learning-interval N` 后台自动 apply，或手动走
`/api/learning/apply`（原 `demos.online_learning_eval` 入口已随 demos 清理
移除，功能由平台服务与前端页面覆盖）。

服务端：`python -m layer4_interface.frontend.platform.server
--learning-interval SECONDS`（0=仅手动，默认 0）。

---

## 7. 扩展路线（Phase 3，未实现）

- **MCTS 价值/rollout 学习**（moon_chess/gomoku）：终局结果 → 小价值模型，
  经 MCTS 已可插拔的 `rollout_policy` 注入（`HybridSolver` 已有先例）。
- **PPO 离线摄入**：`PPOSolver.learn_online()` 从轨迹重建 transitions
  （on-policy 需 IS/行为克隆辅助损失；torch import 保护）。
- **CFR 周期重训轮换**：复用 `train-cli/train.py --game texas_holdem --solver hybrid` 产物，后台 job 模式。
- **狼人杀/LLM 反馈**：`solver_suggestions` + `user_rating` 字段已预留。
- **多实例部署**：store/models 迁移到外部存储（Redis/文件锁）。
</content>