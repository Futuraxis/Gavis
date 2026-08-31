# Gavis 多智能体架构与协同（补充文档）

> 版本: v1.0 | 日期: 2026-08-31 | 状态: 实现中
>
> 本文件是 `docs/design/architecture.md` 的**专题补充**：把 Gavis 中分散在
> 四层里的"多智能体"能力收拢成一张图——对局内多座位博弈、MARL 协作训练、
> 元博弈策略池、陪伴 Agent 体系、以及人机协同闭环——并讲清它们各自依赖的
> 契约与红线。规则语言版本 v5.2（零 BUILTIN + variants 声明式）。

---

## 1. 定位：Gavis 里的"多智能体"是什么

Gavis 没有单一的"多智能体系统"，而是把 **多个 AI 角色同时运作** 这件事
按用途拆成五种形态，分别落在不同层：

| 形态 | 名字 | 所在层 | 协同对象 | 一句话 |
|------|------|--------|----------|--------|
| 一 | 对局内多座位博弈 | L2 引擎 + L4 平台 | AI × AI × 人类 | 一局游戏里多个 AI 座位各看各的投影、各走各的决策 |
| 二 | MARL 协作训练 | L3 `marl/` | AI 学习器 × 对手池 | CTDE 共享 episode runner + 对手编排，让多个策略一起变强 |
| 三 | 元博弈策略池 | L3 `psro/` | 策略池 × 纳什混合 | 策略池 + 收益矩阵 + 最佳响应迭代，逼近均衡 |
| 四 | 陪伴 Agent 体系 | L4 `agent/` | 教练 / 对手 / 啦啦队 × 玩家 | 表达层多身份陪伴，与决策层双脑分离 |
| 五 | 人机协同闭环 | L4 在线学习 / 难度 | AI × 人类 × 经验模型 | 人类行为 → 经验对手模型 → 门禁发布 → AI 对局 |

这五种形态共用同一个数学底座和同一套层间契约，下面先讲底座，再逐形态展开。

---

## 2. 多智能体的形式化底座：POSG

`docs/design/game-model.md` 把每个游戏建模为部分可观测随机博弈（POSG）：

$$\\langle N, S, A, T, O, R \\rangle$$

- **N（玩家集合）**：`rules["players"]` 数组，人数由 `variants` 声明选择
  （狼人杀 9 人、麻将 4 人、UNO 2–10 人、卧底 4–12 人）。
- **A（联合动作空间）**：`actions[].params` 展开，动作模板在衍生视图上过滤。
- **T（转移）**：`effectors` + `chance`；引擎保证同 seed 同状态同动作 →
  同后继，随机性只经 `chance` 引入（各座位共享同一个 `GameEngine` rng）。
- **O（观察）**：`visibility` 声明式投影——**部分可观测是每个 AI 座位
  的默认边界**：`engine.project_observation(state, player)` 只给该座位
  自己的私有视图 + 公开视图，别人手牌/身份永远不出现。
- **R（收益）**：终局 `utility` 逐玩家结算，零和与一般和都支持。

多智能体的**协同基板**因此是引擎本身：每局只有一个 `GameEngine` 实例，
多个 agent 轮流通过 `get_current_player` / `get_legal_actions` /
`apply_action` 在同一状态图上推进（`env.turn` 驱动轮转）。**谁看到什么、
谁的行动合法**全部由规则 JSON 声明，引擎不做任何游戏特化。

---

## 3. 形态一：对局内多座位博弈（运行时多智能体）

### 3.1 引擎侧的支撑原语

- **轮转**：`env.turn` + `phases[].next`，多阶段（狼人杀 night/vote、
  卧底 describe/vote、UNO 罚牌/抢牌）。
- **chance 节点**：发牌、摸牌、身份分配、随机事件都走 `chance`，引擎
  负责 `sample_chance`，agent 不接触随机源。
- **部分可观测**：`visibility` 投影，`my_role` / `my_word` / `dead_roles` /
  `hand_view_*` 等视图只对对应座位可见。
- **text 参数预制能力**：动作参数声明 `"type": "text"` 时不参与合法动作
  枚举（占位 `""`），solver 把文本放进 `ActionInstance.params`，effector
  经 `$text` 读取——自由文本发言桌游（狼人杀发言、卧底描述、自爆猜词）
  的**智能体间通信原语**。

### 3.2 平台装配：一个座位一个求解器

`layer4_interface/frontend/platform/families/social.py` 是运行时多座位的
典型装配（`GameSpec` 数据驱动，无 per-game 分支）：

- **每 AI 座位一个独立 solver 实例**：`_SocialSolverAssembly.solver_for(seat)`
  经 `SolverProvider.create_solver(..., player_id=<seat>)` 现造——每个 agent
  只从**自己的投影**推理（狼人杀 9 座、卧底 4–12 座每个 AI 都只看自己的
  `my_role` / `my_word`）。
- **求解器种类探测**：开局一次：Ollama 可用 → `ollama`（本地大模型发言），
  否则 `random`；快照 `ai_mode` 如实记录，LLM 实际调用失败也降级标注。
- **发言驱动 `_run_ai`**：`while` 循环驱动每个 AI 座位直到人类回合或终局；
  每个座位决策 → `speak` 兜底通稿（不含词本身/身份标签，守卧底红线与
  快照红线）→ `apply_action` → 前滚 chance → 下一座位。
- **记录**：在线学习开启时逐 AI 决策经 `session.recorder` 采集，多座位
  循环同样覆盖。

麻将（4 座 p0–p3）、UNO（2–10 人）则由 `train-cli/games.py` 注册表按
`player_count` 装配；多人局在 P1 平台仍走"玩家 vs 一桌 AI"（AI 座位各用
各的求解器句柄），P2 将升级为桌友群聊（多座位 agent 发言调度）。

### 3.3 快照侧的多智能体呈现

社交族快照只从投影 + 公开视图构建（守隐藏信息红线）：`my_role` /
`my_word`（自己）、`alive` / `discourse`（最近发言）/ `votes` / `deaths`
（公开日志）、`final_roles`（终局亮全场，复盘惯例）。夜晚/发牌等私密
阶段对非本人回合的 `turn` 做**脱敏**——夜间当前行动者是狼人/预言家，
前端高亮该座位等于官方外挂。

### 3.4 与外部的对弈协作

- `frontend/platform/benchmark.py`：后台 job 评测求解器（固定对手/换边），
  AI × AI 的批量化评测。
- `layer4_interface/aifight/openai_compat.py`：OpenAI-compatible 桥，让
  外部 AIFight 客户端把 Gavis 当"模型"调用（Botzone/Gavis envelope →
  `botzone.runner.decide()`）。
- `layer4_interface/botzone/`：Botzone 协议适配，接入外部竞技平台。

---

## 4. 形态二：MARL 协作训练（L3 `marl/`）

`layer3_solvers/marl/`（QMix / HAPPO / MAAC）是**多智能体强化学习**在
Gavis 的落点，全部共享同一套基础设施。

### 4.1 共享 episode runner：`MARLEnv.run_episode`

`marl/env.py` 是整局驱动：chance 节点自动按概率加权解析；每一步
`get_current_player` 定行动者（轮次制，无硬编码 agent 数）；每个决策
记录成 `Transition`（obs / mask / action / log_prob / value / next_value /
reward / done / global_state …），整局聚成 `EpisodeTrajectory`。
**奖励设计：终局 payoff 记入每个玩家自己的最后一条 transition**——否则
轮次制下终局收益永远进不了该玩家的 GAE（γ=0.99 下轻微高估，文档化的取舍）。

### 4.2 固定动作空间投影：`ActionSpace`

引擎的变长 legal 列表投影成定长 one-hot mask，各游戏按槽位布局
（麻将 227 槽、月亮棋 9 槽、德州 48 槽；未知游戏从初始状态探测模板）。

### 4.3 CTDE：集中训练、分布执行

| 求解器 | 集中式组件（训练吃 joint state） | 分布式执行 |
|--------|----------------------------------|------------|
| QMix | MixingNetwork 吃 global_state 拼接 | 每 agent actor 只吃自身 obs |
| HAPPO | 每 agent 一个 MLPCritic 吃全局 obs | 同上 |
| MAAC | 每 agent 一个 AttentionCritic，混全体 (obs, action) | 同上 |

`select_action` 全部**确定性贪心**（masked argmax）；HAPPO 按 agent 分桶
做 per-agent GAE 与优势归一；QMix/MAAC 共用 ReplayBuffer，MAAC 用
SAC 风格 soft-Q（固定温度）+ Polyak 软更新。

### 4.4 协同的关键：对手编排 `OpponentPool`

纯自博弈下学习 agent 的对手永远是"当前的自己"，两网互相追逐、共同漂移
（逼近陷阱）。`marl/opponent_pool.py` 把"这一局与谁打"变成显式决策：

- **对手池**：每 `checkpoint_interval` 局把当前策略冻结成快照入池
  （每玩家一个池，对称增长，避免座位轮换造成单边池为空）；
- **采样模式**：`self`（纯自博弈基线）/ `uniform`（FSP 均匀抽）/
  `pfsp`（`p ∝ win_rate^α` 优先虚构自博弈，OpenAI Five 公式）/
  `curriculum`（`p ∝ decay^age` 旧弱到新强平滑过渡）；
- **WinTracker**：逐池滚动胜负（memory 窗口），供 PFSP 加权；
- **RoleScheduler**：2 人局学习器座位逐局轮换——本局一个座位训练、
  另一座位执行池中冻结快照（on-policy 性质不破）；
- **warmup / 池空自动退化**：回到纯自博弈，与旧行为兼容。

默认编排（麻将注册表 `_MAHJONG_OPPONENT_CFG`）：`pfsp` 模式、
容量 32、每 25 局冻结、warmup 100 局、`eval_interval=50` 做
vs-random 固定基线曲线采样——直接产出"训练曲线平滑"的实测证据。

### 4.5 单智能体 RL 的多智能体化：PPO 自博弈

`layer3_solvers/ppo` 在 moon_chess 上以 **self-play + 双座位轮换 +
零和 bootstrap 取负** 训练（`games.py` 登记 `opponent="self"`、
`entropy_coef=0.05`，600 局）——单 agent 策略在多 agent 对局里通过
自博弈收敛，是 MARL 系列之外的并行路线。

### 4.6 训练注册

`train-cli/games.py`：麻将七变种（含 international）MARL 200 局/座位×3
（qmix/happo/maac，产物 `models/train/<game>/…pt`，平台默认 AI=已训练
MAAC，缺模型回退启发式；136 张组共享 guangdong 产物、108 张组共享
sichuan 产物，**international 不在任何共享组**——无专用 MAAC 产物时走
启发式；同步见 `scripts/sync_maac_models.py`）；德州 6000 局；
moon_chess 2000 局。

---

## 5. 形态三：元博弈策略池（L3 `psro/`）

PSRO（Policy-Space Response Oracles）把多智能体博弈抽象成**元博弈**：

1. **策略池**：维护一组历史策略（初始 = 一个随机策略）；
2. **gamescape**：增量计算收益矩阵——旧配对直接拷贝，**只评估新策略
   涉及的配对**（每配对 Ne 局，ThreadPoolExecutor 并行 + `env.clone()`
   隔离，无共享状态）；审查修复：并行化 + 预算 20000 步防 BR≈随机塌缩；
3. **纳什混合**：`nash_solver.solve_nash` 求混合策略；
4. **最佳响应**：`tabular_q_best_response` 对着纳什混合训练新策略入池；
5. **exploitability**：纳什混合对每个池成员的负偏差（审查 P1-3 修复：
   按列玩家测 `u(nash, pi_i)`，旧代码测反了方向全是噪声）。

用途：moon_chess 训练对手池（5 迭代 × 20000 步，Ne=30）；`games.py`
运行时 `opponent_model="psro"` 可让 Hybrid 用策略池当对手。这也让
"策略多样性"（非单一最优）成为训练目标的一部分。

---

## 6. 形态四：陪伴 Agent 体系（L4 `agent/`，"LLM + Skill"）

`layer4_interface/agent/` 是**表达层**的多智能体：对话引擎（`DialogueEngine`）
+ 人格（`persona.py` PERSONAS）+ 场景（`scenarios.py` SCENARIOS）+ 技能
（`skills.py`）+ 隐藏信息守卫（`hidden_guard.py`）。LLM（统一客户端
`LLMClient`）可选，失败回退模板台词。

### 6.1 三种陪伴身份（决策层与表达层双脑分离）

| 身份 | 上下文 | 观察来源 | 说话边界 |
|------|--------|----------|----------|
| 啦啦队（默认） | `SkillContext` | **玩家**投影 | 默认扫描全拦（任何牌面都改写） |
| 教练（教学局） | `TeachContext` | **玩家**投影（和玩家看一样的牌） | teaching 扫描：放行玩家自己的牌，拦 AI/对手的牌 |
| 对手（二人非教练） | `OpponentContext` | **AI 自己**的投影 + 玩家公开动作序列 | adversarial 扫描：放行 AI 自己的牌力措辞，拦玩家的隐藏牌 + 具体花色点数 |

关键协同设计（`docs/user/teaching.md` 三红线）：

1. **教练看的 = 玩家看的**：教练唯一数据入口是玩家自己的投影
   (`project_observation(state, player_pid)`)——比玩家多知道任何东西
   都是泄密。
2. **双脑分离**：对手脑（会话 solver，AI 座位公平落子）与教练脑
   （`agent/coach.py`，在玩家座位替你算参考动作）互不相通。
3. **防录制污染**：教练的参考动作走 `raw_solver`（未被 `RecordingHandle`
   包装的原始句柄）——不会以 "ai" actor 混进在线学习轨迹。

### 6.2 隐藏信息守卫（多智能体间的信息红线）

`hidden_guard.py` 两道防线：

1. `assert_no_hidden`：拒绝投影观测里携带黑名单字段名（`sb_hole` /
   `hand_p0…` / `roles` / `seerResult` …）的上下文——拦截任何绕过
   `project_observation` 的路径；
2. `scan` 后置令牌扫描，四态互斥：**default**（全拦）/ **teaching**
   （放行玩家牌、拦 AI 牌）/ **adversarial**（放行 AI 自己的牌力、
   拦玩家隐藏牌与具体花色点数——报牌=明牌）/ **revealed**（终局
   showdown 揭底后全放行，可复盘点评）。

### 6.3 多人桌的陪伴（P2 方向）

多人非教练局（麻将 4 人、狼人杀 9 人等）P1 仍走啦啦队 fallback，
P2 升级为**桌友群聊**：每座位一个 agent + 发言调度（`speaker` 按
座位 pid + 显示名），社交族快照的 `discourse` / `votes` / `deaths`
视图即群聊的公共语境。

---

## 7. 形态五：人机协同闭环（在线学习 + 自适应难度）

### 7.1 在线学习（`layer4_interface/online_learning/`）

人类与 AI 的对局不是终点，而是经验来源（详见
`docs/design/online-learning.md`）：

```
对局（人+AI 决策） → TrajectoryRecorder 逐决策捕获 → JSONL 持久化
  → 轨迹 → OnlineLearningSignal → 经验对手模型（empirical table）
  → 门禁发布（样本≥下限 ∧ 覆盖率>0 ∧ 候选胜率 ≥ 基准胜率−0.03）
  → DefaultSolverProvider 注入 Hybrid（empirical_table +
    opponent_model="empirical"）→ 下次对局 AI 用真实人类行为建模
```

要点：

- **捕获零侵入**：`RecordingHandle` 包装求解器句柄，AI 决策（含多动作
  循环、多座位）全部覆盖，不改任何 GameSpec 闭包；
- **门禁与回滚**：发布前不回归赛（候选 vs 当前模型，固定种子换边），
  失败隔离（`apply` 捕获一切异常 → reason="error"），可回滚；
- **防污染**：教练参考动作走 `raw_solver`（见 §6.1）。

### 7.2 自适应难度（`layer4_interface/difficulty/adaptive.py`）

`AdaptiveController` 按玩家近窗口胜率自动升降 AI 搜索预算
（`pick_budget` + `pacing_scale`），AI 与玩家形成**共同调节**的对手关系：
玩家变强 → AI 自动变强，保持对局张力（平台 `adaptive_active` 标记回显）。

### 7.3 偏好记忆（`layer4_interface/profile/`）

档案原子存/清除：默认性格、默认难度、每游戏胜负累计——让"和谁一起玩、
多难"成为可持续的协同配置（`PlayManager` 开局按档案装配）。

---

## 8. 协同机制与契约总览

五形态的协同都有明确的契约边界：

```
Layer 1  Translator    自然语言规则 → rules.json（schema + L2 冒烟校验）
Layer 2  GameEngine    L2→L3 单一契约；多玩家/部分可观测/chance 全部声明式
Layer 3  SolverBase    select_action / train / save / load
Layer 4  BaseBinding   契约 + SolverProvider（L4 定义协议，应用层装配实现）
        └─ 依赖规则：Layer N 只依赖 Layer N-1；L4 不 import L3；
            无循环依赖（装配点在 train-cli/games.py，唯一允许 import L3 处）
```

| 协同需求 | 机制 | 红线/兜底 |
|----------|------|-----------|
| 多座位各看各的 | `visibility` 投影 + 每座位一个 solver | `assert_no_hidden` 黑名单 + 快照只从投影构建 |
| 多策略一起变强 | `OpponentPool`（FSP/PFSP/curriculum）+ RoleScheduler | warmup/池空退化自博弈；每玩家一个池对称增长 |
| 元博弈均衡 | PSRO 策略池 + gamescape + Nash + BR | exploitability 按列玩家方向测（P1-3 修复）；池容量淘汰最旧 |
| 表达层不泄密 | 双脑分离 + `scan` 四态 | teaching/adversarial 镜像扫描；revealed 终局例外 |
| 人机经验回流 | 捕获 → 经验表 → 门禁发布 → Hybrid 注入 | 不回归门禁 + 失败隔离 + 版本回滚；raw_solver 防污染 |
| 难度自适应 | 胜率 → 搜索预算 + pacing | `adaptive_active` 回显；显式档位不受影响 |
| LLM 不可用 | 探测 → 降级 | social: ollama→random + 通稿发言（不含词/身份）；agent: 模板兜底 |

---

## 9. 已修复的协同缺陷（历史教训）

- **P1-1 next_mask 语义**：`run_episode` 前滚 chance 节点后再取
  next_obs/next_mask/next_value——chance 后全零 mask 曾让 QMix target
  ≈ −1e9 直接发散；
- **P1-2 MAAC actor loss 未 detach**：critic 被 actor 梯度二次更新、
  共享优化器失衡 → 改 REINFORCE `-(log_prob·(q_online.detach()−τ·log_prob))`；
- **P2 可复现性**：HAPPO/MAAC 未种 torch/np RNG → 与 QMix 一致在
  `__init__` 种子化；
- **PSRO 塌缩**：BR 预算 2000 → 20000、Ne 10 → 30、元博弈并行化；
- **PPO 自博弈塌缩**：旧默认对手=random 且只训练黑座 → 双座位轮换 +
  零和取负 + 600 局 + entropy 0.05；
- **在线学习门禁**：经验表注入后必须同时把 `opponent_model` 切到
  "empirical"（否则默认 uniform 表从未被读取）。

---

## 10. 现状与路线图

**现状（已实现）**
- 运行时多座位：狼人杀 9 座 / 麻将 4 座 / UNO 2–10 人 / 卧底 4–12 人，
  每 AI 座位独立求解器 + 部分可观测投影 + 发言兜底；AI×AI 评测
  （benchmark / aifight / botzone）；
- MARL：QMix/HAPPO/MAAC 共享 runner + 对手编排（pfsp 默认），
  平台默认 AI = 已训练 MAAC（缺模型回退启发式）；
- PSRO 元博弈：策略池 + 纳什混合 + exploitability；
- 陪伴三身份（啦啦队/教练/对手）+ 隐藏信息守卫四态 + 双脑分离；
- 在线学习 MVP（德州经验对手模型）门禁发布；自适应难度；
  教学对局覆盖全部平台游戏。

**下阶段**
- 多人桌陪伴升级：桌友群聊（每座位 agent + 发言调度，P2）；
- MARL 对手编排扩展到更多游戏（UNO/狼人杀类）；经验对手模型
  复用到 MCTS/PPO（经 `rollout_policy` / 经验表注入的既有接缝）；
- 平台鉴权/工程化，多智能体对局（AI×AI×人混合桌）作为平台一等公民。