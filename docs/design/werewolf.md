# 狼人杀加入 Gavis — 设计方案

> 版本: 0.3 | 日期: 2026-08-12 | 状态: P1-P3 已完成（规则+adapter+LLM 求解器+前端）

## 0. 交付物（2026-08-12）

**P1 引擎与规则**
- 引擎：`text` 参数预制能力（`engine.py:_expand_template` 跳过枚举；
  编译路径对多参数模板自动 fallback 到解释路径，probe 校验兜底）
- `_gen_werewolf.py` 生成器 → `rules/werewolf.json`（默认 9 人 3狼1预1女巫1猎人，
  支持 6/9/12 人、屠边/屠城、首夜保护、女巫自救开关）
- `layer2_engine/games/werewolf/werewolf_adapter.py`：部分可观测过滤 +
  轮转映射（`get_current_player`）；配置与 JSON 角色池强校验
- 测试 4 例；实测随机自对弈零超时，狼胜率 ~65-85%（信息不对称预期）

**P2 LLM 求解器**
- `layer3_solvers/llm/ollama_solver.py`（`OllamaSolver`，SolverBase）：每玩家
  一个实例（独立视角），观察+发言历史拼中文 prompt，JSON 意图→ActionInstance；
  非法/超时输出 fallback 随机合法动作
- 实测（qwen3:8b，`demos/demo_werewolf_llm.py`）：**悍跳/对跳/狼群掩护/好人
  识破放逐悍跳狼** 全流程出现；热态 6-10s/步，一局 ~5-8 分钟
- 单测 6 例（prompt 构建 / speak 映射 / target 映射 / 非法回退 / 异常回退）

**P3 前端**
- `layer4_interface/frontend/play_werewolf/`（server.py + session.py +
  static/index.html，端口 8771）：真人 1 座位 vs 8 个 LLM AI 座位；聊天流
  发言、夜晚技能选择、死亡公布、身份保密（死后公布）；AI 回合自动推进
- session 单测 3 例（开局轮转 / 身份隐藏 / 非法动作拒绝）；API 冒烟通过

**实现要点备忘**（踩坑记录）
- mutable 数组初始为空 → 发牌/死亡名单用 `append`（数组索引即玩家索引）
- interpreter 用 `effectMap` 而非模板 `effectRef` 解析 explicit 条目
- 顺序 branch 会互相覆盖 → 链条末端用"phase 未变"守卫
- `player_index` 用 `lt` 计数（`eq` 会返回 1）；`voteLog` 跨轮统计须按
  round 过滤（否则永远放逐已死者 → 死循环）
- 女巫无药时跳过夜晚阶段（否则无合法动作卡死）；轮次上限兜底平局

## 1. 目标

把**狼人杀**（Werewolf / 社交推理游戏）加入四层架构，复用现有
`SolverAdapter` / `SolverBase` / `run_episode` 契约，并打通 Layer 1
（LLM 规则翻译）与 Layer 3 的 LLM 求解器两条此前"预留"的能力。

## 2. 游戏本质 vs 现有框架的适配点

狼人杀 = **部分可观测 + 自然语言通信**的社交推理游戏：

| 环节 | 本质 | Gavis 可表达性 |
|------|------|----------------|
| 夜晚 | 结构化行动：狼人杀人 / 预言家验人 / 女巫救·毒 | ✅ 动作模板（`actions`） |
| 白天发言 | **自由文本**，是策略核心 | ⚠️ 动作模板无法表达自由文本 |
| 投票放逐 | 结构化行动：投给某玩家 | ✅ 动作模板 |
| 角色分配 | 从角色池随机抽 | ✅ chance 模板 |
| 胜负 | 阵营判赢 | ✅ `utility` 规则 |
| 信息隐藏 | 身份、夜晚行动只对本人可见 | ⚠️ 需 adapter 层过滤（见 §3） |

**核心设计决策：给规则语言加一个通用 `text` 参数预制能力，让自由文本发言原生进规则。**

现有规则语言盘点（狼人杀 95% 可声明）：动作模板+视图枚举、phases
阶段机、`mutable` 数组 + effector `append`（支持 dict-of-expressions 记录，
可作发言日志）、visibility、chance、utility。**唯一缺口：动作参数的
domain 必须可枚举**（`_cartesian_product` 展开），自由文本无法枚举，
文本无处安放。

预制能力定义（通用语言原语，非狼人杀特供）：

```json
"params": {
  "intent": {"view": "intents"},           // 参与枚举 → legal 有限
  "text":  {"type": "text"}                // 新预制能力：不参与枚举
}
```

- **枚举处**（`layer2_engine/core/engine.py:_expand_template`）：声明
  `type: "text"` 的参数跳过 cartesian 展开（占位 `""`），canonicalKey
  模板不含 text → legal 列表保持有限，MCTS/MARL 契约不破
- **运行时零改动**：solver 把文本放进 `ActionInstance.params["text"]`，
  `_build_context` 已自动平铺 params 到 ctx（`ctx['$text']` 可用）→
  effector 用 `append {"speaker": {...}, "text": {"var": "$text"}}`
  写发言日志（`mutable` 数组）
- `SchemaValidator` 不校验参数类型，向后兼容

发言动作：`speak:{intent}`，`intent ∈ {claim, accuse, defend, question,
persuade}`（有限槽位，动作空间紧凑，MARL 也能吃）。LLM 玩家与规则/MARL
玩家在同一个动作空间上对抗，循环赛直接复用。

## 3. 四层分工

### Layer 2 — 引擎

- `rules/werewolf.json`（v5.1，默认 9 人局：4 狼 4 民 1 预言家）
  - `constants`: 角色池 / 玩家数 / 夜晚顺序 / 各角色数量（人数由
    `WerewolfAdapter` 注入，同 mahjong 模式）
  - `chance`: 角色分配（deal）、夜晚结果结算
  - `actions`: 夜晚 `kill:{target}` / `check:{target}` / `save` / `poison:{target}`，
    白天 `vote:{target}` / `speak:{intent}`
  - `phases`: deal → night → day(发言环) → vote → night → … → game_over
  - `utility`: 狼全出局 → 好人胜；狼人数 ≥ 好人 → 狼胜（按 Gavis
    收益归一 ±1）
- `layer2_engine/games/werewolf/werewolf_adapter.py`（`SolverAdapter`）
  - 部分可观测：**adapter 层按玩家过滤**，同 texas 隐藏对手底牌的既有模式
    （身份 / 夜晚行动他人不可见；发言日志全部可见）
  - 夜晚按顺序逐人决策（轮次制，天然适配现有 MARL runner）
  - 发言日志容量控制（环形裁剪）可在 adapter 或规则 effector 中处理

### Layer 3 — 求解器

新包 `layer3_solvers/llm/`：

- `OllamaSolver(SolverBase)`：本地 ollama（已实测可用）
  - `select_action(state)`：拼 prompt（身份 + 观察 + `speech_log` + 规则摘要，
    JSON 输出约束）→ 解析 `{intent, target, speech}` → 映射 `ActionInstance`
  - 非法/超时输出 → fallback 随机合法动作（保底）
  - 配置：model / temperature / speech_log 长度 / max_tokens
- 对照基线：`RandomSolver`（随机合法）、模板玩家（如"跟随跳预言家者"）
- 可选记忆：`bge-m3` 对 `speech_log` 相似度检索，压缩长局上下文

注册 `layer3_solvers/__init__.py`（try/except ImportError 同 PPO/MARL）。

### Layer 4 — 界面

- `layer4_interface/frontend/play_werewolf/`：真人打字发言 vs LLM 玩家，
  裁判合成对话历史（风格对齐现有 play_* 单应用服务）

### Layer 1 — 实战打通

- 用 qwen3:8b 实现 `TranslatorProtocol`：狼人杀规则文本 → `rules.json`，
  过 `SchemaValidator` 校验 —— layer1 预留后的首个实现

## 4. 实施分期

| 阶段 | 内容 | 验收 |
|------|------|------|
| P1 | `rules/werewolf.json` + `WerewolfAdapter`（先随机玩家） | 自对弈可跑通、胜负判定正确、引擎测试 |
| P2 | `OllamaSolver` + `demo_werewolf.py` | 9 人局 LLM 自对弈，JSON 决策稳定 |
| P3 | 循环赛扩展：LLM vs LLM vs 随机/模板基线（复用 `run_episode` 玩家分派） | 阵营胜率统计 + 发言质量抽样 |
| P4 | Layer 1 translator 实战 + 人机对弈界面 | 规则文本 → 可执行 rules.json |

## 5. 已验证的可行性（2026-08-12 实测）

- ollama 本地模型：`qwen3:8b`（LLM）+ `bge-m3`（embedding）
- qwen3:8b 能稳定输出约束 JSON（`{intent, target, speech}`），推理合理
  （双预言家矛盾 → 投票）
- 延迟：冷启动 ~58s（加载 5.2GB），热态 **6-10s/次**；9 人局 3 天 3 夜
  ≈ 30 次调用 ≈ 4 分钟/局 → 评测规模控制在 5-8 局

## 6. 风险与对策

| 风险 | 对策 |
|------|------|
| 8B 模型社交推理肤浅（狼人伪装、悍跳质量低） | 角色提示词预置策略；结果解读时明确"LLM 推理能力上限" |
| 延迟高，评测规模受限 | 控制局数；speech_log 截断；必要时换更小模型 |
| 发言意图槽表达力不足 | intent 槽可扩展（加 `follow:{player}` 等）；文本仍完整保留在 speech_log |
| 规则分支（4 人局、守卫等角色）复杂 | 先标准 9 人局，角色可配 |

## 7. 与现有能力的关系

- **MARL**：发言槽有限 → `ActionSpace`/`GameEncoder` 可为 werewolf 扩展
  （观测量含 speech_log 摘要编码），与 LLM 玩家在相同动作空间对打
- **PSRO/CFR**：暂时不做（语言博弈的效用面不平坦，先让 LLM 基线跑起来）
- **循环赛**：`demos/marl_tournament.py` 的"按玩家分派 solver"骨架可直接复用
  —— 狼人杀 LLM 局换规则即用
