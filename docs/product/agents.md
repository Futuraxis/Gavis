# Gavis「陪你玩 Agent」多 Agent 并行执行 · 子任务文档

## A. 共享上下文（每个子任务都必须遵守）

### A.1 仓库与分层约束

- 仓库根：`D:\Futuraxis\Gavis`（Python 3.11+，`from __future__ import annotations` 全文启用）。
- **四层架构硬约束**：Layer N 只依赖 Layer N-1；**Layer 4（`layer4_interface/`）绝不 import Layer 3（`layer3_solvers/`）**，层间只走契约。
  - L2→L3：`GameEngine`；L3→L4：`SolverProvider` 协议（L4 定义协议，`train-cli/games.py` 装配实现）。
  - 求解器一律经 `layer4_interface/solver_provider.py` 的 `SolverHandle` / `SolverProvider` 获取，不得 `from layer3_solvers import ...`。
- 关键 API（L2 `GameEngine`，见 `layer2_engine/core/engine.py`）：
  - `engine.create_initial_state()`、`engine.get_node_type(state)`、`engine.get_current_player(state)`、`engine.get_legal_actions(state) -> list[ActionInstance]`、`engine.apply_action(state, action) -> State`、`engine.sample_chance(state)`、`engine.apply_chance(state, outcome)`、`engine.is_terminal(state)`、`engine.get_utility(state, player)`。
  - **`engine.project_observation(state, viewer) -> dict`**：按 `visibility` 声明式投影的观测（v5.2，含 `visibility.env` 秘密字段隐藏）。`engine.get_observation(state, player_id)` 是其别名。
  - `engine.eval_expr(expr, extra_ctx=None)`：规则表达式求值（前端展示助手用）。
  - 状态是 `dict`：`state["env"]` 是环境字段，`state["_arrays"]` 是数组。

### A.2 代码规范（详见 `docs/coding-standards.md`）

- `ruff format` + `ruff check`（line-length=120）；Google 风格 docstring（`---` 分隔章节）；全覆盖类型标注（优先 `X | None`）；导入分组（标准库→第三方→项目内）；层内相对导入（`..base`）、跨层绝对导入（`layer2_engine.*`）；Dataclass 配置 + Protocol 契约；pytest（固定 seed=42、fixture）。

### A.3 隐藏信息守卫（红线，子任务 2/4/5 尤其注意）

- 德州底牌、麻将手牌、狼人身份、未翻牌堆都是**隐藏信息**。
- 一切 Agent 对话/建议/复盘的输入，**只允许来自 `engine.project_observation(state, human_pid)`（人类视角投影）+ `get_legal_actions` + 公开字段 `eval_expr`**；**禁止直读 `state["_arrays"]` 里的隐藏数组**（德州 `_sb_hole/_bb_hole`、麻将 `hand_p0..`、狼人身份等）。
- 隐藏信息"放行门"：只有对局结束且已 reveal（如德州 showdown）才可引用对手隐藏牌（参照 `platform/games.py` 的 `_poker_snapshot` 中 `revealed` 门的写法）。

### A.4 两个已定位 bug 根因（子任务 1 专用）

1. **麻将平台无法开局**：`layer4_interface/frontend/platform/games.py` 的 `_mahjong_create_solver`（约 375 行）硬编码 `provider.create_solver("mahjong", "mahjong", ...)`，但求解器注册表 `train-cli/games.py` 的 `GAMES` 里只有 `mahjong_guangdong / mahjong_hongzhong / mahjong_blood`（没有 `"mahjong"`），`create_solver` 首行查 `GAMES.get(game_id)` 得 None → 抛 `ValueError: 未知游戏: mahjong`。三变种 `create_engine` 已正确按 variant 闭包，但 `create_solver` 漏传真实 `game_id`。
2. **在线学习发布不生效**：`train-cli/games.py` 的 `DefaultSolverProvider.create_solver` 里 `if name == "hybrid" and self.online_models is not None:` 只 `kwargs.setdefault("empirical_table", table)`，**没把 `opponent_model` 设为 `"empirical"`**；`HybridConfig.opponent_model` 默认 `"uniform"`，`HybridSolver.__init__` 里 `if cfg.opponent_model == "empirical"` 恒假 → 注入的经验表从未被读取。

---

## B. 子任务拆分总览

6 个并行子任务，**文件所有权互不重叠**（避免合并冲突），依赖只通过冻结契约解耦。

| # | 子任务 | 目录/文件（独占） | 产出 |
|---|--------|------------------|------|
| C1 | P0 阻断 bug 修复 | `layer4_interface/frontend/platform/games.py`、`train-cli/games.py`、`tests/test_layer4_interface/test_platform_session.py`、`test_online_learning*.py` | 麻将可开局；发布后新对局用经验表；回归测试 |
| C2 | Agent 陪伴后端 | 新建 `layer4_interface/agent/`（7 个文件） | P0 两性格 + 九场景技能 + 模板兜底 + 隐藏信息守卫 |
| C3 | 自适应难度 + 偏好记忆后端 | 新建 `layer4_interface/difficulty/`、`layer4_interface/profile/` | 胜率→预算控制器；节奏映射；profile 原子存/清除 |
| C4 | 复盘后端 | 新建 `layer4_interface/review/` | 关键节点/胜负手/失误 + 改进建议 + 通用兜底评估 |
| C5 | 前端 | `platform-frontend/src/**` | 聊天区、Agent 形象、复盘页、设置页、主题 |
| C6 | 训练 quick preset + 旧应用退役 + 文档 | `train-cli/train.py`、归档 `play_*`、`CLAUDE.md`、`docs/user/*`、`docs/design/architecture.md` | 演示时长训练；退役旧应用；文档同步 |

**依赖图**：C1 独立；C2 依赖"隐藏信息守卫"约定（自包含）；C3 依赖 C1 之后的 `history` 读取（只读历史文件，无代码耦合）；C4 依赖 C2 的 `agent/evaluation.py` 契约（接口冻结，可并行实现）；C5 依赖 C2/C3/C4 的契约（只按接口，不按实现）；C6 独立。

**接线约定（重要）**：`platform/server.py`、`platform/session.py`、`platform/games.py` 的 Agent 钩子与 `/api/agent/say`、`/api/match/hint`、`/api/profile`、`/api/review/:id` 路由**不在任何子任务内接线**，由集成阶段（执行者本人）统一收口；C1 对 `games.py`/`train-cli/games.py` 的 bug 修复是唯一例外。

---

## C1. P0 阻断 bug 修复

**目标**：修复 A.4 两个 bug，并补回归测试。

**文件范围**（独占）：
- `layer4_interface/frontend/platform/games.py`
- `train-cli/games.py`
- `tests/test_layer4_interface/test_platform_session.py`
- `tests/test_layer4_interface/test_online_learning.py`、`tests/test_layer4_interface/test_online_learning_api.py`

**实现要点**：

1. 麻将：把 `_mahjong_create_solver(provider, engine, seed, budget)` 改为按 `game_id` 闭包的工厂 `_make_mahjong_solver(game_id: str) -> Callable`，返回 `provider.create_solver(game_id, "mahjong", engine, seed, budget)`；三个麻将 `GameSpec` 的 `create_solver=` 分别用 `_make_mahjong_solver("mahjong_guangdong")` 等。（`RUNTIME_FACTORY` 里已有 `"mahjong"` 键 → `MahjongHeuristicAI`，`mahjong_*` 的 `runtime_solvers=("mahjong","random")`，故只改传参即可。）
2. 在线学习：`DefaultSolverProvider.create_solver` 里 `if name == "hybrid" and self.online_models is not None:` 分支中，`kwargs.setdefault("empirical_table", table)` 之后补 `kwargs.setdefault("opponent_model", "empirical")`。

**验收/测试**：
- 新增：`manager.start("mahjong_guangdong", "p0", "easy", player_count=2)` 用**真实 `default_provider`** 能返回非 None 会话、`snapshot()["my_hand"]` 长度正确；`mahjong_hongzhong`/`mahjong_blood` 同理（2p/4p）。
- 新增：`DefaultSolverProvider(online_models=store).create_solver("texas_holdem","hybrid",engine,seed,budget)` 后，断言 solver 的 `config.opponent_model == "empirical"`（或 `solver._opponent` 是 `EmpiricalModel` 实例）。
- `python -m pytest tests/test_layer4_interface -q` 中麻将与在线学习用例全绿；`ruff check/format` 全绿。

---

## C2. Agent 陪伴后端（`layer4_interface/agent/`，全新建）

**目标**：实现"LLM + Skill"对话引擎的确定性半边——技能（Skill）+ 人格（Persona）+ 模板兜底 + 隐藏信息守卫；P0 先落 2 性格（`gentle` 温柔陪伴、`teacher` 认真教学），`banter`/`cold` 留占位但结构完整。

**文件范围**（独占新建）：`layer4_interface/agent/__init__.py`、`persona.py`、`scenarios.py`、`skills.py`、`evaluation.py`、`hidden_guard.py`、`llm_client.py`、`dialogue_engine.py`

**契约接口（冻结，C4/C5/集成依赖它）**：

```python
# persona.py
@dataclass(frozen=True)
class Persona:
    key: str                       # gentle/teacher/banter/cold
    display_name: str
    verbosity: int                 # 0(高冷少言)~2(健谈)
    tone: str                      # 温柔/教学/吐槽/高冷
    fallback_lines: dict[str, list[str]]   # scenario -> 兜底台词表（离线确定性）

PERSONAS: dict[str, Persona]       # 4 键；P0 实现 gentle/teacher，banter/cold 给占位 Persona

# scenarios.py
SCENARIOS = ("greet","good_move","blunder","help","ai_win","ai_lose","illegal","idle","game_over")

# skills.py
@dataclass
class SkillContext:
    human_pid: str
    observation: dict              # = project_observation(state, human_pid)
    legal_actions: list            # = get_legal_actions(state)
    evaluation: dict               # = evaluation.evaluate(...)
    revealed: bool                 # 隐藏信息放行门

class Skills:
    @staticmethod
    def build(state, human_pid, engine) -> SkillContext      # 唯一数据入口，遵守隐藏信息红线
    def evaluate_position(ctx, engine) -> dict               # {score, summary, mechanical_text}
    def detect_good_move(ctx, engine) -> dict | None
    def detect_blunder(ctx, engine) -> dict | None
    def suggest_hint(ctx, level, provider, engine) -> dict   # level: direction/specific/demo
    def summarize_result(ctx, engine, winner, player_pid) -> dict
    def explain_illegal(ctx, engine, attempted: dict) -> dict
    def idle_reminder(ctx) -> dict
    def greet(ctx, profile: dict | None) -> dict

# evaluation.py
def evaluate(state, viewer, engine) -> dict      # 通用兜底评估（终局 utility、lastPlacedCell 邻域启发式）

# hidden_guard.py
def assert_no_hidden(ctx: SkillContext) -> None  # 校验 ctx.observation 不含隐藏字段，违规 raise
def scan(text: str, game_id: str) -> str         # 后置泄露令牌扫描，命中改写为通用语

# llm_client.py
class OllamaClient:                               # 最小 urllib 客户端到 ollama /api/chat（L4 本地，可选）
    def complete(self, system: str, user: str, max_tokens: int) -> str
    @staticmethod
    def available() -> bool                       # 探测本地 ollama，不可用返回 False

# dialogue_engine.py
@dataclass
class AgentMessage:
    text: str
    mood: str                                     # happy/thinking/sorry/neutral

class DialogueEngine:
    def __init__(self, persona: Persona, llm: OllamaClient | None = None, *, max_len=100, dedup_window_s=300)
    def reply(self, ctx: SkillContext, scenario: str) -> AgentMessage   # LLM 成文 → 失败回退 persona.fallback_lines
```

**实现要点**：
- `Skills.build` 是红线：只调 `engine.project_observation(state, human_pid)` + `engine.get_legal_actions(state)`，不 import `layer3_solvers`；`suggest_hint` 的 `specific/demo` 通过传入的 `SolverProvider`（`provider.create_solver(...)` 或会话 `SolverHandle`）求解，不直接 import L3。
- `dialogue_engine.reply` 串行：清洗（长度上限默认 100、剔控制字符）→ `hidden_guard.scan` → 去重（`(scenario, persona.key, 状态哈希)` 5 分钟窗口）→ 静音开关（静音返回空 `AgentMessage("", "neutral")`）。
- 每个 `SCENARIOS` 场景 × 每个 `Persona` 都要有 `fallback_lines`（至少 1 条兜底），保证无 LLM 时可离线表达。
- 九场景语义对齐 PRD 4.2.2（greet 可提及昵称/上次战绩、blunder 不嘲笑、illegal 说明原因不责备、idle 先等待后提醒、game_over 总结 + 引导复盘）。

**验收/测试**：
- 单测：每场景×每性格出消息；无 LLM 时回退兜底；长度/控制字符清洗；去重；静音；`assert_no_hidden` 对注入隐藏字段的 context 抛错；`scan` 命中德州底牌记法后改写。
- `ruff check/format` 全绿；`pytest` 本目录用例通过。

---

## C3. 自适应难度 + 偏好记忆后端

**目标**：难度自适应控制器 + 偏好/档案本地存储。

**文件范围**（独占新建）：`layer4_interface/difficulty/__init__.py`、`difficulty/adaptive.py`；`layer4_interface/profile/__init__.py`、`profile/store.py`

**契约接口（冻结）**：

```python
# difficulty/adaptive.py
class AdaptiveController:
    def __init__(self, *, target_lo=0.40, target_hi=0.60, window=10)
    def pick_budget(self, game_id: str, difficulty: str, recent: list[dict]) -> int
        # recent: 最近对局 [{"winner": str|None, "player_pid": str, "difficulty": str}, ...]
        # 胜率<target_lo → 降预算（变简单）；胜率>target_hi → 升预算（变难）；锁定则返回原档预算
    def strength_explain(self, game_id, old_budget, new_budget) -> str   # 变化原因（可选展示）

# profile/store.py
class ProfileStore:
    def __init__(self, root: Path)             # 默认 data/profile.json
    def load(self) -> dict                       # 缺省返回默认 profile
    def save(self, profile: dict) -> None        # 原子写 tmp+os.replace
    def clear(self) -> None                      # 删除文件（一键清除）
```

**profile 默认结构**（约定 schema，C5/集成依赖）：
```json
{
  "nickname": "", "agent_call": "",           // 玩家昵称、Agent 对玩家称呼
  "default_persona": "gentle", "default_difficulty": "normal",
  "hint_level": "off",                        // off/direction/specific/demo
  "pacing": "standard",                       // fast/standard/slow
  "adaptive": false, "difficulty_locked": false,
  "learning_enabled": true, "theme": "light",
  "recent": {"<game_id>": {"wins": 0, "plays": 0}}   // 供开场白/自适应
}
```

**实现要点**：
- 难度档预算表复用 `platform/games.py` 的 `difficulty_budgets`（mcts/hybrid 预算）；节奏映射：`fast/standard/slow` → AI 思考上限 ≤1s/≤5s/≤15s（映射为预算缩放，供集成参考，不在此实现计时）。
- 已知限制：麻将启发式无预算旋钮，`adaptive` 对麻将 P0 返回原档并标注"展示档位"，不改启发式（P1 再补强度参数）。
- 存储线程安全：`threading.Lock` + 原子写（照 `platform/history.py` 的 `_atomic_write` 模式）。

**验收/测试**：单测胜率收敛（构造 10 局 low/high 胜率序列断言预算下调/上调）、锁定返回原档、`save→load` 往返、`clear` 后文件不存在；`ruff/pytest` 全绿。

---

## C4. 复盘后端（`layer4_interface/review/`，全新建）

**目标**：赛后复盘分析（关键节点、胜负手、失误检测、一句改进建议），输入为已存历史记录。

**文件范围**（独占新建）：`layer4_interface/review/__init__.py`、`review/analyzer.py`

**契约接口（冻结）**：

```python
# review/analyzer.py
@dataclass
class KeyNode:
    step: int            # 0-based 步序号
    kind: str            # turning_point | winning_move | blunder
    why: str             # 机械文本（如 "效用跳变最大的一手"）

@dataclass
class ReviewReport:
    key_nodes: list[KeyNode]
    improvement: str     # 一句改进建议（机械事实；成文由 C2 dialogue_engine 按人设改写）
    summary: str         # 胜负/步数摘要

def analyze(match: dict) -> ReviewReport
```

**实现要点**：
- `match` 是 `MatchHistory.get()` 的记录：`{"meta": {...}, "moves": [{"step","actor","action","snapshot"}, ...]}`。
- 关键节点判定用**通用兜底评估**：对每步 snapshot 计算评估值（复用 C2 `agent/evaluation.py` 的 `evaluate`，或终局效用代理），取"评估值跳变最大"的一步为 `turning_point`；`winning_move`=胜方最后一次落子；`blunder`=己方评估值显著下降且随后落败的一步。
- `improvement` 产出机械事实文本（如"第 N 手后优势转为劣势，注意守住角"）；**不得引用隐藏信息**（德州未 showdown 的底牌不可入文本）。
- 只 import L2 `GameEngine`（用于 `eval_expr`）与 C2 的 `evaluation`；不 import L3。

**验收/测试**：构造一个含胜负的月亮棋/德州历史记录，断言 `analyze` 返回 ≥1 个 `key_node` 且 `improvement` 非空；德州用例断言文本不含对手底牌；`ruff/pytest` 全绿。

---

## C5. 前端（`platform-frontend/src/**`）

**目标**：补齐聊天区、Agent 形象、复盘页、设置页、首页、主题，让 P0 三游戏具备"陪伴感"界面。

**文件范围**（独占）：`platform-frontend/src/**`（新增/修改，不碰 `layer4_interface/`）

**新增组件**：
- `components/ChatPanel.tsx`：消息气泡流 + 快捷短语（"再来一局"/"这步为什么？"）+ 输入框；气泡含 `AgentMessage{mood}` 表情。
- `components/AgentAvatar.tsx`：头像 + `mood` 表情（开心/思考/遗憾）+ 思考中动画（复用现有 `spinner` 样式）。
- `pages/HomePage.tsx`：欢迎语（读 profile 昵称/性格）+ "继续上一局"/"最近玩过" + 大厅/个人中心入口。
- `pages/ProfilePage.tsx` + `pages/SettingsPage.tsx`：档案（昵称/称呼）+ 性格卡片（介绍+示例+试听占位）+ 提示级别滑块 + 语音/对话开关 + 主题切换 + 一键清除。
- `pages/ReviewPage.tsx`：升级现有 `ReplayPage` → 左时间线（关键手高亮）/中局面回放/右评语 + "导出报告/再来一局"。

**修改**：
- `components/BattleSetup.tsx`：加性格、提示级别、节奏、自适应开关、规则速览入口（对齐 PRD 4.1.2）；麻将按 `player_counts` 过滤座位（内置六变体默认 4 人 → 显 p0-p3；人数超出时避免选到无效座位）。
- `pages/BattlePage.tsx`：嵌入 `ChatPanel` + `AgentAvatar`，布局对齐 PRD 5.3；"关闭对话/静音"后隐藏聊天区纯对局。
- `types.ts`：新增 `AgentMessage`、`ChatState`、`ReviewReport`、`Profile`、`LearningStatus` 扩展；`MatchMeta` 加 `persona/hinted/ai_strength`。
- `api/client.ts`：新增 `/agent/say`、`/match/hint`、`/profile`、`/profile/clear`、`/review/:id` 调用。
- `styles/global.css`：游戏主题色变量 + 浅/深主题切换。

**契约依赖（按接口，不按实现）**：C5 调用的后端路由（`/agent/say`、`/match/hint`、`/profile`、`/review/:id`）由集成阶段接线，本子任务只按约定的请求/响应 JSON 结构写前端（见 C2/C3/C4 的数据类 → JSON 字段映射）。

**验收**：`npm run build` 通过；TS 类型无错；页面可渲染（用假数据自测即可，不要求后端联通）。

---

## C6. 训练 quick preset + 旧应用退役 + 文档

**目标**：提供演示时长的训练预设；退役旧 `play_*` 独立应用；同步文档。

**文件范围**（独占）：`train-cli/train.py`、`layer4_interface/frontend/play_moon_chess|play_gomoku|play_texas_holdem|play_werewolf/`（归档）、`CLAUDE.md`、`docs/user/*`、`docs/design/architecture.md`

**实现要点**：
1. `train-cli/train.py` 新增 `--preset {full,quick}`（默认 `full`）：`quick` 在运行时按比例缩放注册表读出的 `episodes`/预算（如 gomoku CFR iterations 50→10、mcts budget 打折），**不改 `train-cli/games.py`**（避免与 C1 冲突）；在 `--help`/docstring 注明"默认=完整训练，quick=演示校准"。
2. 退役：把四个 `play_*` 目录移入 `archive/`（`play_texas_holdem` 已知因 `layer2_engine.core.poker_utils` 缺失直接崩，直接退役）；狼人杀聊天流/视觉识别能力后续并入平台（本子任务只归档 + 文档标注，不实现平台侧）。
3. 文档：更新 `CLAUDE.md` 端口表/常用命令（移除 8765/8767/8768/8771，保留 8766 视觉与 8770 平台，标注视觉 P2 并入平台）；更新 `docs/user/*` 指向平台；更新 `docs/design/architecture.md` L4 目录树（新增 `agent/`、`difficulty/`、`profile/`、`review/`，移除 play_*）。

**验收**：`python train-cli/train.py --preset quick --game all`（或至少 gomoku+moon）在演示时长内完成；`python -m layer4_interface.frontend.platform.server` 启动正常；`ruff/pytest` 全绿；文档端口表与实际一致。

---

## D. 集成与验收（执行者本人做，不属于 6 个子任务）

1. 接线：`platform/server.py` 挂 `/api/agent/say`、`/api/match/hint`、`/api/profile`、`/api/profile/clear`、`/api/review/:id`；`platform/session.py` 在 move 后判定九场景、快照附带 `chat` 增量消息与 `evaluation`；`platform/games.py` 补 Agent 钩子（仅 C1 之外的接线改动）；`platform/history.py` 扩展 meta（`persona/hinted/ai_strength`）。
2. 全量 `python -m pytest` + `ruff check/format`。
3. 端到端：月亮棋/随机五子棋/德州 完整玩完→写历史→刷新恢复→复盘生成→关闭 Agent 表达仍可玩；麻将三变种 ×2/4 人可开局；在线学习发布后新对局确实用经验表。

---

以上 6 份子任务文档各自独立、接口已冻结、文件所有权不重叠，可直接交给并行 Agent 执行；C2/C3/C4 的契约签名是它们与 C5、集成阶段的唯一耦合点。需要我把某一份再展开到更细的逐文件/逐函数伪代码粒度，或调整子任务边界（例如把 C2 再拆成 skills 与 dialogue 两个），告诉我即可。