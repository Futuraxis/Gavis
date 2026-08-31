# 非教练陪伴重设计：从啦啦队到对手与群聊（2026-09）

## 0. 背景与一句话结论

当前非教练陪伴的身份是「站在玩家身后的啦啦队」：数据入口是玩家投影
（`Skills.build` → `engine.project_observation(state, human_pid)`，与
`Coach.build` 同构），说话时机只在玩家走子后做一次方阵棋盘评分，终局
播报胜负。它对**二人游戏本该是「你的对手」**、对**多人本该是「一桌群
聊」**，现在两者都不是——所以体验「机械」，且与 coach 模式高度同构
（同一观测入口、同一旁观察视角）。

**结论**：把陪伴从「旁观啦啦队」翻成「座内对手 / 桌友」，需要三处架构
改动 + 一处多人群聊调度：

1. 数据入口：二人非教练走 AI 投影（AI 看自己的牌 + 公共信息 + 玩家公开
   应对），而非玩家投影。
2. 说话时机：AI 行动后也要说话（现 `_record_ai_action` 只记 log，
   `session.py:155,159-160`）；玩家行动后的发言从「评分命中才说」放宽。
3. 红线镜像变体：新增 `adversarial` 扫描模式——AI 自己的牌可讲、玩家
   隐藏牌仍拦，与 `teaching` 模式镜像。
4. 多人群聊：`GameSession.agent`（单数）→ `agents: dict[seat, DialogueEngine]`，
   加发言调度。

本文为设计稿，落代码前需评审；落地分期见 §7。

---

## 1. 当前错位（为什么「更像 coach 模式」）

| # | 错位 | 位置 | 后果 |
|---|------|------|------|
| C1 | **数据入口同构**：非教练陪伴与 coach 共用玩家投影 | `skills.py:53` vs `coach.py:89` | 陪伴看不到 AI 的牌，无法以对手身份说话，只能当啦啦队 |
| C2 | **说话时机单边**：AI 行动后陪伴沉默 | `session.py:521-523`（`last.actor != "human"` 即 return）+ `_record_ai_action:159-160` 只记 log | 对手「刚决策完」最该反应的时刻是哑的 |
| C3 | **红线方向是啦啦队视角**：默认 scan 把「我的/你的/对手的 底牌」全拦 | `hidden_guard.py:58-62`（`_TEXAS_PATTERNS`） | 即便喂 AI 投影给 LLM，它一说自己的牌就被改写成「这把牌先不细说」 |
| C4 | **评估只认方阵棋盘**：牌/麻将/社交游戏 score 恒 0.0 | `evaluation.py:51-79` | good_move/blunder 对 5 个卖点游戏永不触发，玩家走子后那段空转 |
| C5 | **单 persona 单触发**：会话只挂一个 `agent: DialogueEngine` | `session.py:80,84` | 多人游戏没有「一桌群聊」的结构 |

C1+C2+C3 是身份错位的根因（陪伴被设计成「看不到对手牌的旁观者」）；
C4 是机械感的放大器（连唯一触发点都常空转）；C5 是群聊的结构缺口。
coach 模式之所以「已对」，是因为它**本就**是旁观者身份（教练站玩家身
后讲玩家的牌），数据入口与红线方向（放行玩家牌、拦 AI 牌）自洽——非
教练陪伴套用同一套入口却要扮演对手，自然拧巴。

---

## 2. 设计目标：按场景的陪伴身份

| 场景 | 陪伴身份 | 数据入口 | 红线模式 |
|------|----------|----------|----------|
| 二人非教练 | **AI 对手** | AI 投影 `project_observation(state, ai_pid)` | `adversarial`（镜像 teaching） |
| 多人非教练 | **桌友群聊** | 各 AI 座位自己的投影 | `adversarial`（各 AI 讲自己的牌） |
| 教练模式（teaching） | 教练（不变） | 玩家投影 | `teaching`（不变） |
| 终局 reveal 后 | 任意 | 揭底后双方牌公开 | `revealed`（全放行） |

身份切换由 `GameSession` 按座位数 + `teaching` 标志决定，不暴露给前端
协议（前端只看 `chat` 增量里多出的 `speaker` 字段）。

---

## 3. 二人对手模式

### 3.1 数据入口：AI 投影而非玩家投影

新增 `OpponentContext`（类比 `TeachContext`，`agent/opponent.py`）：

- `observation = engine.project_observation(state, ai_pid)` —— AI 自己的
  投影：含 AI 自己的手牌视图 + 公共牌 + 玩家公开动作序列。**不含**玩家
  底牌（visibility 规则本就不给 AI 玩家底牌）。
- `ai_hand`：从 AI 投影的私有视图提取（复用 `coach.extract_hand` 的视图
  命名约定，`coach.py:155-180`）。
- `player_actions`：玩家本局公开动作序列（从 `session.log` 过滤
  `actor == "human"`），供「读人」。
- 经 `assert_no_hidden` 校验（AI 投影里本就不该有玩家隐藏字段，守卫仍
  成立——它拦的是 ground-array 键名，AI 自己的视图名不在黑名单）。

`PlayManager._say` / `_chat_after_move` 在二人非教练时改用 `OpponentContext.build`，
与 `_agent_factory` 装配的 DialogueEngine 配套。

### 3.2 说话时机：双触发

现 `step`（`session.py:136-157`）流程：人类 `apply_human` → `run_ai` →
（teaching 算 pending_teach）→ 返回 `move` → `_chat_after_move`。对手模
式需在两处发声：

- **AI 行动后**：`_record_ai_action` 回调（现只 append log）扩展为同时
  队列一句对手反应（场景 `opp_react`）——AI 刚做了决策、刚看到玩家怎么
  应对，这是对手视角最自然的发声点。
- **玩家行动后**：`_chat_after_move` 非教练分支不再要求 `last.actor ==
  "human"` 才说话，且不再强依赖 `detect_good_move/blunder` 命中——对手
  读人（场景 `opp_read`）可在任意玩家行动后发声（按人设分寸 + 去重窗口
  节制频率）。

去重窗口（`dialogue_engine.py:125-133`，5 分钟）天然防刷屏，复用即可。

### 3.3 红线镜像变体（adversarial scan）

`hidden_guard.scan` 现有两态：默认（全拦）+ teaching（放行玩家牌、拦 AI
牌）。新增第三态 `adversarial`，与 teaching **镜像**：

| 模式 | 自称 | 「你」指 | 可讲 | 仍拦 |
|------|------|----------|------|------|
| default | — | — | 无牌面 | 所有牌面 |
| teaching | 教练「我」 | 玩家 | 玩家的牌 | AI/对手的牌 |
| **adversarial** | AI 对手「我」 | 玩家 | **AI 自己的牌** | **玩家的隐藏牌** |
| revealed | — | — | 双方牌（showdown 揭底后） | 无 |

实现：新增 `_ADVERSARIAL_*_PATTERNS`，拦「玩家的所有格 + 牌面」
（`_PLAYER_POSSESSIVE` = `(?:你的|玩家的?)` + 底牌/手牌/牌面记法），
放行「我的/AI 的 + 牌面」。复用现有 `_OPPONENT_POSSESSIVE`
（`hidden_guard.py:92`）做对称定义。`scan(text, game_id, *,
teaching=False, adversarial=False, revealed=False)` 四态互斥。

### 3.4 小心思分寸：读人 ≠ 偷牌

用户要的「按人设加入玩家的小心思」= AI 对手基于**公开动作序列**对玩家
意图的合理推断（吐槽型「你又虚张声势了」、温柔型「你跟得有点犹豫，是
不是没牌？」、高冷型「你节奏乱了」）。这与「偷看玩家底牌」是两回事：

- 读人基于公开下注/弃牌/摸打序列，**红线允许**——真人对手也会这么猜。
- 偷牌是直接报玩家未公开牌面（「你手里其实有 A」），**红线拦**——
  adversarial scan 改写为「这把牌先不细说」。
- AI 对手**本来就没有**玩家底牌（其投影不含），scan 只是双保险防 LLM
  幻觉编造。

人设分寸落在 prompt 与 `fallback_lines`：
- `gentle` 对手：点到为止的温和读牌，不戳穿。
- `banter` 对手：夸张识破，损着玩但不报牌。
- `cold` 对手：只点关键节奏，惜字如金。
- `teacher` 在非教练二人局可作为「认真型对手」（讲自己思路而非教学）。

终局 showdown 前：AI 可讲自己牌、读玩家公开行为，不报玩家牌；
showdown 后（`revealed=True`，`session.py:92` 已有此标记）：双方牌公开，
全放行，可做完整复盘式对手点评。

---

## 4. 多人群聊模式

### 4.1 多座位 DialogueEngine

`GameSession.agent: DialogueEngine | None`（单数，`session.py:84`）→
`agents: dict[str, DialogueEngine]`（按 AI 座位）。二人对手模式是其特例
（单元素 dict，`agents = {ai_pid: engine}`），结构统一。

- 装配：`PlayManager.start` 遍历 `spec.seat_options - {player_pid}`，
  每个 AI 座位一个 `DialogueEngine`（可同人设可异人设，由配置/随机）。
- `pending_chat` 每条新增 `speaker`（座位 pid + 显示名，如「下家 p2」）。
- 前端 ChatPanel 按 `speaker` 渲染不同头像/名字（现单气泡需扩展）。

### 4.2 发言调度

群聊要解决「谁说、什么时候说、不抢话」：

- **轮转优先**：当前行动者行动后优先发言（它刚决策）。
- **不连续两 seat 抢话**：同一 `_chat_after_move` 内同 seat 不连续两
  条；跨 seat 可接力（像桌边对话）。
- **随机插话**：非行动 seat 按概率（如 20%）插一句旁观点评，模拟桌
  边闲聊——但受去重窗口节制。
- **频率上限**：单次快照 `chat` 增量不超过 N 条（防刷屏），超出的留
  `pending_chat` 下次快照再投递。

调度是确定性 + 概率的混合，落在 `_chat_after_move` 的多人分支。

### 4.3 桌位身份与人设

麻将/UNO 有自然桌位语义（上家/下家/对家），狼人杀/卧底本就是发言游戏
（社交族最契合群聊）。人设可按桌位分配：
- 同人设：一桌性格统一（如全 `gentle` 友善局）。
- 异人设：混合更有戏（一桌 `banter` + `cold` + `teacher`）。
- 配置：`agent_factory` 扩展为 `agent_factory(seat, persona_hint)`，
  由 `PlayManager` 按桌位 + 随机/配置装配。

社交族（狼人杀/卧底）的「发言」本就是游戏机制（`text` 参数预制能力，
见 CLAUDE.md），群聊陪伴与游戏内发言需区分通道（陪伴是桌边闲聊、不影
响游戏推进；游戏内发言是规则动作）。

---

## 5. 改动点清单（按文件）

| 文件 | 改动 | 性质 |
|------|------|------|
| `layer4_interface/agent/opponent.py` | **新增** `OpponentContext` + `Opponent.build`（AI 投影入口、`ai_hand`、`player_actions`） | 新文件，类比 `coach.py` |
| `layer4_interface/agent/hidden_guard.py` | 新增 `_ADVERSARIAL_*_PATTERNS` + `scan(..., adversarial=, revealed=)` 多态 | 扩展，与 teaching 镜像 |
| `layer4_interface/agent/scenarios.py` | 追加 `opp_react`（AI 行动后）/ `opp_read`（玩家行动后读人）/ `opp_taunt`（按人设小心思） | 追加，既有键名顺序不变 |
| `layer4_interface/agent/persona.py` | 四人设补 `opp_react/opp_read/opp_taunt` 的 `fallback_lines` | 扩展兜底台词 |
| `layer4_interface/agent/dialogue_engine.py` | `reply` 支持 `adversarial/revealed` scan 标记 + speaker 字段；`_scenario_payload` 增对手事实（`ai_hand`、`player_actions`） | 扩展 |
| `layer4_interface/frontend/platform/session.py` | `GameSession.agent`→`agents: dict`; `pending_chat` 条加 `speaker`; `step` 的 `_record_ai_action` 触发对手说话; `_chat_after_move` 非教练双触发 + 多人调度 | 核心接线 |
| `layer4_interface/agent/__init__.py` | 导出 `Opponent`, `OpponentContext` | 注册 |
| `platform-frontend/src/chat/*` | ChatPanel 按 `speaker` 渲染多气泡/头像 | 前端契约扩展 |
| `layer4_interface/agent/evaluation.py` | （可选，分期）为牌类引入公开特征评估，缓解 C4 | 见 §7 Phase 3 |

coach 模式零改动（其数据入口与红线已自洽）；默认非二人非教练的旧行为
（啦啦队 + greet + 终局播报）作为 fallback 保留，避免破坏现有测试。

---

## 6. 红线与安全考量

- **adversarial 不削弱红线**：AI 对手讲自己的牌是其本分（它本就看得到
  自己的牌），不向玩家泄露**新的**隐藏信息；玩家底牌仍由 visibility
  规则 + adversarial scan 双保险拦住。红线只从「全拦」变为「定向放行
  AI 自己的牌」，方向与 teaching 镜像、非新发明。
- **读人不越界**：基于公开动作的意图推断是允许的；直接报玩家未公开牌
  面（哪怕 LLM 幻觉编造）仍被 adversarial scan 改写。
- **revealed 通道**：终局 showdown 后双方牌公开，全放行——这是现有
  `revealed` 标记（`session.py:92`、`coach.py:92`）的自然延伸，不是
  新开闸门。
- **不污染在线学习**：对手说话是表达层，不影响 AI 决策轨迹；`raw_solver`
  仍只在 coach 通道用（`session.py:99`），对手模式不碰训练数据。
- **多人发言不抢游戏推进**：群聊陪伴是桌边闲聊通道，社交族的「游戏内
  发言」（`text` 参数）是规则动作，两者分离，陪伴不替玩家发言。

---

## 7. 落地分期

| Phase | 范围 | 验证目标 |
|-------|------|----------|
| **P1** | 二人德州对手模式 PoC | `OpponentContext` + 双触发 + adversarial scan + 对手兜底台词。验证：AI 走子后说一句体现「看了自己牌 + 玩家应对」；showdown 前不报玩家牌、showdown 后可揭底；adversarial scan 拦「你的底牌是 X」、放行「我一对 K」 |
| **P2** | 多人群聊（麻将 4 人） | `agents: dict` + 发言调度 + 桌位身份。验证：不连续两 seat 抢话、有桌边对话感、单快照 chat 增量不刷屏 |
| **P3** | 评估层扩到牌类（缓解 C4） | 德州 pot odds / 麻将听牌数等公开启发式，让 `opp_read` 有具体素材而非恒「胶着」 |
| **P4** | 扩到 UNO / 社交族（狼人杀/卧底） | 群聊 + 游戏内发言通道分离 |

P1 先做，跑通「对手反应有没有味」再扩；P3 与 P1 可并行（评估素材直接
喂给 `OpponentContext` 的 `player_actions` / 公开特征）。

---

## 8. 验证（测试要点）

- **红线**：adversarial 模式下，构造 LLM 回复「你底牌是 ♠A」→ scan 改
  写为「这把牌先不细说」；回复「我手里一对 K」→ 原样保留。
- **数据入口**：`OpponentContext.observation` 经 `assert_no_hidden` 不
  抛（AI 投影不含玩家隐藏字段）；`ai_hand` 取到 AI 自己的牌、`player_actions`
  为玩家公开动作序列。
- **说话时机**：AI 行动后 `pending_chat` 出现 `opp_react` 一条；玩家行
  动后出现 `opp_read`（不依赖 score 命中）。
- **二人特例**：`agents` 单元素 dict，行为与多人调度一致。
- **群聊调度**：4 人麻将单次快照 `chat` 增量 ≤ N，无同 seat 连续两条。
- **coach 不回归**：`teaching=True` 路径零改动，既有 `test_teaching.py`
  全绿。

---

## 9. reasoning 展示约束（调试模式，已落地）

思维链（reasoning）是模型思考过程。**后端照常产出 / 透传 / 存档**
（`dialogue_engine.py` 的 `AgentMessage.reasoning`、`pending_chat` 条目、
`conversations` 归档、流式 SSE `reasoning` 事件均保留），但**前端默认不
展示**——避免把模型思考过程暴露给玩家，影响沉浸感与对手身份的代入。

实现（已落地，独立于对手重设计、当前啦啦队模式即生效）：

- 前端新增「调试模式」开关：`platform-frontend/src/settings.ts` 加
  `getStoredDebug/setStoredDebug`（localStorage `gavis.debug`，默认 off）。
- 设置页「开关」面板提供切换（`SettingsPage.tsx`，与主题/对话/语音同列）。
- `MessageBubble.tsx` 的 reasoning 折叠块由 `{msg.reasoning && getStoredDebug()}`
  守门——这是前端**唯一**的 reasoning 渲染点（ChatPage 与对局页陪伴消息
  气泡都经它渲染；`snapshotChat.ts` 把 chat 增量的 reasoning 带进
  ChatMessage、`useChatRuntime.ts` 流式累积 reasoning，这些透传/累积链路
  **不动**，仅渲染受开关节制）。

约束适用范围：所有陪伴身份（啦啦队 / 对手 / 群聊多气泡）。对手重设计的
多座位 `agents` 各自的 reasoning 同样默认隐藏、调试模式才展示——便于开
发期排查成文质量，不污染玩家体验。

后端不动：reasoning 仍是 `AgentMessage` 的一等字段、仍进 `pending_chat`、
仍随对话存档——调试价值（日志 / 回放 / 存档检索）完整保留。这是「展示
层约束」，与隐藏信息安全红线正交：红线管「不该说什么」，调试开关管
「思考过程要不要给玩家看」。
