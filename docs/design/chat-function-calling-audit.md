# Chat 链路 Function Calling 排查报告（2026-09）

以「月亮棋是什么？」会产生大幻觉为出发点，对平台对话链路的 function calling
能力做了系统排查。结论：**工具集全部是动作类、且架构上不存在"执行工具并把
结果回传给模型"的循环**，知识类问题只能靠模型参数记忆作答 —— 而平台 15 款
游戏（月亮棋 / 随机五子棋 / 六变种麻将 / 六变体 UNO…）大多是自命名规则，
通用语料里根本没有，幻觉是必然的。本次已修复核心缺口（见 §4）。

## 1. 症状复现

用户在 ChatPage 输入「月亮棋是什么？」：

```
ChatPage → POST /api/chat → chat_turn()
  ├─ LLM 可用：build_tools() 给出 13 个动作类工具（play_game/make_move/…），
  │   system prompt 只有「可用游戏: 月亮棋(moon_chess)、…」—— 光秃秃的名字，
  │   没有任何描述。模型无工具可调 → 直接用参数知识生成 reply.text
  │   → intent="chat" 原样返回 → 幻觉内容渲染给用户。
  └─ LLM 不可用：fallback_intent() 正则全部不命中
      → 「我在的，你可以试试：玩月亮棋…」—— 不幻觉，但也零信息。
```

讽刺的是：权威资料一直就在同一进程里 —— `GameSpec.description`（每款一句
中文简介）、`rules/*.json` 的 `meta.description`、`docs/user/play_*.md`
（完整玩法说明）、`/api/games` 端点（前端大厅就在用，chat 后端自己不用）。

## 2. 系统性缺口清单

按严重度分级（P0 = 直接造成幻觉的根因）：

| # | 级别 | 缺口 | 位置 |
|---|------|------|------|
| R1 | P0 | **零信息查询工具**：13 个工具全是动作（play/resume/move/hint/restart/history/review/create/settings/platform/benchmark/learning/help），没有 describe_game / list_games 任何只读查询 | `chat.py build_tools` |
| R2 | P0 | **无 tool-result 回传循环**：`chat_turn` 只取 `reply.tool_calls[0]` 直接映射 intent，从不执行工具、从不构造 `role:"tool"` 消息 —— 即使加了查询工具，模型也拿不到结果来组织回答。OpenAI function calling 的完整闭环（请求调用→执行→`role:"tool"` 回传→基于结果作答）缺失 | `chat.py chat_turn` |
| R3 | P1 | **上下文注入丢字段**：`_collect_games` 收集了 game_id/display_name/kind/family，唯独丢掉 `spec.description` —— system prompt 里模型连一句正确的游戏简介都看不到 | `chat.py _collect_games` |
| R4 | P1 | **反幻觉红线不对称**：system prompt 规则 3 只防对局隐藏信息（手牌/身份），没有「平台知识必须来自工具/上下文，不得编造」的红线 | `chat.py _system_prompt` |
| R5 | P1 | **兜底双路径无知识回答**：后端 `fallback_intent` 与前端 `classifyLocal`（断连兜底）都不处理知识问句 —— 尽管 description 是确定性数据 | `chat.py` / `intents.ts` |
| R6 | P2 | **intent 契约无知识通道**：`chat` 兜底 intent 无结构化参数（game_id / chips），前端无法对知识回答渲染「来一局」快捷动作 | 前后端契约 |
| R7 | P2 | **并行 tool_calls 丢弃**：`tool_calls[0]` 之外静默忽略（"介绍一下血战到底然后来一局"这类复合请求无法一次表达） | `chat.py chat_turn` |
| R8 | P2 | **ToolCall 无 id**：OpenAI 协议要求 `role:"tool"` 消息用 `tool_call_id` 回指发起调用的 assistant 消息；`LLMClient` 解析时把端点给的 id 丢弃了，后端永远无法构造合规的多轮回传 | `llm.py` |
| R9 | P2 | **其他 LLM 消费点**：`agent/dialogue_engine.py`（陪伴聊天 `complete_chat` 无工具、无游戏知识注入，提到游戏时同样会幻觉）。对照：`layer1_translator`（结构化输出 + schema 校验，非 FC 场景）与 `layer3 social/llm_policy`（游戏内发言，无工具需求）合理 | `layer4_interface/agent` |

注：项目在「对局隐藏信息」上防御非常严格（poker reveal-gate、UNO 手牌只露
张数、visibility 投影），但「静态知识」这条线此前完全裸奔 —— 两条红线应当
同样硬。

## 3. 修复设计（本次实施）

核心思路：**知识类问题 = 工具取数 + 模型转述**，两层保障：

1. **LLM 路径（agentic loop）**：新增只读信息工具 `describe_game(game_id)` /
   `list_games()`，后端就地执行，结果以 `role:"tool"` 消息回传，模型基于
   权威资料作答；循环有界（`_MAX_TOOL_ROUNDS = 3`），预算耗尽时直接用工具
   执行结果作答（确定性、零幻觉兜底）。
2. **无 LLM 路径（确定性回答）**：`_WHAT_IS_RE`（是什么/怎么玩/规则/玩法…）
   + `_find_game` → 注册表 description + `docs/user/play_*.md` 规则段直接
   拼出回答；不点名任何已注册游戏则维持原兜底（不猜、不编）。

数据源（单一事实来源，全部 fail-soft）：

- `GameSpec.description` / custom entry `description` —— 一句话简介；
- `docs/user/play_*.md` 的「游戏规则 / 基础规则 / 六种变体」段 —— `_DOCS_RULES_SECTIONS`
  映射 + 正则提取 + 缓存 + 900 字截断；读取失败回退空串，只拼 description；
- `spec.player_counts` / 难度档等注册表元数据。

配套改动：

- **`LLMClient.ToolCall` 增加 `id`**（默认 `""`，向后兼容）并解析保留，
  使 `role:"tool"` 回传能合规关联；
- **`_collect_games` 保留 `description`**，system prompt 游戏目录逐条带简介，
  并新增「知识红线」规则（先调 describe_game/list_games，资料没有的不编造）；
- 动作类工具行为完全不变（一击映射 + 校验 fail-soft）。

## 4. 改动清单

| 文件 | 改动 |
|------|------|
| `layer2_engine/core/llm.py` | `ToolCall.id` 字段 + 解析保留；`complete_tools`/`_chat` 的 messages 注解放宽为 `list[dict[str, Any]]`（允许 assistant+tool_calls / role:"tool" 消息） |
| `layer4_interface/frontend/platform/chat.py` | `_collect_games` 带 description；`_game_rules_text`（docs 规则段提取，缓存）；`_game_brief`；system prompt 目录+知识红线；`build_tools` 注册 describe_game/list_games；`_execute_info_tool` 执行器；`chat_turn` 有界 tool loop + 预算耗尽确定性兜底；`fallback_intent` WHAT_IS 分支（先于 play —— "怎么下"含开局动词但语义是问规则） |
| `tests/test_layer2_engine/test_llm_client.py` | tool_call id 解析保留测试 |
| `tests/test_layer4_interface/test_chat.py` | `_ScriptedLLM`（多轮脚本化假 LLM）；info 工具循环 / list_games / 预算耗尽 / system 简介注入 / fallback 知识回答（含"怎么下"优先于 play、未点名游戏不猜）共 9 个新用例 |

验证：`tests/test_layer4_interface/` + `test_llm_client.py` 共 44 例全过；
全量 layer4 相关 395 passed / 8 skipped（既有 skip）；ruff format + check 通过；
无 LLM 冒烟输出（节选）：

```
[no-llm] chat
3×3 经典月亮棋：三子连珠即胜，棋盘满时最旧的棋子被挤出。
- 棋盘：3×3，共 9 格
- 棋子：每方最多 3 子，落第 4 子时最老的棋子被驱逐（FIFO）
- 获胜：任意一方三子连成一线（横/竖/斜）即胜
- 平局：50 手内无人三连，判平局
想试一试的话，说"玩月亮棋"即可开局。
params: {'game_id': 'moon_chess', 'chips': ['玩月亮棋']}
```

## 5. 遗留与后续建议（P2）

1. ~~**前端渲染知识回答的快捷动作**~~（已实施，2026-09 第二轮）：后端已在
   `params` 里返回 `game_id` + `chips`（["玩月亮棋"]），
   `useChatRuntime.dispatch` 的 `case 'chat'` 不再丢弃 params（白名单
   透传 game_id/chips），`MessageBubble` 对 chat intent 复用 clarify 的
   `Chips` 组件渲染；已随 `npm run build` 重建 dist。
2. ~~**`classifyLocal`（前端断连兜底）补同样的信息分支**~~（已实施）：
   `intents.ts` 新增 `WHAT_IS_RE` 分支（与后端 `_WHAT_IS_RE` 对齐、
   先于 play——"怎么下"含开局动词但语义是问规则），从游戏目录
   `description` 拼确定性回答并带 `chips`；不点名游戏维持原兜底
   （不猜、不编）。
3. ~~**并行 tool_calls**~~（已实施）：`chat_turn` 的 loop 不再只看
   `tool_calls[0]`——信息类调用逐个就地执行、逐个以 `role:"tool"`
   回传（按 `tool_call_id` 成对关联，合成 id 全局递增）；动作类只取
   首个，混合批次动作优先（单动作 intent 契约）。
4. ~~**陪伴聊天（`agent/dialogue_engine.py`）知识注入**~~（已实施）：
   `reply` 新增 `game_id` 关键字参数（平台 session 调用点已传
   `session.game_id`），成文时把权威资料注入 user prompt 并在 system
   prompt 立红线（提到玩法只依据资料、资料没有的不编造）。资料拼装
   抽取为 `frontend/platform/game_knowledge.py`（`game_knowledge_text`）
   ——chat 信息工具、无 LLM 兜底与陪伴注入三方共用的单一事实来源
   （只依赖 games.py + 标准库，无循环依赖）。
5. ~~**短名匹配**~~（已实施）：`GAME_ALIASES` 别名表（随
   `game_knowledge.py` 维护，`/api/games` 已暴露 `aliases` 字段）；
   后端 `_find_game` 与前端 `findGame` 同步升级为别名 + 大小写不敏感
   子串匹配，最长匹配胜出（"UNO 7-0" 优先于裸 "UNO"）——play 分支
   同样受益（"来一局德扑"/"玩uno" 都能开局）。
6. **狼人杀 / 谁是卧底**：rules + train-cli 已登记，但平台 GAMES 未注册
   （`docs/user/play_undercover.md` 存在）—— 若上平台，
   `DOCS_RULES_SECTIONS`（现位于 `game_knowledge.py`）需同步补映射。
