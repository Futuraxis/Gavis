# 前端契约与信封约定（经验教训沉淀 2026-08）

> 背景：平台首页整页崩溃（`Cannot read properties of undefined (reading 'moon_chess')`）
> 与 `/api/match/hint`、`/api/agent/say` 连续 400，根因都在**前端 API 客户端
> 与后端 JSON 契约不一致**。本文沉淀约定、教训与回归防线，供后续接线/改契约时对照。

## 一、后端统一信封约定

平台后端所有业务接口都返回 `{"ok": <bool>, <key>: <payload>}` 信封：

| 接口 | key | 说明 |
|------|-----|------|
| `GET  /api/profile` | `profile` | 档案对象（含 `recent`） |
| `PUT  /api/profile` | `profile` | 请求体必须是 `{"profile": {...}}`（后端读 `payload["profile"]`） |
| `POST /api/profile/clear` | `profile` | 清空后的档案 |
| `GET  /api/games` | `games` | 游戏列表 |
| `GET  /api/match/active` | `sessions` | 活跃对局 |
| `GET  /api/history` | `matches, total, has_more` | 对局记录（平台「对局记录」页数据源）。查询参数全部可选：`game_id`、`result`（`win`/`lose`/`draw`，玩家视角，口径 = `layer4_interface/result.player_won`）、`q`（关键字，匹配 game_id 与座位称呼）、`since`（ISO 日期前缀，含当天）、`limit`（默认 100，clamp 1..500）、`offset`（默认 0）。**脏参数宽容解析**（`?limit=abc` 回落到默认值，不 400/500）。`matches[]` 为 meta 列表，含 `seat_names`（按 game_id 现算注入）与回填的 `won`（旧记录用末手快照 `final_roles`/`winners` 解析），元数据字段有 `seed`/`family`/`custom`/`player_count`/`variant`（旧记录缺省 `null` —— 前端必须按「缺就不显示」渲染）。`total` 是筛选后的总数（分页切片之前），`has_more = offset+len(matches) < total` |
| `GET  /api/history/:id` | `match` | 单局全量记录（含 `moves` 明细 + `seat_names`），复盘/回放页用 |
| `POST /api/match/start|move|state` | `session` | 对局快照 |
| `POST /api/agent/say` | `message` | `{scenario, text, mood}` |
| `POST /api/match/hint` | `hint` | `{level, direction, mechanical_text, hint}` |
| `POST /api/chat` | `intent, text, mood, params` | 请求体 `{text, game_id?, history?}`；`history` 为之前若干轮 `{role: user\|assistant, content}`（最新在后），后端清洗并限长（24 条 / 6000 字符），system 由后端现构。`params` 随意图而定：`play` = `{game_id, config?, config_ignored?}`（`config` = 已按注册表校验的玩家偏好，前端不再有开局配置卡；`config_ignored` = 该游戏不支持、被丢弃的偏好项，前端必须在开局提示里说明）；`create` = `{game_id?, game?, ...}`（`game` = 已落盘的注册表条目，前端刷新目录 + 提示「玩X」；**无 `game`** 时前端打开平台创建游戏页） |

**前端铁律：客户端函数必须解包命名 key，绝不允许把整个信封当业务对象存。**
`apiGet<T>` 返回的是 `T & {ok}` 信封，业务类型应写成 `{ <key>: T }` 再解包。

## 二、两个参数语义要分清

- `game_id`（路由/注册）：`/battle/<game_id>`、`/api/match/start` 用，指**游戏注册 id**（如 `moon_chess`）。
- `session.game_id`（会话 id）：`/api/match/move|state|hint`、`/api/agent/say` 用，指**对局会话 id**（uuid4().hex[:8]，`?game=<id>` 恢复）。

`agentSay` / `matchHint` 的后端参数是**会话 id** —— 从 `session.game_id` 取，
不是路由上的游戏 id。缺了就 400 `参数错误: 'game_id'`。

## 三、教训清单

1. **类型伪装成契约**：`getProfile(): Promise<Profile>` 返回了 `{ok, profile}`，
   `profile.recent` 为 `undefined` 而 TS 无感知 —— 类型只描述了期望，没验证运行时。
   修复：客户端解包（`apiGet<{profile}>().then(d => d.profile)`）。
2. **模式不一致**：`/games`、`/match/active` 的调用点都正确解包，只有
   `getProfile`/`agentSay`/`matchHint` 漏了 —— 同层不同规矩，逐个翻车。
   修复：统一在 **api/client.ts 内**解包，调用方永远拿到业务对象。
3. **防御缺位**：页面直接 `profile.recent[g.game_id]`，字段缺失即整页崩。
   修复：`recentOf(profile)`（`src/profile.ts`）单一可信兜底，页面不再各自防御。
4. **后端契约测试完备但前端空白**：`tests/test_platform_server.py` 已锁定信封
   形状，但没人测前端客户端是否按契约发请求/解包。修复：新增
   `platform-frontend/tests/client.contract.test.ts`（零依赖 node:test）。
5. **静默降级掩盖问题**：`saveProfile` 请求体没包 `{profile}` → 后端 400 →
   前端 catch 分支提示"已保存（本地）"，看起来"成功"实则没持久化。
   修复：请求体套 `{profile}` 并新增契约断言。

## 四、回归防线（已落地）

- `cd platform-frontend && npm run test:frontend` — 8 个契约用例：
  `getProfile`/`saveProfile` 解包、`agentSay`/`matchHint` 请求体携带会话 id +
  解包、`ok=false`/网络失败/非 JSON 抛 `ApiError`、`recentOf` 兜底。
- `python -m pytest tests/test_layer4_interface/test_platform_server.py` —
  后端侧信封形状（`TestCompanionIntegration` 等）已锁定。
- 改契约时的检查顺序：先改后端 → 后端测试 → 前端 client 契约测试 → 页面。

## 五、约定速查（新端点接线时）

1. 响应一律 `{"ok": ..., "<key>": ...}`，key 用复数/单数语义清晰的字段名。
2. 客户端函数签名：请求所需参数全显式（如 `agentSay(gameId, scenario, extra)`），
   内部 `apiPost<{key}>('/path', {...}).then(d => d.key)` 解包。
3. 会话级接口传 `session.game_id`；注册级接口传路由 `game_id`。
4. 失败处理：页面 `catch` 落本地兜底文案，客户端只抛 `ApiError`。
5. 页面读取档案字段一律走 `recentOf` 等单一兜底函数，不做内联 `?? {}`。

## 六、慢操作必须给进度 / 不阻塞（2026-09 教训）

「用户看不到东西 + 玩起来卡」是同一类问题的两种形态 —— 慢操作要么给进度，
要么挪出关键路径：

- **创建自定义游戏**（`POST /api/custom/games`）：LLM 翻译要跑 1-3 分钟，
  故支持 `?stream=1` / `Accept: text/event-stream`：`stage{stage,detail}` →
  `result` / `error` → `done`；创建页显示阶段文案 + 已等待秒数。失败绝不
  空手而归：端点预检不可达 / 翻译失败 → 确定性模板兜底，并在
  `llm_fallback` + `validation.warnings` 里醒目告警（前端必须展示）。
- **陪伴 Agent 发言**（`session.GameSession.queue_agent_reply`）：平台以
  `agent_async=True` 构造 `PlayManager`，成文在后台线程；`/match/start` 与
  `/match/move` 不再等 LLM（实测每步 16-33s → 0.4-4s）。代价是发言**不在**
  响应里，前端必须轮询 `POST /api/match/state`（2.5s，`busy` 期间跳过）把
  快照增量 `chat` 落进对话流；不轮询就会「AI 不说话」。
  - 例外：终局播报同步生成（会话随后从注册表移除，异步消息没人取）。
  - 默认 `agent_async=False` 保持同步契约（测试与非平台调用方）。
- **开局第一帧必须在 AI 思考之前**（`PlayManager.start` 的 `on_progress`）：
  流式开局原来只有「AI 每走一步一帧」，AI 先手的棋类（或大棋盘搜索慢的对局）
  在第一手算完前前端拿不到任何快照 → 对话页没有 `session`，棋盘区不渲染，
  用户看到的就是「说了玩X，界面调不出来」。现在先推一帧空局面（实测 0.25s）。
- **自定义游戏 + `adaptive=true` 不得炸开局**：对话开局的表单初值是
  `adaptive: true`（`battleConfig.ts`），而 `AdaptiveController` 的预算表只
  登记内置 `GAMES`，对自定义 id 抛 `ValueError("未知游戏: …")` →
  `/match/start` 直接失败。会话层 `_pick_budget` 已回落到该 `GameSpec` 自己的
  normal 档（前端无需改动）。
- **AI 搜索要有时间闸**：迭代预算管不住墙钟（成本 ∝ 棋盘面积 × 迭代数）。
  自定义网格游戏按难度上限 1.5s / 3s / 6s（`families/grid.py`
  `MOVE_TIME_LIMIT_S`，与迭代预算先到先停）。