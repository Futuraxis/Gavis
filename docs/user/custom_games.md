# 自定义游戏 · 使用说明（Layer 1 纳入平台工作流）

玩家可以在平台里用**自然语言规则**或**已有游戏的变体（模板+改动描述）**直接
生成可对弈的新游戏——这是 Layer 1（规则翻译）接入对弈工作流后的主入口。
翻译 → 校验 → 规则族识别 → 注册 → 大厅出现 → 正常人机对弈，全链路自动完成。

## 快速开始

```bash
python -m layer4_interface.frontend.platform.server   # 8770（需先 npm run build）
```

浏览器打开 **http://127.0.0.1:8770/** → 顶部导航「创建游戏」（或直接进 /create）：

**入口一（对话里，最省事）**：在对话页直接说规则，例如
「做一个 8×8 四子连珠的游戏，叫四子棋」或「基于随机五子棋改成 15×15 连五」——
助手通过 `create_game` 工具**在对话内**完成翻译+校验+落盘（默认走确定性模板翻译，
快），回复给出创建结果，随后一句「玩四子棋」即可开局。只说「创建一个新游戏」而
没描述规则时助手会追问，不会替你编规则。需要 LLM 翻译、要管理/删除已有自定义
游戏时，进下面的创建页。

**入口二（创建游戏页）**：

1. **模式一：规则描述（from_scratch）**——在文本框里用一句话描述规则，
   例如「8×8 棋盘，四子连珠获胜，黑棋先手」。
2. **模式二：基于模板变体（variant）**——选一个基础游戏（月亮棋/随机五子棋/
   德州扑克/麻将/狼人杀/谁是卧底），再描述要做的改动，例如「棋盘改成 7×7，
   五子连珠获胜，每步落子后 30% 概率抹去一格」。变体翻译走确定性参数路径
   （能安全消费的参数直接应用）；复杂改动走 LLM 全模板改写或 v5.5 增量补丁
   修复循环，输出必过 `engine_validator`（schema + L2 冒烟，对所有声明变体
   boot）。
3. 可选：游戏名称、LLM 翻译开关（勾选后**优先用 LLM 翻译**；推理模型约需
   1-3 分钟，页面会实时显示「当前阶段 + 已等待秒数」，不需要盲等）。
4. 点「创建游戏」→ 结果面板展示校验结论、置信度、规则族与变更摘要；
   成功后该游戏自动出现在大厅（卡片带「🛠 自定义·<族>」徽标），
   点击即可按普通对局流程开始（难度/座位/陪伴 Agent/提示/复盘全功能可用）。
5. **删除自定义游戏/变体（平台端 UI）**：
   - 大厅：自定义游戏卡片右下角有「🗑 删除」按钮（内置游戏无此按钮），
     点击后确认即可删除。
   - 创建页：底部「我的自定义游戏」管理列表展示全部自定义游戏与模板变体
     （id、族、创建时间），每条右侧有「🗑 删除」按钮；新建成功自动刷新该列表。

## 创建体验（失败也绝不让你空等）

创建页的请求走 **SSE 流式**（`POST /api/custom/games?stream=1`），服务端按
阶段推进度：`translate`（翻译规则）→ `validate`（schema + L2 冒烟校验）→
`register`（注册到大厅），页面显示阶段文案与已等待秒数。三条红线：

- **勾了 LLM 也绝不空手而归**：LLM 端点不可达时**先探活**（1-3 秒）再决定，
  不浪费一次 300s 超时；翻译失败（超时 / 输出过不了校验）时自动改走确定性
  模板，成功则返回游戏并在结果面板打出醒目告警（响应里的 `llm_fallback` +
  `validation.warnings` 首条）——**不会**把模板产物伪装成 LLM 成果。
- **不注册玩不了的游戏**：注册前跑「可玩性探针」（各族可选钩子
  `probe_playable`）。网格族要求：`constants.board_size` 为正整数、初始局面
  存在合法落子、落子动作能换算成棋盘格位——否则创建阶段就明确报错
  （「生成的规则在平台上无法对弈」），而不是让大厅里多出一个点哪都非法的
  废游戏。落子参数名容错：模型把落子参数叫 `square`/`pos` 也能对弈
  （不再只认 `cell`）。
- **失败原因可读**：错误里带 LLM 真实原因（端点/HTTP 状态/输出预算被思维链
  吃光）与模板兜底原因，两条都会显示。
- **说「玩X」界面立刻出现**：棋盘开局的第一帧在 **AI 先手思考之前**就推给前端
  （此前要等 AI 第一手算完，大棋盘上就是几十秒空白页，用户看到的是「对话里
  调不出来界面」）；自定义游戏走对话开局的默认 `adaptive=true` 也**不会**再
  报 `未知游戏`（`AdaptiveController` 的预算表只登记内置游戏，会话层已回落到
  该游戏自己的 normal 档）。
- **AI 不会想太久**：自定义网格游戏的搜索带**时间上限**（easy 1.5s /
  normal 3s / hard 6s，先到先停）——迭代预算管不住墙钟，16×16 棋盘上
  normal=800 次迭代要 ~55s、easy=200 次也要 ~14s，玩家点一下等半分钟同样是
  「玩不了」。小板照旧跑满预算，大板自动变浅但立刻响应。

## 支持的规则族（平台可对弈子集）

| 族 | 识别信号（规则形状） | 渲染 | AI |
|----|---------------------|------|-----|
| `grid` 网格 | `derivedViews.cell` 为 grid + `board` 数组 | 通用网格棋盘 | MCTS |
| `poker` 扑克 | hand/community 数组 + raise/call/fold 动作 | 扑克桌 | Hybrid（不完全信息搜索） |
| `mahjong` 麻将 | hand/melds/discard 数组 + 吃碰杠胡动作 | 麻将桌 | 麻将启发式 |
| `social` 社交推理 | text 发言参数动作 + 夜晚/投票阶段 | 聊天桌 | 本地 Ollama（每 AI 座位一个实例；不可用回退随机） |

- 识别不到的规则会明确提示「暂不支持平台对弈」并附校验错误，不会静默失败。
- 社交类（狼人杀/谁是卧底式）对局需要本地 Ollama 才有真实的 AI 发言；
  没有时 AI 回退随机动作，快照里标注 `ai_mode`。

## 数据与安全

- 自定义游戏持久化在 `data/custom_games/<game_id>.json`（已 gitignore），
  通过 `GET / DELETE /api/custom/games[/<id>]` 管理（大厅不显示已删除游戏）。
- 所有自定义/变体规则都经过 **schema 校验 + L2 引擎冒烟校验**（
  `engine_validator.validate()` = schema + `smoke_validate(variants="all")`）
  才会注册；引擎一律 `allow_codegen=False` 纯解释器路径构造——
  见 `docs/design/security-notes.md`。
- 隐藏信息红线与内置游戏一致：AI 底牌/手牌/角色只在对局结束或揭晓后可见，
  陪伴 Agent 与复盘只读玩家视角投影。

## API 一览

| 接口 | 说明 |
|------|------|
| `POST /api/custom/games` | `{mode:"from_scratch"\|"variant", rule_text?, base_game_id?, change_text?, game_name?, source_lang?, use_llm?}` → `{ok, game_id, game, confidence, family, diff_summary?, validation, llm_fallback?}`；失败 400 + 中文原因。加 `?stream=1`（或 `Accept: text/event-stream`）走 SSE：`stage{stage,detail}` 进度事件 → `result`（同上信封）或 `error`（含 `error/validation/diff_summary`）→ `done` |
| `GET /api/custom/games` | 自定义游戏列表 |
| `DELETE /api/custom/games/{game_id}` | 删除（404 当不存在） |
| `GET /api/games` | 大厅列表（内置 + 自定义合并，自定义条目带 `custom:true` 与 `family`） |

`llm_fallback = {used: true, reason}` 表示「勾了 LLM 但这次没用上，已回落确定性
模板」，前端必须醒目提示（`validation.warnings` 首条同义）。

## 内部结构（开发者速览）

- `layer1_translator/variant_translator.py` — 变体翻译（确定性参数路径 + LLM
  全模板改写 / v5.5 增量补丁修复循环，输出必过 `engine_validator`）
- `layer1_translator/prompt_builder.py` — 从零翻译的接地提示（v5 方言速查 +
  完整参考规则示例）；`layer1_translator/local_client.py` — 翻译尺度的输出
  预算/超时（`LLM_MAX_TOKENS` / `LLM_TIMEOUT_S`）
- `layer1_translator/template_translator.py` — 确定性模板翻译（7 个已知模板：
  moon_chess/stochastic_gomoku/texas_holdem/mahjong/werewolf/uno +
  gomoku 别名）
- `layer4_interface/frontend/platform/families/` — 规则族包（自动发现；
  `detect` 判定 + `build_spec` 产出 GameSpec + 可选 `probe_playable`
  可玩性探针；grid/poker/mahjong/social 四族）
- `layer4_interface/frontend/platform/custom_games.py` — CustomGameStore
  （持久化）+ CustomGameRegistry（翻译→校验→族识别→可玩性探针→建 spec→注册
  编排，`on_stage` 进度回调）
- `layer4_interface/frontend/platform/server.py` — 创建路由的 SSE 阶段进度、
  LLM 端点预检与确定性降级策略（`_create_custom_game`）
- 平台会话/求解器装配：`session.py` 查表回退到自定义注册表；
  `train-cli/games.py` 的 `create_solver(..., allow_unknown=True)` 为未登记
  自定义游戏装配通用运行时求解器（mcts/random/ollama/mahjong/hybrid 白名单，
  其余名字仍拒绝）