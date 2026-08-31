# 自定义游戏 · 使用说明（Layer 1 纳入平台工作流）

玩家可以在平台里用**自然语言规则**或**已有游戏的变体（模板+改动描述）**直接
生成可对弈的新游戏——这是 Layer 1（规则翻译）接入对弈工作流后的主入口。
翻译 → 校验 → 规则族识别 → 注册 → 大厅出现 → 正常人机对弈，全链路自动完成。

## 快速开始

```bash
python -m layer4_interface.frontend.platform.server   # 8770（需先 npm run build）
```

浏览器打开 **http://127.0.0.1:8770/** → 顶部导航「创建游戏」（或直接进 /create）：

1. **模式一：规则描述（from_scratch）**——在文本框里用一句话描述规则，
   例如「8×8 棋盘，四子连珠获胜，黑棋先手」。
2. **模式二：基于模板变体（variant）**——选一个基础游戏（月亮棋/随机五子棋/
   德州扑克/麻将/狼人杀/谁是卧底），再描述要做的改动，例如「棋盘改成 7×7，
   五子连珠获胜，每步落子后 30% 概率抹去一格」。变体翻译走确定性参数路径
   （能安全消费的参数直接应用）；复杂改动走 LLM 全模板改写或 v5.5 增量补丁
   修复循环，输出必过 `engine_validator`（schema + L2 冒烟，对所有声明变体
   boot）。
3. 可选：游戏名称、LLM 翻译开关（需本地模型可用；不可用自动回落确定性翻译）。
4. 点「创建游戏」→ 结果面板展示校验结论、置信度、规则族与变更摘要；
   成功后该游戏自动出现在大厅（卡片带「🛠 自定义·<族>」徽标），
   点击即可按普通对局流程开始（难度/座位/陪伴 Agent/提示/复盘全功能可用）。
5. **删除自定义游戏/变体（平台端 UI）**：
   - 大厅：自定义游戏卡片右下角有「🗑 删除」按钮（内置游戏无此按钮），
     点击后确认即可删除。
   - 创建页：底部「我的自定义游戏」管理列表展示全部自定义游戏与模板变体
     （id、族、创建时间），每条右侧有「🗑 删除」按钮；新建成功自动刷新该列表。

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
| `POST /api/custom/games` | `{mode:"from_scratch"\|"variant", rule_text?, base_game_id?, change_text?, game_name?, source_lang?, use_llm?}` → `{ok, game_id, game, confidence, family, diff_summary?, validation}`；失败 400 + 中文原因 |
| `GET /api/custom/games` | 自定义游戏列表 |
| `DELETE /api/custom/games/{game_id}` | 删除（404 当不存在） |
| `GET /api/games` | 大厅列表（内置 + 自定义合并，自定义条目带 `custom:true` 与 `family`） |

## 内部结构（开发者速览）

- `layer1_translator/variant_translator.py` — 变体翻译（确定性参数路径 + LLM
  全模板改写 / v5.5 增量补丁修复循环，输出必过 `engine_validator`）
- `layer1_translator/template_translator.py` — 确定性模板翻译（7 个已知模板：
  moon_chess/stochastic_gomoku/texas_holdem/mahjong/werewolf/uno +
  gomoku 别名）
- `layer4_interface/frontend/platform/families/` — 规则族包（自动发现；
  `detect` 判定 + `build_spec` 产出 GameSpec；grid/poker/mahjong/social 四族）
- `layer4_interface/frontend/platform/custom_games.py` — CustomGameStore
  （持久化）+ CustomGameRegistry（翻译→校验→族识别→建 spec→注册 编排）
- 平台会话/求解器装配：`session.py` 查表回退到自定义注册表；
  `train-cli/games.py` 的 `create_solver(..., allow_unknown=True)` 为未登记
  自定义游戏装配通用运行时求解器（mcts/random/ollama/mahjong/hybrid 白名单，
  其余名字仍拒绝）