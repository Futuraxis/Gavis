# 谁是卧底（Undercover）使用说明

谁是卧底是经典的派对语言游戏：平民拿到同一个词，卧底拿到一个相似但不同的词，
白板则没有词。大家轮流用一句话描述自己的词，再投票把最可疑的人投出局。

Gavis 中的实现基于 `rules/undercover.json`（v5.2 声明式 variants），规则全部
声明在单个 JSON 里，引擎纯数据解析。**更贴近实际玩法**：开局玩家**不知道**
自己是平民还是卧底（只看自己的词，靠发言推断阵营）；投票阶段除了投票，
还可以选择**自爆**并猜测别人的词。

## 配置

```python
# train-cli/games.py 注册表（自动登记）
GameSpec(
    game_id="undercover", display_name="谁是卧底",
    engine=EngineSpec(rules="undercover.json", variant="fruit_normal", player_count=8),
    runtime_solvers=("ollama", "random"),
)
```

- **人数**：4..12 人（默认 8），运行时用 `player_count` 覆盖
  （`GameEngine(rules, player_count=N)`；`train-cli/games.py` 里改
  `EngineSpec.player_count`）。
- **配比**：固定 1 卧底 + 1 白板 + N 平民（人数在 variants 里由公式声明）。
- **主题（词对类别）+ 难度（词对相似度）**：`variant` 形如 `{theme}_{difficulty}`
  —— 6 主题 × 3 难度 = 18 个 variant，全部在 `variants.options` 声明，未知 variant → `ValueError`。
  - **主题**（`fruit`/`food`/`animal`/`object`/`place`/`plant`）：词对所属类别。
  - **难度**（`easy`/`normal`/`hard`）：词对相似度档——`easy` 词对差异大（如
    苹果/香蕉）、`normal` 同类相近（如 包子/饺子）、`hard` 极易混淆（如 菠萝/凤梨、
    猎豹/花豹、骡子/马）。每档每主题约 3 对，每局再从该档词池**随机抽一对**。

## 平台对弈

平台大厅已内置本游戏（`platform/games.py` 注册表，social 族）：

- **入口**：启动平台后浏览器打开 `http://127.0.0.1:8770`，大厅点「谁是卧底」→「开始对局」；
  对局界面为社交聊天桌（身份/词语/存活/发言记录/投票），与自定义社交游戏共用。
- **人数**：默认 8 人（1 卧底 + 1 白板 + 6 平民），可选手数 8/4/5/6/7/9/10/11/12。
- **主题 + 难度选择**：大厅「谁是卧底」开始对局时可选择**主题**（下拉，6 类别）
  与**难度/节奏**（平台统一 3×3 网格）。主题决定词对类别，难度决定词对相似度档
  （详见下文「难度与节奏」）。词对在 `rules/undercover.json` 的 `variants.options`
  声明，平台按 `{theme}_{tier}` 拼 variant 开局，每局再从该档词池**随机抽一对**。
- **AI**：每个 AI 座位一个独立求解器 —— 本地 Ollama 可用时走大模型发言
  （快照 `ai_mode=ollama`），否则随机策略（`ai_mode=random`，页面如实标注
  「本地大模型 / 随机策略」）；即使 Ollama 中途失败也会如实降级标注并随机兜底。
- **实时动态（全游戏通用）**：开局与每一步都走 SSE 流式——AI 每次发言/行动
  落地即推一帧玩家投影快照，发言逐条上屏（「谁在说话」的座位名牌与打字
  动画同步切换），人类自己的发言/投票也立即回显，不再等服务端把整轮 AI
  循环跑完才一次性看到全部内容。
- **隐藏信息红线**：快照只从 `engine.project_observation` 构建——你只能看到自己的
  身份与词语；他人身份/词语（含 AI 的）在终局前不出现。

## 难度与节奏

平台统一 **`difficulty × pacing` 3×3 网格**：两个维度各自独立影响卧底玩法的不同
侧面，组合出 9 种节奏（与麻将选 variant、狼人选难度共用同一套平台契约）。

| 维度 | 取值 | 影响（谁是卧底） |
|---|---|---|
| `difficulty` | `easy`/`normal`/`hard` | **词对相似度档**——`easy` 词对差异大、线索
 好抓；`hard` 极易混淆（如菠萝/凤梨、猎豹/花豹、骡子/马），平民难辨卧底。
 **同时**叠加 AI 发言强度提示（`_UNDERCOVER_HINT`：`easy` 直白描述、
 `normal` 克制模糊——每句只给一个泛化特征、禁止堆叠多个专属细节；
 `hard` 高度模仿/误导）。 |
| `pacing` | `fast`/`standard`/`slow` | **AI 发言温度**——`fast` 高温(0.9)发散、描述
 跳脱易露馅；`standard`(0.7)平衡；`slow`(0.5)精准、强伪装、线索少。 |

- **自适应模式（`adaptive=true`）**：词对档锚定 `normal`（AI 强度仍按自适应档调节），
  只有显式 `difficulty` 才选 `easy`/`hard` 词对池——保证「自适应」不会因词对过难
  让平民开局即崩盘。
- **主题**与难度正交：选 `animal_hard` 即「动物·极易混淆」词池；选 `food_normal`
  即「食物·同类相近」。每档每主题约 3 对，每局随机抽一对，重复游玩不重复。

## 规则

每轮流程：

1. **描述**：所有存活玩家按座位轮流用一句话描述自己的词（`speak`，自由文本，
   不能说出词本身；`text` 参数走 v5.1 预制能力，不参与合法动作枚举）。
2. **投票**：所有存活玩家轮流**投票**指认卧底（目标为其他存活玩家，不能投
   自己），或选择**自爆**（`self_destruct`）点名一个存活玩家并猜其词语。
3. **结算**：得票最多者出局；**平票无人出局**，直接进入下一轮。
   （自爆会中断本轮投票，跳过本轮 resolve，直接进入下一轮 describe。）

### 自爆（self_destruct）

玩家开局**不知道**自己是平民还是卧底（`my_role` 不投影），只看自己的词
（`my_word`）；白板看到「白板」(无词) 自知是白板。靠大家的描述推断阵营后，
可在投票阶段选择自爆，点名一个存活玩家并猜其词语：

| 自爆者身份 | 猜对 | 猜错 |
|---|---|---|
| 平民 | 直接淘汰（平民不该自爆） | 直接淘汰 |
| 卧底 | 猜对**平民词**（target 是平民且 guess==其词）→ **卧底直接获胜** | 卧底淘汰，游戏继续 |
| 白板 | 猜对目标词 → **白板直接获胜** | 白板淘汰，游戏继续 |

自爆失败只淘汰自爆者本身（**不**触发「卧底/白板被投出→平民胜」——那是投票
专属）；若淘汰后剩余存活触发生存胜利条件仍会结算。

胜负判定：

| 条件 | 胜方 |
|---|---|
| 卧底被投出 | 平民（`winner=civilian`） |
| 白板被投出 | 平民 |
| 卧底自爆猜对平民词 | 卧底（`winner=undercover`） |
| 白板自爆猜对目标词 | 白板（`winner=blank`） |
| 存活 ≤ 3 人且白板存活 | 白板（`winner=blank`） |
| 存活 ≤ 2 人且卧底存活 | 卧底（`winner=undercover`） |
| 存活 ≤ 2 人且无卧底 | 平民 |
| 轮次达到 players+8 | 平局（`winner=None`，所有人收益 0） |

## 可观测性（部分信息博弈）

- **身份隐藏**：不开 `my_role` 视图——每个玩家只看**自己的词**
  （`my_word` 按 viewer 过滤）；平民/卧底不知道自己的身份标签，要靠发言推断。
  白板看到「白板」(无词) 自知是白板。
- 死后身份与词语公开（`dead_roles` / `dead_words` 保留 `alive==0` 的行）。
- 描述、投票、自爆事件、出局记录全员可见；`env` 字段（轮次、出局者、胜方）全员可见。

## 运行

```bash
python train-cli/train.py --list                       # 注册表一览（含 undercover）
python train-cli/train.py --game undercover --solver all --skip-eval
# 无专用可训练求解器（solvers 为空，脚本自动跳过），
# 运行时求解器：ollama（自由文本发言）/ random（均匀随机）
```

## 规则实现

- 规则源文件：`_gen_undercover.py`；改规则后运行
  `python _gen_undercover.py` 重新生成 `rules/undercover.json`。
- 引擎测试：`tests/test_layer2_engine/test_undercover.py`
  （variants、发牌、观测过滤、轮转、平票、胜负、收益）。