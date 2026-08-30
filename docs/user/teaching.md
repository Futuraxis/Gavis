# 教学对局使用说明

教学对局（teaching match）是平台的第四种陪伴能力：**教练 Agent 能看到你自己的牌并进行推理**——像一位坐在你身后看牌的教练，带你打、给你讲。

它解决了此前"教学与复盘对最需要陪伴的牌类游戏实质空转"的覆盖盲区（评估/提示层只对方阵棋盘有效，牌类游戏拿不到手牌级推理）。

## 怎么开

- **平台界面**：任意游戏的开局配置卡勾选「教学对局」（默认搭配「认真教学」人格，可改）。
- **API**：`POST /api/match/start` 传 `"teaching": true`（或 `"mode": "teaching"`）。
- 支持全部平台游戏：月亮棋 / 随机五子棋 / 德州扑克 / 麻将六变种 / UNO 六变体 / 自定义游戏。

## 教练的节奏

| 时机 | 场景 | 内容 |
|------|------|------|
| 开局 | `teach_greet` | 教练开场：说明教学局规则 |
| 轮到你 | `teach_turn` | 读牌导读：你的手牌 + 选择面（**不剧透答案**） |
| 你走完 | `teach_move` | 讲评：你打的 vs 教练在同样局面会打的（参考动作），一致/偏差与原因 |
| 要提示 | `/api/match/hint`（`specific`/`demo`） | 升级为教练参考动作（求解器在**你的座位**上算的真实走法） |
| 自由聊天 | `/api/agent/say` | 教练能围绕你的牌回答（"我现在听什么？"） |

消息经既有 chat 增量通道（快照 `chat` 数组）投递，前端对战页 / 对话页都会落到对话流里。

## 设计红线（怎么融合进框架）

教学对局不改引擎、不改求解器、不加规则语言原语，全部落在 Layer 4 既有接缝上：

1. **教练看的 = 玩家看的**。教练的唯一数据入口是 `engine.project_observation(state, player_pid)`——玩家自己的投影（含玩家自己的手牌视图 `hand_view_p0` / `sb_hole_view`）。教练从不比你知道更多：**看不到 AI 的手牌、牌墙、未翻牌堆**。`hidden_guard.assert_no_hidden` 照常校验教学上下文。
2. **双脑分离**。对手脑（会话 `solver`，在 AI 座位公平落子）与教练脑（`agent/coach.py`）互不相通，AI 自己的落子路径零改动。教练的"推理"是在**玩家回合**用同一求解器契约（`select_action` 按状态当前行动者推理）替你算一手参考动作——求解器给玩家座位推理时读的正是你的手牌。
3. **在线学习防污染**。参考动作走会话的 `raw_solver`（未被 `RecordingHandle` 包装的原始句柄）：教练替你算的动作不会以 "ai" actor 混进训练轨迹；AI 座位的决策照常采集。
4. **泄露扫描定向放宽**。`hidden_guard.scan` 的教学变体：你的牌可以讨论（你本来就看得到），**AI/对手的**隐藏信息（"我的/AI 的底牌/手牌/身份"）仍然拦截。非教学对局扫描行为不变。
5. **对局记录**：历史记录 `meta.teaching` 标记教学局（旧记录缺省 `null` = 非教学局）。

## 代码地图

| 位置 | 职责 |
|------|------|
| `layer4_interface/agent/coach.py` | `Coach` / `TeachContext`：玩家投影 + 参考动作 + 讲评对照 |
| `layer4_interface/agent/scenarios.py` | 追加 `teach_greet` / `teach_turn` / `teach_move`（末尾追加，契约兼容） |
| `layer4_interface/agent/hidden_guard.py` | `scan(..., teaching=True)` 教学泄露模式 |
| `layer4_interface/agent/dialogue_engine.py` | 教练系统提示 + 教学机械事实 payload |
| `layer4_interface/frontend/platform/session.py` | `PlayManager.start(teaching=)` / `step` 参考动作 / teach 消息编排 |
| `platform-frontend/src/chat/snapshotChat.ts` | 快照 chat 增量 → 对话流（教练消息玩家可见） |

## 已知边界

- 教练讲评依赖求解器给出的参考动作；MCTS 类随机求解器两次调用可能给出不同参考（参考是"一手合理走法"而非唯一最优）。
- `teach_turn` 导读刻意不带参考动作（不剧透）；要直接答案用提示接口。
- 玩家的牌以规则 id（如 `w1`、`d9`）进入 LLM 机械事实，成文质量取决于 LLM 可用性；无 LLM 时回退人格兜底台词。
