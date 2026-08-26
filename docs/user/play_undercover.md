# 谁是卧底（Undercover）使用说明

谁是卧底是经典的派对语言游戏：平民拿到同一个词，卧底拿到一个相似但不同的词，
白板则没有词。大家轮流用一句话描述自己的词，再投票把最可疑的人投出局。

Gavis 中的实现基于 `rules/undercover.json`（v5.2 声明式 variants），规则全部
声明在单个 JSON 里，引擎纯数据解析。

## 配置

```python
# train-cli/games.py 注册表（自动登记）
GameSpec(
    game_id="undercover", display_name="谁是卧底",
    engine=EngineSpec(rules="undercover.json", variant="fruit", player_count=8),
    runtime_solvers=("ollama", "random"),
)
```

- **人数**：4..12 人（默认 8），运行时用 `player_count` 覆盖
  （`GameEngine(rules, player_count=N)`；`train-cli/games.py` 里改
  `EngineSpec.player_count`）。
- **配比**：固定 1 卧底 + 1 白板 + N 平民（人数在 variants 里由公式声明）。
- **场景（词对）**：`variant` 选择 —— `fruit`（平民「苹果」/ 卧底「香蕉」，
  默认）或 `food`（平民「汉堡」/ 卧底「肉夹馍」）；未知 variant → `ValueError`。

## 规则

每轮流程：

1. **描述**：所有存活玩家按座位轮流用一句话描述自己的词（`speak`，自由文本，
   不能说出词本身；`text` 参数走 v5.1 预制能力，不参与合法动作枚举）。
2. **投票**：所有存活玩家轮流投票，目标为**其他存活玩家**（不能投自己，无弃权）。
3. **结算**：得票最多者出局；**平票无人出局**，直接进入下一轮。

胜负判定：

| 条件 | 胜方 |
|---|---|
| 卧底被投出 | 平民（`winner=civilian`） |
| 白板被投出 | 平民 |
| 存活 ≤ 3 人且白板存活 | 白板（`winner=blank`） |
| 存活 ≤ 2 人且卧底存活 | 卧底（`winner=undercover`） |
| 存活 ≤ 2 人且无卧底 | 平民 |
| 轮次达到 players+8 | 平局（`winner=None`，所有人收益 0） |

## 可观测性（部分信息博弈）

- 每个玩家只能看到**自己的身份与词**（`my_role` / `my_word` 按 viewer 过滤）。
- 死后身份与词语公开（`dead_roles` / `dead_words` 保留 `alive==0` 的行）。
- 描述、投票、出局记录全员可见；`env` 字段（轮次、出局者、胜方）全员可见。

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