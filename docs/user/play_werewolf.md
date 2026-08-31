# 狼人杀（Werewolf）使用说明

狼人杀是经典的社交推理语言游戏：夜晚狼人秘密击杀、预言家查验身份、女巫救人或
下毒、猎人开枪；白天大家轮流发言推理，再投票放逐最可疑的人。胜负由「屠边」判定
——好人阵营（村民/神职）或狼人阵营，谁先被清空谁输。

Gavis 中的实现基于 `rules/werewolf.json`（v5.2 声明式 variants，配比/流程全部声明
在单个 JSON 里，引擎纯数据解析，无 per-game 适配器）。

## 配置

```python
# train-cli/games.py 注册表（训练侧自动登记）
GameSpec(
    game_id="werewolf",
    engine=EngineSpec(rules="werewolf.json"),
    runtime_solvers=("ollama", "random"),
)
```

- **人数**：固定 9 人一档（3 狼人 + 3 村民 + 预言家 + 女巫 + 猎人）；配比由
  `constants.role_pool` 决定，平台侧 `player_counts == (9,)`，不提供其他人数。
- **夜晚顺序**：狼人击杀（`kill`）→ 女巫（`heal` 救人 / `poison` 下毒，不可自救、
  首夜保护开）→ 预言家查验（`check`，结果 `seerResult` 仅本人可见）→ 猎人开枪
  （`shoot`）。
- **白天**：发言（`speak`，自由文本 + 意图槽）→ 投票（`vote`）→ 放逐结算
  （被放逐的猎人可开枪 `shoot_lynched`）→ 进入下一夜。
- **胜负**：`win_mode=side` 屠边 —— 狼人全部出局 → 好人胜；狼人数 ≥ 好人方人数
  → 狼人胜。

## 平台对弈

狼人杀已内置平台（`platform/games.py` 注册表 + `session.py` `_BUILTIN_FAMILY`，
social 族，与谁是卧底共用社交聊天桌）：

- **入口**：启动平台后浏览器打开 `http://127.0.0.1:8770`，大厅点「狼人杀」→
  「开始对局」；或直接对平台聊天说「玩狼人杀」/「来一局狼人杀」开局。
- **对局**：界面为社交聊天桌（身份/存活/发言记录/投票/夜晚技能），与谁是卧底
  共用 `SocialChatTable`；夜晚阶段行动者身份脱敏（`turn=None`）。
- **AI**：每个 AI 座位一个独立求解器 —— 本地 Ollama 可用时走大模型发言
  （快照 `ai_mode=ollama`），否则随机策略（`ai_mode=random`，页面如实标注
  「本地大模型 / 随机策略」）；Ollama 中途失败也会如实降级标注并随机兜底。
- **隐藏信息红线**：快照只从 `engine.project_observation` 构建 —— 你只能看到
  自己的身份（`my_role`）与已公开的死者身份（`dead_roles`）；夜晚查验结果
  （`seerResult`）仅预言家可见；他人身份在终局前不出现。
- **夜晚脱敏**：狼人/女巫/预言家/猎人的私密夜晚阶段，非本人回合时快照不暴露
  当前行动者身份（`turn=None`），白天发言/投票顺序照常公开。

## 难度与节奏

平台统一 **`difficulty × pacing` 3×3 网格**（与谁是卧底、麻将选 variant 共用同一套
平台契约）。狼人杀**无词对**（variant 固定 `default` 配比），两个维度全部作用在
**AI 发言质量**上：

| 维度 | 取值 | 影响（狼人杀） |
|---|---|---|
| `difficulty` | `easy`/`normal`/`hard` | **AI 角色发言策略强度**（`ROLE_GUIDE` 按角色
 × 难度三档）——`easy` 引导简短直白、易暴露阵营倾向；`hard` 引导细致伪装、
 主动带节奏/泼脏水/做身份。狼人/预言家/女巫/猎人/村民各有独立策略档。 |
| `pacing` | `fast`/`standard`/`slow` | **AI 发言温度**——`fast` 高温(0.9)发散、
 逻辑跳跃易自爆；`standard`(0.7)平衡；`slow`(0.5)精准、逻辑严密强伪装。
 节奏本身影响难度：`fast` 下 AI 更跳脱（破绽多）、`slow` 下 AI 更精准（难抓）。 |

- **自适应模式**：`adaptive=true` 时 AI 强度按自适应档调节（`AdaptiveController`），
  与显式 `difficulty` 二选一锚定 `ROLE_GUIDE` 档。
- 平台 9 人固定配比（3 狼/3 村/预言家/女巫/猎人）不受难度影响，难度只改 AI 表现。

## 规则

每轮流程：

1. **夜晚**：存活狼人轮流选择击杀目标（`kill`）；女巫选择是否使用解药（`heal`，
   首夜可救，不可自救）或毒药（`poison`）；预言家查验一名玩家（`check`）；猎人可
   在夜晚开枪（`shoot`）。首夜保护开启（第一夜狼人无法击杀被保护目标）。
2. **白天发言**：所有存活玩家按座位轮流发言（`speak`，自由文本，附加意图槽
   `claim/accuse/defend/question/persuade`）。
3. **白天投票**：所有存活玩家轮流投票放逐一名玩家（`vote`）；被放逐者若是猎人，
   可在放逐时开枪（`shoot_lynched`）。
4. **结算**：狼人全部出局 → 好人胜（`winner=good`）；狼人数 ≥ 好人方存活数 →
   狼人胜（`winner=wolf`）；超过轮次上限时按存活阵营兜底结算（避免死局循环）。

## 可观测性（部分信息博弈）

- 每个玩家只能看到**自己的身份**（`my_role` 按 viewer 过滤，`drop` 规则）。
- 死者身份公开（`dead_roles` 保留 `alive==0` 的行）。
- 发言记录、投票记录、放逐/死亡公布全员可见；`env` 字段（当前阶段、轮次、胜负）
  全员可见；`seerResult` 仅预言家可见（`visibility.env` 投影）。

## 运行

```bash
python train-cli/train.py --list                        # 注册表一览（含 werewolf）
python train-cli/train.py --game werewolf --solver all --skip-eval
# 无专用可训练求解器（solvers 为空，脚本自动跳过），
# 运行时求解器：ollama（自由文本发言）/ random（均匀随机）
# 平台侧：内置游戏已注册（platform/games.py + session.py），见上文「平台对弈」
```

## 规则实现

- 规则源文件：`_gen_werewolf.py`；改规则后运行
  `python _gen_werewolf.py` 重新生成 `rules/werewolf.json`。
- 引擎测试：`tests/test_layer2_engine/test_werewolf.py`
  （variants、发牌、夜晚顺序、观测过滤、轮转、投票放逐、胜负、收益）。
- 求解器测试：`tests/test_layer3_solvers/test_bayes_werewolf.py`
  （贝叶斯狼人推理基线）。
- 平台会话测试：`tests/test_layer4_interface/test_family_social.py`
  （`TestWerewolfSmoke` 开局/快照红线、`TestNightTurnMasking` 夜晚脱敏）。