# Gavis 项目指南

## 项目概述

自适应策略游戏 AI Agent — 四层水平集成架构。

## 目录结构

```
rules/                 游戏规则 JSON (v5.2, 零 BUILTIN + variants 声明式: texas_holdem / mahjong / werewolf / undercover / uno)
layer1_translator/    LLM 规则翻译层 (已实现: 模板 + LLM 编排 + schema 校验)
layer2_engine/        游戏引擎核心 (GameEngine — 无 per-game 适配器, 无 interfaces/)
layer3_solvers/       求解器 (MCTS/CFR/PPO/PSRO + mahjong/heuristic + marl/(QMix/HAPPO/MAAC))
layer4_interface/     交互界面 (Binding/Encoding/Frontend/OnlineLearning + agent/difficulty/profile/review)
train-cli/            训练 CLI：games.py 游戏注册表（配置驱动，17 游戏）+ train.py 统一训练脚本
train_cli.py          根目录导入桥 → train-cli/（使连字符目录可 import / python -m train_cli）
tests/                测试 (902 cases, 8 skipped — 含 36 个层二 UNO 引擎测试)
platform-frontend/    平台前端 (React + Vite + TS, 构建产物 dist/ 已 gitignore)
data/                 运行时数据 (对局记录 data/matches/, 已 gitignore)
archive/              原始旧代码只读存档
docs/                 架构设计 + 六篇合并分析文档
custom-games 工作流: layer1_translator/variant_translator.py（变体翻译）+ layer4_interface/frontend/platform/families/（规则族: grid/poker/mahjong/social，detect+build_spec 自动发现）+ custom_games.py（注册表）+ 前端 /create 创建游戏页；用户文档 docs/user/custom_games.md
```

## 代码规范

完整的代码规范、格式化规则、命名约定和层间契约详见 **[docs/coding-standards.md](docs/coding-standards.md)**。

核心要点：
- **Python 3.11+**, `from __future__ import annotations`
- `ruff format` + `ruff check` (line-length=120)
- 全覆盖类型标注，优先 `X | None` 而非 `Optional[X]`
- 标准库 → 第三方 → 项目内部导入分组
- 层内相对导入 (`..base`)，跨层绝对导入 (`layer2_engine.*`)
- Google 风格文档字符串（`---` 分隔章节标题）
- Dataclass 配置 + Protocol 契约 + ABC 抽象基类
- 自定义异常层次（每层一个 Base 类）
- pytest 测试（固定 seed、fixture、可选依赖 skipif）

## 关键约定

- **Python 3.11+**, 类型标注全覆盖 (`from __future__ import annotations`)
- `ruff` 格式化 + lint (line-length=120)
- `pytest` 运行测试: `python -m pytest`
- 层间通信只能通过契约（Layer2→3: `GameEngine`；Layer3: `SolverBase`；Layer4: `BaseBinding`）
- **禁止循环依赖**：Layer N 只能依赖 Layer N-1
- Layer 4 (Interface) 不依赖 Layer 3 (Solver)
- 规则 JSON 放在 `rules/` 顶层，不在层目录内
- **v5.1 零 BUILTIN**：规则自足，`BUILTIN_FUNCTIONS` 已退役；引擎从
  `rules["functions"]` 读取 alias 定义（`{"params": [...], "expr": {...}}`）
- 语言原语只从数学操作角度增加（`choose/range/sort/group/at/add/sub/...`），
  禁止游戏特供原语
- 判定遵循增量局部原则：围绕 `lastPlacedCell` / `lastDiscard` /
  `last_action` 做 O(1) 局部判定
- **v5.2 variants 声明式**：变种/人数/配比都在单个 JSON 的 `variants`
  节声明（`variant` + `player_count` + `options[n].constants` 补丁；
  `$variant/$player_count/$constants/$players` 上下文），引擎做纯数据
  解析，**没有任何 constants 注入 API**；未知 variant → ValueError
- `rules/mahjong.json` 由 `_gen_mahjong.py` 生成（六变种 × 2/4 人由
  variants 声明：guangdong/hongzhong/blood/sichuan/changsha/taiwan），
  改规则改生成器再重新生成
- `rules/werewolf.json` 由 `_gen_werewolf.py` 生成（配比 9 人/3 狼在
  variants 声明，消费者只校验不注入）；结算阶段（夜晚/放逐）用
  `chance` 模板 + `effectMap` 表达（explicit 概率 1.0）；部分可观测由
  `visibility` 声明（`my_role`/`dead_roles` 视图 + `env` 字段投影 →
  `seerResult` 仅预言家可见）；轮转由 env/`turn` 驱动（狼人杀 `speak`
  用 text 参数预制能力）
- `rules/uno.json` 由 `_gen_uno.py` 生成（108 张牌 × 2-10 人 × 六变体
  classic/seven_zero/jump_in/stacking/draw_until/strict_wild4 全在
  variants 声明）；出牌/罚牌/抢牌/叠加全部用 `env` + `chance` 表达；
  牌堆无实体数组（`deck = 108 − 手牌 − 弃牌`，`undrawn_cards` 查询 +
  `deck_count` 别名）；手牌部分可观测（`visibility` 隐藏他人牌面但
  保留张数）；`hand_of` 别名（10 分支 switch）可内联进查询/合法
  条件——编译器对 switch 生成无 walrus 的首匹配 if/elif 链
  （`:=` 在 comprehension 迭代表达式里是 SyntaxError，曾经的
  switch-in-comprehension 会让整套规则回退纯解释器；现可正常编译）
- **text 参数预制能力**：动作参数声明 `"type": "text"` 时不参与合法
  动作枚举（占位 `""`），solver 把文本放进 `ActionInstance.params`，
  effector 经 `$text` 读取（`_build_context` 自动平铺 params）——
  自由文本发言游戏（狼人杀等）用，规则语言通用能力非游戏特供

## 求解器注册

新求解器 → `layer3_solvers/base.py` 实现 `SolverBase` → 注册到 `__init__.py`

## 文档

| 文件 | 内容 |
|------|------|
| `docs/design/architecture.md` | 当前四层架构设计 (v0.2) |
| `docs/design/online-learning.md` | 在线学习设计（捕获/经验对手模型/门禁发布） |
| `docs/merge/01~06.md` | 六篇合并分析与方案文档 |
| `docs/design/security-notes.md` | 安全与性能决策记录（审计 3.6 修复项与暂缓项） |
| `docs/design/gamerule/v4.1.md` | 规则语言设计 (v5.0: `docs/design/gamerule/v5.0.md`, v5.1: `docs/design/gamerule/v5.1.md`, 语法参考: `docs/design/gamerule/v5.1-reference.md`) |
| `docs/user/play_moon_chess.md` | 月亮棋人机对弈使用说明 |
| `docs/user/play_texas_holdem.md` | 德州扑克人机对弈使用说明 |
| `docs/user/play_mahjong.md` | 麻将人机对弈使用说明（六变种 × 2/4 人） |
| `docs/user/play_undercover.md` | 谁是卧底使用说明（场景词对 × 4-12 人，text 发言桌游） |
| `docs/user/play_uno.md` | UNO 使用说明（六变体 × 2-10 人，rules/uno.json） |
| `docs/user/custom_games.md` | 自定义游戏使用说明（自然语言规则 / 模板变体 → 平台可对弈；四规则族） |

## 常用命令

```bash
python -m pytest tests/ -v                          # 跑测试
python train-cli/train.py --list                    # 查看游戏注册表一览
python train-cli/train.py --game all                # 训练所有已登记游戏的默认管线 (完整训练)
python train-cli/train.py --game moon_chess --solver hybrid   # Hybrid 训练 (产物 models/train/<game>/, 已 gitignore)
python train-cli/train.py --game stochastic_gomoku --solver hybrid --preset quick  # 快速演示校准 (quick 预设: 0.2 缩放)
python train-cli/train.py --game mahjong_guangdong --solver qmix,happo,maac  # MARL 训练
python -m train_cli --game texas_holdem --solver hybrid         # 等价桥接入口
python -m layer4_interface.frontend.vision.server            # 视觉识别应用 (8766, P2 并入平台)
python -m layer4_interface.frontend.platform.server          # 平台前端服务 (8770, 需先 npm run build; --learning-interval N 开启在线学习 auto-apply)
cd platform-frontend && npm install && npm run build         # 构建平台前端
cd platform-frontend && npm run dev                          # 平台前端开发模式 (5173, /api 代理到 8770)
python _gen_mahjong.py                                       # 重新生成 rules/mahjong.json (改 _gen_mahjong.py 后)
python _gen_werewolf.py                                      # 重新生成 rules/werewolf.json (改 _gen_werewolf.py 后)
python _gen_undercover.py                                    # 重新生成 rules/undercover.json (改 _gen_undercover.py 后)
python _gen_uno.py                                           # 重新生成 rules/uno.json (改 _gen_uno.py 后; 六变体 × 2-10 人)
```
