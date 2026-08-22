# Gavis 项目指南

## 项目概述

自适应策略游戏 AI Agent — 四层水平集成架构。

## 目录结构

```
rules/                 游戏规则 JSON (v5.1, 零 BUILTIN: texas_holdem / mahjong / werewolf 纯 alias)
layer1_translator/    LLM 规则翻译层 (预留)
layer2_engine/        游戏引擎 (GameEngine + SolverAdapter)
layer3_solvers/       求解器 (MCTS/CFR/PPO/PSRO + mahjong/heuristic + marl/(QMix/HAPPO/MAAC))
layer4_interface/     交互界面 (Binding/Encoding/Frontend 按应用分目录)
demos/                演示入口 + 统一基准
tests/                测试 (551 cases)
platform-frontend/    平台前端 (React + Vite + TS, 构建产物 dist/ 已 gitignore)
data/                 运行时数据 (对局记录 data/matches/, 已 gitignore)
archive/              原始旧代码只读存档
docs/                 架构设计 + 六篇合并分析文档
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
- 层间通信只能通过 Protocol（`SolverAdapter`, `SolverBase`, `BaseBinding`）
- **禁止循环依赖**：Layer N 只能依赖 Layer N-1
- Layer 4 (Interface) 不依赖 Layer 3 (Solver)
- 规则 JSON 放在 `rules/` 顶层，不在层目录内
- **v5.1 零 BUILTIN**：规则自足，`BUILTIN_FUNCTIONS` 已退役；引擎从
  `rules["functions"]` 读取 alias 定义（`{"params": [...], "expr": {...}}`）
- 语言原语只从数学操作角度增加（`choose/range/sort/group/at/add/sub/...`），
  禁止游戏特供原语
- 判定遵循增量局部原则：围绕 `lastPlacedCell` / `lastDiscard` /
  `last_action` 做 O(1) 局部判定
- `rules/mahjong.json` 由 `_gen_mahjong.py` 生成（变种/人数由
  `MahjongAdapter` 注入 constants），改规则改生成器再重新生成
- `rules/werewolf.json` 由 `_gen_werewolf.py` 生成（人数/角色配比由
  `WerewolfAdapter` 注入 constants）；结算阶段（夜晚/放逐）用
  `chance` 模板 + `effectMap` 表达（explicit 概率 1.0），轮转映射在
  adapter 的 `get_current_player`（狼人杀 `speak` 用 text 参数预制能力）
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
| `docs/merge/01~06.md` | 六篇合并分析与方案文档 |
| `docs/design/security-notes.md` | 安全与性能决策记录（审计 3.6 修复项与暂缓项） |
| `docs/design/gamerule/v4.1.md` | 规则语言设计 (v5.0: `docs/design/gamerule/v5.0.md`, v5.1: `docs/design/gamerule/v5.1.md`) |
| `docs/user/play_moon_chess.md` | 月亮棋人机对弈使用说明 |
| `docs/user/play_texas_holdem.md` | 德州扑克人机对弈使用说明 |
| `docs/user/play_mahjong.md` | 麻将人机对弈使用说明（三变种 × 2/4 人） |

## 常用命令

```bash
python -m pytest tests/ -v                          # 跑测试
python -m demos.benchmark_all --game moon_chess     # 基准评测
python -m demos.train_hybrid --game all             # 三游戏训练 Hybrid 模型 (产物在 models/hybrid/, 已 gitignore)
python -m demos.demo_mcts --size 9 --budget 5000    # MCTS 演示
python -m demos.demo_texas_holdem --budget 1500     # 德州扑克 MCTS 演示
python -m layer4_interface.frontend.play_moon_chess.server   # 月亮棋人机对弈 (8765)
python -m layer4_interface.frontend.vision.server            # 视觉识别应用 (8766)
python -m layer4_interface.frontend.play_gomoku.server       # 随机五子棋人机对弈 (8767)
python -m layer4_interface.frontend.play_texas_holdem.server # 德州扑克人机对弈 (8768)
python -m layer4_interface.frontend.play_werewolf.server      # 狼人杀人机对弈 (8771, 需本地 ollama qwen3:8b)
python -m demos.demo_werewolf_llm                             # 狼人杀 LLM 自对弈演示
python -m layer4_interface.frontend.platform.server          # 平台前端服务 (8770, 需先 npm run build)
cd platform-frontend && npm install && npm run build         # 构建平台前端
cd platform-frontend && npm run dev                          # 平台前端开发模式 (5173, /api 代理到 8770)
python _gen_mahjong.py                                       # 重新生成 rules/mahjong.json (改 _gen_mahjong.py 后)
python _gen_werewolf.py                                      # 重新生成 rules/werewolf.json (改 _gen_werewolf.py 后)
python -m demos.train_marl --game mahjong_2p                # MARL 训练 (moon_chess/texas_holdem/mahjong_2p)
python -m demos.marl_tournament --game all                  # MARL 单循环赛 (产物 data/marl_tournament/)
python -m demos.marl_report                                 # 循环赛分析报告 (data/marl_tournament/REPORT.md)
```
