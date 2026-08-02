# Gavis 项目指南

## 项目概述

自适应策略游戏 AI Agent — 四层水平集成架构。

## 目录结构

```
rules/                 游戏规则 JSON (v5.0)
layer1_translator/    LLM 规则翻译层 (预留)
layer2_engine/        游戏引擎 (GameEngine + SolverAdapter)
layer3_solvers/       求解器 (MCTS/CFR/PPO/PSRO)
layer4_interface/     交互界面 (Binding/Encoding/Frontend 按应用分目录)
demos/                演示入口 + 统一基准
tests/                测试 (89 cases)
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

## 求解器注册

新求解器 → `layer3_solvers/base.py` 实现 `SolverBase` → 注册到 `__init__.py`

## 文档

| 文件 | 内容 |
|------|------|
| `docs/design/architecture.md` | 当前四层架构设计 (v0.2) |
| `docs/merge/01~06.md` | 六篇合并分析与方案文档 |
| `docs/design/gamerule/v4.1.md` | 规则语言设计 (v5.0: `docs/design/gamerule/v5.0.md`) |
| `docs/user/play_moon_chess.md` | 月亮棋人机对弈使用说明 |
| `docs/user/play_texas_holdem.md` | 德州扑克人机对弈使用说明 |

## 常用命令

```bash
python -m pytest tests/ -v                          # 跑测试
python -m demos.benchmark_all --game moon_chess     # 基准评测
python -m demos.demo_mcts --size 9 --budget 5000    # MCTS 演示
python -m demos.demo_texas_holdem --budget 1500     # 德州扑克 MCTS 演示
python -m layer4_interface.frontend.play_moon_chess.server   # 月亮棋人机对弈 (8765)
python -m layer4_interface.frontend.vision.server            # 视觉识别应用 (8766)
python -m layer4_interface.frontend.play_gomoku.server       # 随机五子棋人机对弈 (8767)
python -m layer4_interface.frontend.play_texas_holdem.server # 德州扑克人机对弈 (8768)
```
