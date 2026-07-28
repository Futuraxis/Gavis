# Gavis 项目指南

## 项目概述

自适应策略游戏 AI Agent — 四层水平集成架构。

## 目录结构

```
rules/                 游戏规则 JSON (v4.1)
layer1_translator/    LLM 规则翻译层 (预留)
layer2_engine/        游戏引擎 (GameEngine + SolverAdapter)
layer3_solvers/       求解器 (MCTS/CFR/PPO/PSRO)
layer4_interface/     交互界面 (Binding/Encoding/Frontend)
demos/                演示入口 + 统一基准
tests/                测试 (89 cases)
archive/              原始旧代码只读存档
docs/                 架构设计 + 六篇合并分析文档
```

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
| `docs/design/gamerule/v4.1.md` | 规则语言设计 |

## 常用命令

```bash
python -m pytest tests/ -v                          # 跑测试
python -m demos.benchmark_all --game moon_chess     # 基准评测
python -m demos.demo_mcts --size 9 --budget 5000    # MCTS 演示
python -m layer4_interface.frontend.app_server      # Web 前端
```
