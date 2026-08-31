# Gavis 代码规范 v1.0

> 版本: 1.0 | 适配: Python 3.11+ | 格式化: ruff (line-length=120)

---

## 目录

1. [语言与运行时](#1-语言与运行时)ç
2. [格式化与 Lint](#2-格式化与-lint)
3. [导入规范](#3-导入规范)
4. [命名规范](#4-命名规范)
5. [类型标注](#5-类型标注)
6. [文档字符串](#6-文档字符串)
7. [类与继承](#7-类与继承)
8. [错误处理](#8-错误处理)
9. [测试规范](#9-测试规范)
10. [`__init__.py` 与模块结构](#10-__init__py-与模块结构)
11. [注释与分隔线](#11-注释与分隔线)
12. [层间通信与依赖](#12-层间通信与依赖)
13. [git 与提交规范](#13-git-与提交规范)

---

## 1. 语言与运行时

### 1.1 Python 版本

- **目标版本**: Python 3.11+
- **强制要求**: 每个 `.py` 文件第一行应为 `from __future__ import annotations`

```python
"""Module docstring here."""

from __future__ import annotations
```

### 1.2 编码

- 所有源文件使用 UTF-8 编码（Python 3 默认，无需显式声明）
- 字符串中按需使用中文，但标识符、注释中的术语、关键词应使用英文

### 1.3 文件头

每个 `.py` 文件必须包含模块级文档字符串：

```python
"""SolverBase — abstract base for all Layer 3 solvers.

Every solver (MCTS, CFR, PPO, PSRO) implements this interface so that
demos, benchmarks, and the auto-selector can treat them uniformly.
"""
```

- 第一行：简短标题（不超过 80 字符）
- 空一行后跟详细描述（可选）
- 使用第三人称陈述语气

---

## 2. 格式化与 Lint

### 2.1 工具链

- **格式化器**: `ruff format`
- **Linter**: `ruff check`
- **行长度**: 120 字符（已在 `pyproject.toml` 中声明）
- **目标版本**: py311

### 2.2 `pyproject.toml` 配置（参考）

```toml
[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]
```

### 2.3 通用格式规则

- 缩进：4 空格（禁止 Tab）
- 空行：
  - 顶级定义（类/函数）之间：2 空行
  - 方法之间：1 空行
  - 逻辑段落之间可加空行
- 行尾禁止空白字符
- 文件末尾以空行结尾

### 2.4 禁止的写法

- 禁止使用 `from module import *`
- 禁止使用 `# type: ignore` 而不附带注释说明理由
- 禁止使用 `eval()` / `exec()`（表达式求值使用 `ExprEvaluator`）
- 禁止硬编码 magic number — 使用具名常量

### 2.5 推荐写法

- 优先使用 `list[...]` / `dict[...]` 而非 `List[...]` / `Dict[...]`（Python 3.9+）
- 优先使用 `X | Y` 而非 `Optional[X]`（Python 3.10+）
- 优先使用 `str.removeprefix` / `.removesuffix` 而非切片
- 优先使用 `pathlib.Path` 而非 `os.path`
- 优先使用 f-string 而非 `%` 格式化 或 `.format()`

---

## 3. 导入规范

### 3.1 分组顺序

每个导入块之间空一行，三组顺序如下：

```python
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import (
    State,
    ActionInstance,
    ChanceOutcome,
)
from ..base import SolverBase, SolverConfig
```

1. **标准库** （`os`, `sys`, `math`, `dataclasses`, `typing`, …）
2. **第三方库** （`numpy`, `pydantic`, `torch`, `httpx`, …）
3. **项目内部导入** （`layer2_engine.*`, `..base`, …）

可选导入（try/except）应紧贴对应的第三方分组之后。

### 3.2 导入风格

- **层内部引用**：使用相对导入（`from ..base import SolverBase`）
- **跨层引用**：使用绝对导入（L2→L3 契约经 `from layer2_engine.core.engine import GameEngine`；L1/L4 协议同理，如 `from layer1_translator.protocol import TranslatorProtocol`）
- **仅导入需要的符号**，避免 `from module import *`
- **多行导入**：当符号超过 3 个时使用带括号的多行形式

### 3.3 `__all__`

模块公开的 API 通过 `__all__` 显式列出：

```python
__all__ = [
    "SolverBase",
    "SolverConfig",
    "MCTS",
    "MCTSConfig",
    "CFR",
    "CFRConfig",
]
```

---

## 4. 命名规范

| 类别 | 规范 | 示例 |
|------|------|------|
| 模块/包 | 小写+下划线 | `state_graph.py`, `moon_state_encoder.py`, `expr_eval.py` |
| 类 | PascalCase | `GameEngine`, `SolverBase`, `MCTSNode` |
| 函数/方法 | snake_case | `create_initial_state`, `select_action` |
| 变量 | snake_case | `legal_actions`, `info_sets` |
| 常量 | UPPER_CASE | `REQUIRED_TOP_LEVEL`, `FEATURE_DIM` |
| 私有成员 | `_` 前缀 | `_register_functions`, `_build_age_map` |
| "受保护"成员 | `_` 前缀（即 Python 中无真正的 protected） | — |
| 类型变量 | PascalCase 或 `_T` 后缀 | `State = dict[str, Any]` |
| 异常类 | PascalCase + `Error` 后缀 | `BindingError`, `ImageLoadError` |

### 4.1 特殊约定

- **Protocol 类**：以 `…Protocol` 或描述性名词命名（`TranslatorProtocol`, `BaseBinding`, `SolverProvider`；注意 L2→L3 契约是具体类 `GameEngine`，不是 Protocol）
- **私有模块**：以下划线开头（当前项目中不使用，仅在必要时引入）
- **测试类**：以 `Test` 开头
- **测试方法**：以 `test_` 开头，snake_case

---

## 5. 类型标注

### 5.1 原则

- **全覆盖**：所有函数参数和返回值必须标注类型（`def __init__` 标注返回值 `-> None`）
- **变量标注**：复杂变量建议标注（`errors: list[str] = []`）
- **`None` 处理**：优先使用 `X | None` 而非 `Optional[X]`

```python
def select_action(self, state: State) -> ActionInstance | None: ...
```

- **集合类型**：使用 `list[X]`, `dict[str, X]`, `set[X]` 而非大写形式

### 5.2 类型别名

对于重复使用的复杂类型，定义顶层类型别名：

```python
State = dict[str, Any]
NodeType = Literal["player", "chance", "terminal"]
Obs = dict[str, Any]
```

### 5.3 泛型

- **`dict` 字面量**：标注其值类型（`age_map: dict[str, int] = {}`）
- **`list` 字面量**：标注其元素类型（`errors: list[str] = []`）
- **`@dataclass` 字段**：每个字段必须标注类型

---

## 6. 文档字符串

### 6.1 模块级

每个 `.py` 文件必须包含模块 docstring：

```python
"""CFR (Counterfactual Regret Minimization) solver.

Uses External Sampling MC-CFR with depth-limited recursion and
rollout-based leaf evaluation.  Implements ``SolverBase``.
"""
```

### 6.2 类级

```python
class MCTS(SolverBase):
    """Monte Carlo Tree Search with chance-node handling."""

    def __init__(self, engine: GameEngine, config: SolverConfig | None = None): ...
```

或用更详细的格式：

```python
class SolverBase(ABC):
    """Every solver in Layer 3 implements this interface.

    Usage::

        solver = MCTS(engine, SolverConfig(seed=42))
        action = solver.select_action(state)
        metrics = solver.train(episodes=100)
        solver.save("model.pt")
    """
```

### 6.3 方法级

- **公开方法**：必须包含 docstring（单行或详细）
- **私有方法**：可选，必要时添加注释而非 docstring
- **重写方法**：可以不写 docstring（继承父类的），但如果行为有差异需加

格式：

```python
def select_action(self, state: State) -> ActionInstance | None:
    """Return the best action for ``state``, or None if no legal moves."""
    ...


def train(self, episodes: int, **kwargs) -> SolverMetrics:
    """Run training for ``episodes`` self-play or simulated episodes.

    Returns training metrics (win rate, average return, etc.).
    """
    ...
```

- 使用 `` `` `` 引用参数、返回值、标识符（reStructuredText 风格）
- `:param:`, `:returns:` 等 rst 指令**可选**；简短方法可用单行
- 复杂逻辑必须附 inline comment 说明「为什么」

---

## 7. 类与继承

### 7.1 抽象基类（ABC）

```python
from abc import ABC, abstractmethod


class SolverBase(ABC):
    @abstractmethod
    def select_action(self, state: State) -> ActionInstance | None: ...
```

- 抽象方法体使用 `...`（Ellipsis）而非 `pass`
- 非抽象方法体使用 `pass` 或实现体

### 7.2 Protocol

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class TranslatorProtocol(Protocol):
    def translate(self, request: TranslateRequest) -> TranslateResponse: ...
```

- Protocol 方法体使用 `...`
- 加上 `@runtime_checkable` 以便 `isinstance()` 检查
- Protocol 中不要写 `__init__`

### 7.3 Dataclass

```python
@dataclass
class MCTSConfig(SolverConfig):
    budget: int = 5000
    ucb_c: float = 1.414
    rollout_depth: int = 20


@dataclass
class SolverMetrics:
    episodes: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)
```

- 可变默认值必须使用 `field(default_factory=...)`
- 继承 dataclass 时注意父类的 `@dataclass` 装饰器

### 7.4 继承原则

- 避免过深的继承层次（不超过 3 层）
- 优先使用 Protocol 或组合而非继承
- 层间通过 Protocol 耦合，而非通过具象类

---

## 8. 错误处理

### 8.1 自定义异常

为每个层定义基础异常，再派生子类：

```python
class BindingError(Exception):
    """Base binding error."""


class ImageLoadError(BindingError):
    """Image could not be loaded."""


class InvalidBoardError(BindingError):
    """Board layout is invalid."""
```

- 继承层次不超过 2 层（`Base → Specific`）
- 异常名以 `Error` 结尾
- docstring 描述触发条件

### 8.2 异常处理原则

- **捕获具体异常**，禁止裸 `except:`
- **只在能恢复或需要转换的地方捕获**
- 顶层（Demo/CLI/API）捕获并转成用户可读的信息
- 层间协议方法不因异常改变语义

```python
# 好的写法
try:
    result = risky_operation()
except ImageLoadError:
    raise BindingError("Failed to process image") from None

# 好的写法（可选导入）
try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
```

---

## 9. 测试规范

### 9.1 框架

- 使用 `pytest`（已在 `pyproject.toml` 中配置）
- 运行方式：`python -m pytest`

### 9.2 文件结构

```
tests/
├── test_layer1_translator.py
├── test_layer2_engine/
│   ├── __init__.py
│   ├── test_moon_chess.py
│   └── test_engine.py
├── test_layer3_solvers/
│   ├── __init__.py
│   └── test_solvers.py
├── test_layer4_interface/
│   ├── __init__.py
│   └── test_binding.py
├── test_integration.py
```

- 测试文件以 `test_` 前缀命名
- 测试类以 `Test` 开头
- 测试函数以 `test_` 开头

### 9.3 组织方式

```python
"""Tests for GameEngine (Layer 2, rules-driven)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from layer2_engine.core.engine import GameEngine

RULES_DIR = Path(__file__).resolve().parents[3] / "rules"


@pytest.fixture
def engine() -> GameEngine:
    rules = json.loads((RULES_DIR / "moon_chess.json").read_text(encoding="utf-8"))
    return GameEngine(rules, seed=42)


class TestMoonChessBasics:
    def test_create_initial_state(self, engine: GameEngine):
        state = engine.create_initial_state()
        assert state["env"]["turn"] == "p_black"

    def test_get_node_type(self, engine: GameEngine):
        state = engine.create_initial_state()
        assert engine.get_node_type(state) == "player"


class TestMoonChessGameplay:
    def test_place_piece(self, engine: GameEngine): ...
```

### 9.4 规则

- 测试必须独立可重复（固定 seed）
- 测试数据使用 **fixture** 而非 setup/teardown 方法
- Protocol 符合性检查用 `isinstance(obj, Protocol)`（带 `@runtime_checkable`）
- 使用 `@pytest.mark.skipif` 处理可选依赖
- 测试方法必须标注返回值类型 `-> None`
- 优先测试行为（behavior）而非实现细节

### 9.5 可选依赖跳过

```python
try:
    from layer3_solvers.ppo import PPOSolver, PPOConfig

    _HAS_TORCH = True
except ImportError:
    PPOSolver = None
    PPOConfig = None
    _HAS_TORCH = False


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestPPO:
    def test_select_action_no_training(self, engine: GameEngine): ...
```

---

## 10. `__init__.py` 与模块结构

### 10.1 模块 docstring

每个 `__init__.py` 应有简短描述：

```python
"""Layer 2: Env/Engine — declarative, adapter-free game engine.

Loads ``rules.json`` (v5.2, zero BUILTIN + variants/visibility declarative)
and provides a full game runtime that all Layer 3 solvers consume via the
``GameEngine`` contract (no per-game adapters, no ``interfaces/``).
"""
```

### 10.2 导出

```python
from .core.engine import GameEngine
from .core.state_graph import (
    State,
    NodeType,
    ActionInstance,
    ChanceOutcome,
)

__all__ = [
    "GameEngine",
    "State",
    "NodeType",
    "ActionInstance",
    "ChanceOutcome",
]
```

- `__init__.py` **不要**包含业务逻辑
- 只做导入和重新导出
- `__all__` 与导入保持同步

### 10.3 可选导入

当依赖可选时，在 `__init__.py` 中使用 try/except：

```python
try:
    from .ppo.solver import PPOSolver, PPOConfig
except ImportError:
    PPOSolver = None
    PPOConfig = None

__all__ = [...]
if PPOSolver is not None:
    __all__.extend(["PPOSolver", "PPOConfig"])
```

---

## 11. 注释与分隔线

### 11.1 小节分隔线

用注释分隔线组织类内部的方法：

```python
# ── Required ──────────────────────────────────────────────────


@abstractmethod
def select_action(self, state: State) -> Optional[ActionInstance]: ...


# ── Optional (save/load) ──────────────────────────────────────


def save(self, path: str) -> None:
    pass


# ── Name ───────────────────────────────────────────────────────


@property
def name(self) -> str:
    return type(self).__name__
```

### 11.2 注释风格

- `#` 后空一格再写文字
- 复杂逻辑需要解释「为什么」而非「是什么」
- 需要强调时使用 `# NOTE:`, `# HACK:`, `# TODO:`, `# FIXME:`

```python
# NOTE: 空棋盘上所有格子合法
return [action for action in self._generate_actions()]

# TODO: 支持非方阵棋盘
```

---

## 12. 层间通信与依赖

### 12.1 硬性规则

- **禁止循环依赖**：Layer N 只能依赖 Layer N-1
- **Layer 4 (Interface) 原则上不依赖 Layer 3 (Solver)**（唯一例外：
  `layer4_interface/botzone/mahjong_format.py` 直引 `SolverConfig` +
  `MahjongHeuristicAI` 的 Botzone 薄适配边界）
- 层间通信只能通过契约（L2→L3: `GameEngine` 具体类；L3: `SolverBase`；
  L4: `BaseBinding` / `SolverProvider` 注入协议）

### 12.2 依赖方向

```
Layer 1 (Translator)          ──→  Layer 2 (Engine)  (校验通道: engine_validator
                                  = schema + smoke_validate(variants="all"))
Layer 2 (Engine)              ──→  (无 per-game 适配器；v5.2 规则自足)
Layer 3 (Solver)               ──→  Layer 2 (契约: GameEngine, 13 方法)
Layer 4 (Interface)            ──→  Layer 2 (契约: BaseBinding / VisionBridge)
Layer 4 (Interface)            ──→  (不依赖 Layer 3; 例外: botzone/mahjong_format.py)
```

### 12.3 Protocol 规则

- 定义在 **消费方** 所在的层
- `GameEngine` 是 Layer 2 的**具体类**（非 Protocol；v5.2 起无 per-game
  `SolverAdapter`），被 Layer 3 求解器消费
- `SolverBase` 定义在 Layer 3，被 train-cli 工厂与 L4 装配消费
- `BaseBinding` 定义在 Layer 4，被测试和前端消费
- `TranslatorProtocol`（L1）、`SolverProvider`（L4 注入点）是 Protocol，
  分别被 L1 消费者与 L4 装配消费

---

## 13. git 与提交规范

### 13.1 分支

- 主分支：`main`
- 功能分支：`feat/<description>`
- 修复分支：`fix/<description>`
- 重构分支：`refactor/<description>`

### 13.2 提交信息

```
<type>: <简短描述>

<可选详细说明>

Co-Authored-By: Claude <noreply@anthropic.com>
```

类型：
- `feat` — 新功能
- `fix` — 修复
- `docs` — 文档
- `style` — 格式
- `refactor` — 重构
- `test` — 测试
- `chore` — 构建/工具

### 13.3 提交前检查

- 通过 ruff 格式化 + lint
- 通过 pytest（所有测试）
- 无调试代码（`print`, `breakpoint()`）

---

## 附录：术语表

| 英文 | 中文 | 说明 |
|------|------|------|
| Adapter | 适配器 | 连接两个独立模块的桥梁 |
| Binding | 绑定 | 将外部输入（图像、VLM）转为结构化数据 |
| Engine | 引擎 | 游戏规则解释器 |
| Protocol | 协议 | Python `typing.Protocol` 定义的接口契约 |
| Solver | 求解器 | 决策算法（MCTS/CFR/PPO/PSRO） |
| Translator | 翻译器 | 将自然语言规则转为 rules.json |
| Layer | 层 | 架构中的水平层次 |
| Bridge | 桥接 | 连接 Layer 4 与 Layer 2 的纯函数 |
| Rollout | 推演 | 从当前状态模拟到终局 |
