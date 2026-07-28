# Gavis 项目代码风格分析文档

> 文档日期: 2026-07-28
> 覆盖范围: 四个代码库的代码风格、命名规范、质量实践评估

---

## 目录

1. [Project A: Gavis Engine](#1-project-a-gavis-engine)
2. [Project B: Moon Chess AI (PSRO)](#2-project-b-moon-chess-ai-psro)
3. [Project C: Moon Chess Vision + PPO](#3-project-c-moon-chess-vision--ppo)
4. [代码风格总结对比](#4-代码风格总结对比)

---

## 1. Project A: Gavis Engine

### 1.1 整体印象

**科学计算/研究风格**。代码由有算法背景的人编写，注重正确性和性能，对工程规范（类型标注、格式化）关注较少。

### 1.2 命名规范

| 类别 | 惯例 | 示例 | 是否符合 PEP 8 |
|------|------|------|:---:|
| 类名 | PascalCase | `GameEngine`, `MCTSNode`, `ActionInstance` | ✅ |
| 方法名 | snake_case | `create_initial_state`, `apply_action` | ✅ |
| 函数名 | snake_case | `clone_state`, `check_five_in_row` | ✅ |
| 变量名 | snake_case | `top_visits`, `vanish_count` | ✅ |
| 常量 | UPPER_CASE | `SYMBOLS`, `COLOR_NAMES` | ✅ |
| 私有 | `_` 前缀 | `_iterate`, `_select_ucb1_key`, `_board` | ✅ |

命名总体良好，少数问题：
- `_board` 作为 "私有" 字段暴露在公开的 `state` dict 中，易误导调用者
- `ucb_c` 命名偏短，可改为 `ucb_exploration_constant`

### 1.3 类型标注

```python
# ⚠️ 方法签名有标注但不够精确
def create_initial_state(self) -> dict:     # 应标注为更具体的类型或 TypedDict
def get_node_type(self, state: dict) -> str: # 应标注 Literal['player','chance','terminal']
def apply_action(self, state: dict, action: ActionInstance) -> dict:

# ✅ 内部方法标注充分
def _select_ucb1_key(self, node: MCTSNode) -> Optional[str]:

# ❌ 关键状态字段缺失标注
# state['_board'] → list[Optional[str]] 但未标注
# state['env']['turn'] → 嵌套 dict 但未标注
```

**总体评价：** 60% 覆盖率。函数签名有标注，但复杂类型缺乏 TypedDict。

### 1.4 文档与注释

```python
# ✅ 类有 docstring
class MCTS:
    """Monte Carlo Tree Search with chance-node handling."""

# ✅ 方法有 docstring
def search(self, state: dict) -> Optional[ActionInstance]:
    """Run MCTS from the given state, return the best action."""

# ✅ 关键设计决策有注释
# `_board` is the source of truth — `nodes` dict is cleared and
# rebuilt lazily only when needed for display.

# ❌ demo.py 中的函数有一行 docstring 但缺少详细说明
def play_one_game(...) -> dict:
    """Play a single game with MCTS for both sides. Returns game stats."""
```

中文/英文混合用于注释，docstring 用英文，行内解释性注释用中文。风格不一致但可读。

### 1.5 代码组织

- ✅ 模块职责清晰：`engine.py` / `state_graph.py` / `expr_eval.py` 各司其职
- ✅ 文件行数合理 (~300-500 行)
- ✅ 求解器方法按阶段分组 (`# Selection`, `# Expansion`, `# Simulation`, `# Backpropagation`)
- ❌ `demo.py` 中 `render_board` / `play_one_game` / `run_tournament` 纠缠度较高
- ❌ MCTS 和 CFR 共享的 rollout 逻辑没有抽取成公用函数

### 1.6 精确保真 (Code Correctness)

| 检查点 | 状态 |
|--------|:---:|
| 无未使用的 import | ✅ |
| 无 `except: pass` | ✅ |
| 无魔法数字硬编码 | ✅ (配置为参数) |
| 有边界检查 | ⚠️ 部分 (engine.get_node_type 对未知 phase 返回 'terminal') |
| 资源泄漏风险 | ✅ (无外部资源) |
| 可重现性 | ✅ (seed 参数) |

### 1.7 示例代码片段分析

```python
# ✅ 好的风格: 显式、自文档化
def _iterate(self, root_state: dict, root: MCTSNode):
    """One full MCTS iteration: select → expand → simulate → backprop."""
    state = clone_state(root_state)
    node = root
    path: list = [(None, node)]
    while not node.is_leaf() and node.is_fully_expanded():
        ...
    if node.node_type != 'terminal' and not node.is_fully_expanded():
        node = self._expand(node, state)
    value = self._rollout(state)
    self._backpropagate(path, value)

# ❌ 可改进: action_stats 重复 search 逻辑
def action_stats(self, state: dict) -> list[tuple]:
    """Return (action, visits, avg_value) for display/debug."""
    root = MCTSNode(node_type=self.engine.get_node_type(state))
    # ^^ 与 search() 相同的初始化代码 — 应复用
    root_type = root.node_type
    if root_type == 'player':
        root.untried_actions = self.engine.get_legal_actions(state)
    elif root_type == 'chance':
        root.untried_outcomes = list(self.engine.get_chance_outcomes(state))
    for _ in range(self.budget):
        self._iterate(state, root)
    # ^^ 与 search() 相同的 search 循环 — 应复用
```

**风格评分：** ★★★☆☆ (3.5/5)

---

## 2. Project B: Moon Chess AI (PSRO)

### 2.1 整体印象

**研究原型/实验脚本风格**。代码由机器学习研究者编写，关注算法实验而非工程品质。

### 2.2 命名规范

| 类别 | 惯例 | 示例 | 是否符合 PEP 8 |
|------|------|------|:---:|
| 类名 | PascalCase | `MoonChessEnv`, `Agent` | ✅ |
| 函数名 | snake_case | `solve_nash`, `gamescape` | ✅ |
| 变量名 | snake_case | `nash_pi`, `expls`, `divs` | ✅ |
| 缩写 | 过度使用 | `pi`, `R`, `expl`, `div`, `Ne`, `niter` | ❌ 可读性差 |

### 2.3 类型标注

```python
# ❌ 完全没有类型标注
def solve_nash(R_matrix):    # 返回类型? 参数类型?
def estimate_reward(env, num_episodes, p1, p2, max_steps=200):
def gamescape(env, pi, Ne):
```

**总体评价：** 0% 覆盖率。没有任何函数或变量有类型标注。

### 2.4 文档与注释

```python
# ⚠️ 注释主要描述"代码在做什么"而不是"为什么这样做"
def gamescape(env, pi, Ne):
    R = np.zeros([len(pi), len(pi)])
    for i in tqdm(...):
        for j in range(len(pi)):
            if j <= i:
                R[i, j] = -R[j, i]    # 利用对称性，减少一半计算
                continue
            R[i, j] = estimate_reward(env, Ne, Agent(pi[i]), Agent(pi[j]))
    return R

# ❌ 危险的注释
# 我保留原始逻辑，但更标准的是用线性规划求解纳什均衡，此处保持原作者意图
# 为了兼容，我们仍用原作者的写法（可能有误，但先不改）
```

- 有注释，但主要是中文自然语言描述代码流程
- 缺少 docstring（没有函数有 `"""docstring"""`）
- 对错误的容忍态度（"可能有误，但先不改"）是工程红线

### 2.5 代码组织

- ❌ 166 行的 `PSRO/train.py` 包含所有逻辑：参数解析、PSRO 主循环、工具函数
- ❌ 函数顺序随意：`solve_nash` 在 `estimate_reward` 前面，但后者被前者调用（阅读顺序错乱）
- ❌ 变量名过度缩写伤害可读性：`pi`、`R`、`expl`、`div`、`Ne`
- ❌ `num_steps_per_iter` 函数内被重命名为参数 `num_steps_per_iter`，但局部又赋值给同名参数（可读性差）

### 2.6 代码质量问题

| 检查点 | 状态 |
|--------|:---:|
| import 排序 | ❌ 未排序 (标准库/第三方/本地混排) |
| 行长度 | ⚠️ 偶有超过 100 字符 |
| 魔法数字 | ✅ 全部参数化 |
| 死代码注释 | ✅ 无 |
| 可重现性 | ✅ (seed 参数，但默认 time.time()) |
| 全局变量 | ❌ `tmp` 在 PSRO_Q 中未使用（垃圾变量） |

### 2.7 示例代码片段分析

```python
# ❌ 最危险的代码: 线性规划约束定义两次
def solve_nash(R_matrix):
    D = R_matrix.shape[0]
    A_ub = -R_matrix
    b_ub = np.zeros(D)
    # ... 中间无关代码 ...
    A_ub = R_matrix    # ← 覆盖了前面的负号！
    b_ub = np.zeros(D)
    A_eq = np.zeros([D, D])    # ← 应为 (1, D)
    b_eq = np.zeros(D)
    A_eq[0,:] = 1
    b_eq[0] = 1
    c = np.ones(D)
    re = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,1))
    nash_p = np.maximum(re.x, 0)
    return nash_p

# 问题:
# 1. A_ub 被定义了两次，第一次完全没用到
# 2. A_eq 初始化为 [D×D] 全零但只需要 1 行
# 3. A_ub = R_matrix 意味着约束 R_matrix * x ≤ 0，这不正确（收益矩阵为正时需要 x≥0 约束）
# 4. 对负值的 np.maximum(..., 0) 掩盖了求解失败
```

**风格评分：** ★★☆☆☆ (1.5/5)

---

## 3. Project C: Moon Chess Vision + PPO

### 3.1 整体印象

**工程团队风格**。代码由习惯团队协作、有工程规范意识的开发者编写。注重代码可读性、可维护性和测试覆盖。

### 3.2 命名规范

| 类别 | 惯例 | 示例 | 是否符合 PEP 8 |
|------|------|------|:---:|
| 类名 | PascalCase | `PPOAgent`, `ActorCriticNetwork`, `RolloutBuffer` | ✅ |
| 方法名 | snake_case | `select_action`, `record_transition`, `compute_returns_and_advantages` | ✅ |
| 函数名 | snake_case | `action_index_to_cell_id`, `cell_id_to_action_index` | ✅ |
| 变量名 | snake_case | `state_vector`, `action_mask`, `legal_actions` | ✅ |
| 私有 | `_` 前缀 | `_validate_mask`, `_resolve_device`, `_build_age_map` | ✅ |
| 异常 | 描述性 PascalCase | `InvalidActionMaskError`, `VisionModelResponseError` | ✅ |

命名精准、一致、有描述性。

### 3.3 类型标注

```python
# ✅ 完整覆盖
class PPOAgent:
    def select_action(
        self,
        state_vector: np.ndarray,
        action_mask: np.ndarray,
        legal_actions: list[str] | None = None,
    ) -> tuple[int, float, float]:
        ...

    def record_transition(
        self,
        *,
        state: np.ndarray,
        action: int,
        action_mask: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
        next_value: float,
    ) -> None:
        ...

# ✅ 使用 Protocol 定义接口契约
class CellClassifier(Protocol):
    def classify(self, cell_image: np.ndarray) -> tuple[str | None, float]: ...

# ✅ 使用 Literal / TypeAlias (通过字符串)
# ✅ 复杂的 dict 类型有显式字段访问 (通过 GameStateAdapter)
```

**总体评价：** 95%+ 覆盖率。在所有公开接口上都有完整标注。

### 3.4 文档与注释

```python
# ✅ 模块 docstring
"""PPO Agent 实现。"""

# ✅ 类 docstring
class PPOAgent:
    """只负责 PPO 自己的动作选择与更新。"""

# ✅ 方法 docstring
def update(self) -> dict[str, float]:
    """Update policy using PPO clipped objective."""

# ✅ "为什么"注释而非"是什么"注释
def _build_age_map(self, piece_order: dict[str, list[dict]]) -> dict[str, int]:
    age_map: dict[str, int] = {}
    for entries in piece_order.values():
        sorted_entries = sorted(entries, key=lambda item: item["placedSeq"])
        for age, entry in enumerate(sorted_entries, start=1):
            age_map[entry["cellId"]] = age
    return age_map
    # 注意: age=1 是最新棋子, 不是最老棋子

# ✅ 测试文件也有 docstring
"""评估脚本。"""
```

注释语言以中文为主，清晰、准确。

### 3.5 代码组织

- ✅ 模块化程度高：5 个目录各司其职
- ✅ `__init__.py` 精心组织导出符号
- ✅ `PPOAgent` 的 `config` 使用 `@dataclass(slots=True)` 模式
- ✅ 测试文件与源文件一一对应
- ❌ `training/train_mnist_classifier.py` 位置错误（与项目无关）
- ❌ `binding/vision_binding.py` 和 `binding/qwen_vision.py` 职责边界模糊

### 3.6 代码质量实践

| 检查点 | 状态 |
|--------|:---:|
| import 排序 | ✅ 标准库→第三方→本地 |
| `from __future__ import annotations` | ✅ 全部文件 |
| `slots=True` dataclass | ✅ |
| 测试覆盖率 | ✅ pytest 套件 |
| 异常层次 | ✅ 7 种自定义异常 |
| 资源清理 | ✅ `try/finally` in `app_server.py` |
| Python 版本兼容 | ✅ `3.11+` |
| 格式化一致性 | ✅ 缩进统一，行长度控制 |

### 3.7 示例代码片段分析

```python
# ✅ 优秀的现代 Python 风格
@dataclass(slots=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 32

# ✅ 完整的 mask 校验
def _validate_mask(
    self,
    action_mask: np.ndarray,
    *,
    legal_actions: list[str] | None = None,
) -> torch.Tensor:
    mask = np.asarray(action_mask, dtype=np.float32)
    if mask.shape != (self.action_dim,):
        raise InvalidActionMaskError(...)
    if not np.any(mask > 0):
        raise InvalidActionMaskError(...)
    if legal_actions is not None:
        legal_indices = {cell_id_to_action_index(cell_id) for cell_id in legal_actions}
        mask_indices = {idx for idx, value in enumerate(mask.tolist()) if value > 0}
        if legal_indices != mask_indices:
            raise InvalidActionMaskError(...)
    return torch.as_tensor(mask, dtype=torch.float32, device=self.device)
```

**风格评分：** ★★★★★ (5/5)

---

## 4. 代码风格总结对比

### 4.1 风格雷达图

```
                        C: Vision+PPO
                    ★★★★★ ★★★★☆ ★★★★★
                 ┌─────────────────────┐
                 │   Type Annotations  │
                 │       ★            │
                 │  ★       ★         │
    Naming ──────┤    ★   ★           ├────── Docstrings
                 │     ★              │
                 │  ★       ★         │
                 │   ★★★  ★★★        │
                 │  Testing  │  Modularity│
                 └─────────────────────┘
                    ★★☆☆☆ ★★☆☆☆ ★★☆☆☆
                     A: Gavis Engine

                 ┌─────────────────────┐
                 │   Type Annotations  │
                 │       ☆            │
                 │  ☆       ☆         │
    Naming ──────┤    ☆   ☆           ├────── Docstrings
                 │     ☆              │
                 │  ☆       ☆         │
                 │   ☆☆☆  ☆☆☆        │
                 │  Testing  │  Modularity│
                 └─────────────────────┘
                    ☆☆☆☆☆ ☆☆☆☆☆ ☆☆☆☆☆
                     B: Moon PSRO
```

### 4.2 量化评分

| 维度 | 权重 | A: Gavis Engine | B: Moon PSRO | C: Vision+PPO |
|------|------|:---:|:---:|:---:|
| PEP 8 遵循 | 15% | 4/5 | 2/5 | 5/5 |
| 类型标注 | 15% | 3/5 | 0/5 | 5/5 |
| 命名质量 | 15% | 4/5 | 2/5 | 5/5 |
| 文档/docstring | 15% | 3/5 | 1/5 | 5/5 |
| 模块化 | 15% | 4/5 | 1/5 | 5/5 |
| 异常处理 | 10% | 2/5 | 1/5 | 5/5 |
| 测试覆盖 | 10% | 0/5 | 0/5 | 4/5 |
| 可维护性 | 5% | 3/5 | 1/5 | 5/5 |
| **加权总分** | **100%** | **3.05/5** | **1.05/5** | **4.90/5** |

### 4.3 风格特征总结

```
A: Gavis Engine
   类型: 算法研究者
   优点: 命名规范、逻辑清晰、注释聚焦设计决策
   弱点: 缺 TypedDict、缺测试、demo 层代码结构松散
   改进建议: 补 TypedDict、提取公用 rollout、为 state 定 Protocol

B: Moon PSRO
   类型: 快速原型编写者
   优点: 核心算法思路正确、全局有注释
   弱点: 无类型标注、命名过度缩写、隐式 Bug (线性规划)、无测试
   改进建议: 全面重构、至少加函数签名类型标注、修正线性规划

C: Vision+PPO
   类型: 工程型开发者
   优点: 现代 Python 实践、完整类型标注、测试覆盖、异常层次
   弱点: 训练脚本太短 (20 episodes)、混入无关文件
   改进建议: 移走 MNIST 训练、增加训练 episode 数、增加训练日志
```
