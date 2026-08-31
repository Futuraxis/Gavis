# Gavis 随机博弈模型

> 版本: 1.0 | 日期: 2026-07-28 | 对应规则版本: v5.0（**现行实现 v5.2**：variants/visibility 声明式，语法见 [gamerule/v5.2.md](gamerule/v5.2.md)；moon_chess/stochastic_gomoku 仍是 5.0.0 存量规则）

---

## 1. 形式化定义

Gavis 将一个游戏建模为 **部分可观测随机博弈（Partially Observable Stochastic Game, POSG）**，形式化为六元组：

$$\langle N, S, A, T, O, R \rangle$$

其中：

| 符号 | 含义 | 在 Gavis 中的对应 |
|------|------|-------------------|
| $N$ | 玩家集合 | `players` 数组 |
| $S$ | 状态空间 | `groundState` 声明存储结构 |
| $A$ | 联合动作空间 $A = A_1 \times \cdots \times A_n$ | `actions[].params` → domain → filter → cartesian |
| $T$ | 转移概率 $S \times A \times S \to [0,1]$ | `effectors` + `chance` |
| $O$ | 观察函数 $S \times N \to \text{Obs}$ | `visibility` 规则投影 |
| $R$ | 收益函数 $S \times A \times N \to \mathbb{R}$ | `utility` 规则 |

### 1.1 完美信息简化

对于完美信息游戏，$O$ 退化为恒等映射（每个玩家看到完整状态），Gavis 的模型退化为 **随机博弈（Stochastic Game, SG）**：

$$\langle N, S, A, T, R \rangle$$

当前实现**同时覆盖**：完美信息（moon_chess/stochastic_gomoku/mahjong）与
部分可观测 POSG（texas_holdem/werewolf/undercover/uno 有 `hiddenWorld` 或
`visibility` 声明式投影）。

---

## 2. 架构原则

### 2.1 核心原则

> **存模板，不存实例；存分布，不存树；存紧凑状态，不存图。**

- **状态**：存储最小化的基础事实（ground state），不存储推导信息
- **动作**：存储动作模板（`ActionTemplate`），不在 JSON 中枚举动作实例
- **随机性**：存储概率分布（`ChanceTemplate`），不在 JSON 中枚举随机历史
- **观察**：存储可见性规则（`VisibilityRule`），不在 JSON 中枚举信息集

### 2.2 双层状态架构

```
┌──────────────────────────────────────┐
│  Ground State (基础存储)              │
│  连续数组 / 标量 / 轻量结构            │
│  引擎真实持有、克隆、效果直接操作      │
│  大小 = O(游戏复杂度的下界)           │
├──────────────────────────────────────┤
│  Derived Views (推导视图)             │
│  坐标运算 / 枚举 / 正则表达式 / 算术   │
│  按需计算、查询在视图上过滤            │
│  大小 = 规则数量 (常数)               │
├──────────────────────────────────────┤
│  Query / Action / Observation        │
│  在 derived views 上展开动作和观察    │
│  效果直接修改 ground state            │
└──────────────────────────────────────┘
```

这种分层的意义：

- **存储效率**：MCTS 模拟中克隆一个 9 元素数组，而不是 9 个 node 对象
- **表达力**：`derivedViews` 可以用任意运算/正则推导视图，不受存储格式限制
- **解耦**：引擎的程序逻辑操作 ground state（机械确定），游戏设计者声明 derived views（语义层）

---

## 3. 状态空间 S

### 3.1 Ground State

状态的基础存储层，形式化为一个类型化元组：

$$S_{\text{ground}} = (\text{arrays}, \text{scalars}, \text{records})$$

- **Arrays**：连续内存块，O(1) 索引访问，浅拷贝克隆
- **Scalars**：标量值（回合数、阶段名、胜者）
- **Records**：小型键值结构（不超过 10 个字段）

通过 `groundState` 在 JSON 中声明：

```json
{
  "groundState": {
    "board": {
      "type": "array",
      "length": {"expr": "board_size * board_size"},
      "element": "player_id?"
    },
    "pieceOrder": {
      "type": "array",
      "mutable": true,
      "element": {"cell_id": "string", "player_id": "string"}
    },
    "env": {
      "turn": {"type": "string", "initial": "p_black"},
      "round": {"type": "int", "initial": 0},
      "phase": {"type": "string", "initial": "playing"},
      "winner": {"type": "string?", "initial": null}
    }
  }
}
```

`length` 可以是表达式，在初始化时求值一次。这避免了枚举。

### 3.2 Derived Views

从 ground state 通过推导规则生成的可遍历实体集合。每个 view 类似于一个**虚拟关系表**：

$$V_i = \text{derive}(\text{ground}, \text{rules}_i)$$

推导规则类型：

| 类型 | 描述 | 示例 |
|------|------|------|
| `grid` | 从一维数组按坐标推导二维实体 | `board[9] → 9 cells with (x, y)` |
| `enumerate` | 从数组按索引推导编号实体 | `deck[52] → 52 cards, card_0..card_51` |
| `regex` | 从字符串模式匹配推导实体 | 正则 `"cell_(\d+)_(\d+)"` 提取坐标 |
| `literal` | 从字面量推导（如玩家列表） | `["p_black", "p_white"]` |

```json
{
  "derivedViews": {
    "cell": {
      "from": {"array": "board", "type": "grid", "cols": {"const": 3}},
      "fields": {
        "id": {"template": "cell_{row}_{col}"},
        "occupant": {"get": ["$self", "value"]},
        "x": {"var": "$col"},
        "y": {"var": "$row"}
      }
    }
  }
}
```

推导视图按需惰性计算，查询时物化。效果不直接操作视图。

### 3.3 初始状态

`initialState` 描述 ground state 各字段的初始值：

```json
{
  "initialState": {
    "board": {"_fill": null},
    "pieceOrder": {"_fill": []},
    "env": {"_from": "groundState.env.initial"}
  }
}
```

初始化过程：
1. 解析 `groundState` 获取 schema
2. 对每个字段求初始值表达式（支持 `_fill`, `_from` 等构造）
3. 初始化时 `derivedViews` 不物化，首次查询时构建

---

## 4. 动作空间 A

### 4.1 动作展开

动作空间不枚举，而是通过模板 + 参数域 + 合法性条件生成：

$$A(s) = \{a \mid \text{template}(a) \in \text{actions}, \text{domain}(a, s) \neq \emptyset, \text{legal}(a, s) = \text{True}\}$$

展开流程：

```
对每个 ActionTemplate t:
  检查当前 phase 匹配 → t.phases
  确定 actor → eval(t.actor)
  对每个参数 p:
    计算 domain → eval(p.domain)    ← 在 derived views 上过滤
    应用 filter → eval(p.filter)    ← 逐项检验
  所有参数的笛卡尔积
  对每个组合:
    检查 legal → eval(t.legal)
    生成 canonicalKey → eval(t.canonicalKey)
    生成 ActionInstance
```

### 4.2 参数域

参数域引用 `queries`，queries 在 derived views 上过滤：

```json
"queries": {
  "empty_cells": {
    "view": "cell",
    "filter": {"eq": [{"get": ["occupant"]}, {"const": null}]}
  }
}
```

等价于 Python：
```python
[cell for cell in derive_view("cell", state) if cell.occupant is None]
```

---

## 5. 转移函数 T

### 5.1 确定性转移

由 `effectors` 描述。每个 effector 是一组操作序列，直接修改 ground state：

$$s' = f_e(s) \quad \text{其中 } e \text{ 是 effector, } f_e \text{ 是操作序列}$$

操作类型：

| 操作 | 效果 | 作用目标 |
|------|------|---------|
| `setIndex` | `array[index] = value` | Ground array |
| `append` | `array.push(value)` | Ground array |
| `trimByKey` | `array = array[-max:]`（按 key 分组） | Ground array |
| `setEnv` | `env.key = value` | Environment scalar |
| `inc` | `env.key += by` | Environment counter |
| `branch` | 条件分支 | 控制流 |
| `callEffect` | 调用另一个 effector | 子过程 |

### 5.2 随机转移

由 `chance` 描述。chance 节点在指定 phase 被触发，输出一个概率分布：

$$T(s, a, s') = \sum_{c \in C(s)} P_c(s) \cdot [s' = f_c(s)]$$

概率分布类型：

| 类型 | 描述 | 用例 |
|------|------|------|
| `explicit` | 枚举每个 outcome 的概率（`prob` 数值表） | 结算型随机（v5.2 常用 explicit 1.0 + `effectMap`） |
| `uniform` | 从候选集中均匀抽样 | 洗牌后抽牌 |

> v4.1 的 `weighted`（权重分布）已在 v5.0/v5.1 移除，v5.2 只保留
> `explicit` / `uniform` 两种形式。

```json
{
  "chance": [{
    "id": "vanish",
    "phases": ["vanish_check"],
    "probability": {
      "explicit": [
        {"outcome": "vanish", "prob": 0.5},
        {"outcome": "keep", "prob": 0.5}
      ]
    },
    "effectMap": {
      "vanish": "do_vanish",
      "keep": "do_keep"
    }
  }]
}
```

### 5.3 确定性边界

引擎保证：给定相同 seed、相同状态、相同动作，`apply_action` 始终产生相同后继。随机性只通过 `chance` / `sample_chance` 引入。

---

## 6. 观察函数 O

### 6.1 可见性投影

$$O(s, i) = \text{project}(s, i, \text{visibility})$$

`project` 对每个 derived view 的每个字段进行可见性判断：

```
对每个 derived view v:
  对 v 中的每个实体 e:
    对 e 的每个字段 f:
      如果 visibility 规则允许 i 看到 (e, f):
        保留 f
      否则:
        隐藏/掩码 f
```

Visibility 规则（v5.2 声明式；v4.1 的 `rules[] + fields + filter` 列表写法已废弃）：

```json
{
  "visibility": {
    "default": "partial",
    "my_role": {
      "when": {"eq": [{"get": ["$node._index"]}, {"call": ["player_index", "$viewer"]}]},
      "keep": true
    },
    "dead_roles": {
      "when": {"eq": [{"get": ["alive[$node._index]"]}, 0]},
      "keep": true
    },
    "env": {
      "seerResult": {"when": {"eq": [{"get": ["roles[$player_index($viewer)]"]}, "seer"]}}
    }
  }
}
```

可见性语义（v5.2）：

| 机制 | 含义 |
|------|------|
| `default: "public"` | 全部公开（完美信息） |
| `default: "partial"` | 默认隐藏，只有 `keep: true` 且 when 命中的行才放行 |
| 视图级 `when` + `keep` | 对视图实体逐行过滤（如 werewolf `my_role` 只对 `_index == $viewer` 的行） |
| `env` 级条目 | 对 env 字段做 `$viewer` 条件投影（如 `seerResult` 仅预言家可见） |

> v4.1 的 `private/hidden/masked` 五档与 `aggregateAs` 已在 v5.x 移除；
> 现行为「public 全放行 / partial 默认隐藏 + drop 行 + env 投影」。

### 6.2 完美信息简化与部分可观测现状

完美信息游戏（moon_chess/stochastic_gomoku/mahjong）用：

```json
{"visibility": {"default": "public"}}
```

`project_observation` / `get_observation` 返回完整的 derived views，等价于恒等投影。

**部分可观测游戏已是现行实现的一部分**（不再是"未来工作"）：

| 游戏 | 机制 |
|------|------|
| texas_holdem | `hiddenWorld` + 投影（v5.1.0） |
| werewolf | `visibility` 声明式：`roles`/`dead_roles`/`seerResult` 投影（v5.2） |
| undercover | `visibility`：他人牌身份隐藏，无 `my_role`（v5.2） |
| uno | `visibility`：他人手牌隐藏但保留张数（v5.2） |

---

## 7. 收益函数 R

终局状态下，对每个玩家计算收益：

$$R(s, i) = \sum_{r \in \text{utility}} [\text{player}_r = i \land \text{when}_r(s)] \cdot \text{value}_r(s)$$

```json
{
  "utility": [
    {"player": "p_black", "value": 1, "when": {"eq": [{"get": ["$env", "winner"]}, "p_black"]}},
    {"player": "p_black", "value": -1, "when": {"eq": [{"get": ["$env", "winner"]}, "p_white"]}},
    {"player": "p_black", "value": 0, "when": {"eq": [{"get": ["$env", "winner"]}, null]}}
  ]
}
```

支持零和博弈（如上）和一般和博弈。

---

## 8. 终局条件 Z

$$Z(s) = \bigvee_{t \in \text{terminal}} \text{condition}_t(s)$$

任一终局条件满足时游戏结束。无优先级——满足即终止。

---

## 9. 与旧模型的对比

| 维度 | 旧 v4.1 实现 | 新 v5.0/v5.2 模型 |
|------|-------------|-------------|
| 状态表示 | `_board` 特化数组 + `nodes` 字典 | 任意 ground arrays + 推导规则 |
| 实体声明 | 硬编码在 `create_gomoku_state` | `groundState` + `derivedViews` |
| 棋盘格 | `create_gomoku_state(bs)` 生成 | `grid(board, cols)` 推导 |
| 关系 | 不存在 | 外键字段 |
| 可见性 | 无 | 字段级投影（v5.2 起 `visibility` 声明式：public/partial + drop + env 投影） |
| 概率分布 | 仅 explicit | explicit / uniform（weighted 已移除） |
| 外部函数 | `check_five_in_row` 硬编码注册 | v5.1 起零 BUILTIN：`functions` 节纯 alias（`{"params", "expr"}`） |
| 变体/人数 | 硬编码 | v5.2 `variants` 声明式（options dict + 常量补丁 + trim） |
| 玩家 | `p_black`/`p_white` 字符串硬编码 | `players` 数组声明 |
| 阶段流转 | `set phase = "game_over"` goto 式 | 声明式 `phases[].next` |
| 求解器契约 | SolverAdapter 9 方法 | `GameEngine` 13 方法（L2→L3 唯一契约） |
| JSON 自足 | 否（需要引擎硬编码） | 是（JSON + GameEngine 即可） |

---

## 10. GameEngine 契约（L2→L3 唯一契约，取代 SolverAdapter）

v5.2 起 **SolverAdapter 已退役**（v4.1 时代的 9 方法 per-game 适配器接口，
`layer2_engine/interfaces/solver_adapter.py` 已不存在）。求解器统一消费
`GameEngine`（`layer2_engine/core/engine.py`）：

```python
class GameEngine:
    def __init__(self, rules, seed=None, variant=None, player_count=None,
                 allow_codegen=True): ...
    def create_initial_state(self) -> State: ...
    def get_node_type(self, state) -> NodeType: ...
    def get_current_player(self, state) -> str | None: ...
    def get_legal_actions(self, state) -> list[ActionInstance]: ...
    def apply_action(self, state, action) -> State: ...
    def get_chance_outcomes(self, state) -> list[ChanceOutcome]: ...
    def apply_chance(self, state, outcome) -> State: ...
    def sample_chance(self, state) -> tuple[ChanceOutcome, State]: ...
    def is_terminal(self, state) -> bool: ...
    def get_utility(self, state, player) -> float: ...
    def project_observation(self, state, viewer) -> Obs: ...
    def get_info_set_key(self, state, player) -> str: ...
    def eval_expr(self, expr, extra_ctx=None): ...
```

要点：

- 求解器只依赖 `GameEngine`，不依赖任何 per-game 适配器；所有游戏
  （含变体/人数解析、部分可观测投影）都从规则 JSON 声明驱动。
- `project_observation` 为部分可观测游戏提供投影（v5.2 `visibility`
  声明式）；`get_info_set_key` 输出 sha256 64 字符信息集键（非完整 JSON，
  旧 Hybrid cfr_table 需重训）。
- `allow_codegen=False` 时纯解释器路径（平台自定义游戏一律如此）。
