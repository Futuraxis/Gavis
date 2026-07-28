# Gavis 项目架构优缺点分析

> 文档日期: 2026-07-28
> 版本: v2.0 (更正版)
> 分析框架: 从"四层架构"视角评估各代码库的贡献与缺口

---

## 目录

1. [分析框架](#1-分析框架)
2. [按层评估现有代码](#2-按层评估现有代码)
3. [Project A: Gavis Engine (Layer 2 + 部分 Layer 3)](#3-project-a-gavis-engine-layer-2--部分-layer-3)
4. [Project B: Moon Chess PSRO (Layer 3)](#4-project-b-moon-chess-psro-layer-3)
5. [Project C: Vision + PPO (Layer 3 + Layer 4)](#5-project-c-vision--ppo-layer-3--layer-4)
6. [整体架构缺口分析](#6-整体架构缺口分析)

---

## 1. 分析框架

以下按照**原始架构愿景的四层**来评估现有代码：

```
Layer 1: Translator     — LLM 将规则翻译为 DSL/JSON      [所有代码: ❌ 未实现]
Layer 2: Env/Engine     — 从 JSON 提供游戏运行环境        [gavis/core/: ✅ 有]
Layer 3: Solver          — 多种算法训练/搜索 AI           [各 solvers: ⚠️ 部分]
Layer 4: Interface      — VLM 从外界识别 + 给出建议       [binding/: ⚠️ 部分]

在线自学习闭环           — 四层联动，持续进化              [所有代码: ❌ 未实现]
```

---

## 2. 按层评估现有代码

| 层 | Gavis Engine | Moon PSRO | Vision+PPO |
|:---:|:---:|:---:|:---:|
| **Layer 1 (Translator)** | ❌ | ❌ | ❌ |
| **Layer 2 (Engine)** | ✅ `GameEngine` 完整实现 | ❌ 不用 Engine | ❌ 不用 Engine |
| **Layer 3 (Solver)** | ✅ MCTS + CFR | ⚠️ PSRO (缺依赖) | ✅ PPO (训练不足) |
| **Layer 4 (Interface)** | ❌ 只有 CLI | ❌ 只有 CLI | ✅ Image/Vision Binding |
| **自学习闭环** | ❌ | ❌ | ❌ |

**核心结论：** 三个项目正好拼出 Layer 2→3→4 的骨架，但 Layer 1 和自学习闭环完全缺失。

---

## 3. Project A: Gavis Engine (Layer 2 + 部分 Layer 3)

### 3.1 在四层架构中的位置

```
Layer 1: Translator   ─── ❌
Layer 2: Env/Engine   ─── ✅ 主贡献: GameEngine + rules.json
Layer 3: Solver        ─── ✅ MCTS + CFR 实现
Layer 4: Interface    ─── ❌ CLI 仅用于调试
```

### 3.2 优点 (按层评估)

**Layer 2 贡献:**

1. **声明式规则引擎是 Layer 1→Layer 2 的桥梁**：`rules.json` 格式是 Translator (LLM) 的输出目标和 Engine 的输入。只要 Translator 能生成合法的 `rules.json`，新游戏就自动获得了完整的运行环境。

2. **v4.1 规则格式有足够表达力**：actions / effects / chance / phases / terminal / utility 的组合可以覆盖大部分回合制策略游戏的语义。

3. **`GameEngine` 的 `SolverAdapter` 接口族**：`get_observation()` 和 `get_info_set_key()` 方法的存在说明 Engine 已经预见到了 RL 求解器和 CFR 的需求。

4. **Chance 节点的一等支持**：对于"掷骰子""抽牌""随机消失"等随机性，Engine 原生支持，意味着 Translator 翻译带随机性的游戏规则时没有障碍。

**Layer 3 贡献:**

5. **MCTS 支持 chance node**：不是所有 MCTS 实现都处理随机节点，这份实现正确地在 player 和 chance 节点之间切换。

6. **CFR 使用 External Sampling**：在不完美信息博弈扩展时不需要重写框架。

### 3.3 缺点 (按层评估)

**Layer 2 缺口:**

1. **月亮棋的 `pieceOrder` 语义不能在 rules.json 中表达**。当前 engine 的状态模型是"cell 有 occupant"，没有"list of pieces in order"。需要扩展 `state.env` 或增加新的 effect 操作。

2. **`clone_state()` 的 nodes 重建策略**在状态中有复杂数据时（如 pieceOrder 的列表）可能出错。

3. **只有一种游戏 (stochastic_gomoku) 验证了 Engine 的正确性**。Engine 能否跑月克棋、围棋、黑白棋等尚未验证。

**Layer 3 缺口:**

4. **求解器与 Engine 的耦合是硬编码的**（直接 import `GameEngine`），如果 Layer 4 需要给 Solver 传入从 VLM 识别出的 `GameState`，当前的硬耦合可能成为障碍。

5. **无公共求解器接口**——MCTS 和 CFR 各自定义了不同的方法签名，新增求解器没有模板可遵循。

**架构缺口:**

6. **完全看不到 Layer 4 的存在**。Engine 没有被设计为从"截图"状态启动，`create_initial_state()` 总是从空棋盘开始。如果要支持"中局截图 → 加载状态 → 决策"，需要额外的方法。

---

## 4. Project B: Moon Chess PSRO (Layer 3)

### 4.1 在四层架构中的位置

```
Layer 1: Translator   ─── ❌
Layer 2: Env/Engine   ─── ❌ 自己实现 Gym 风格环境
Layer 3: Solver        ─── ✅ PSRO 主算法
Layer 4: Interface    ─── ❌ CLI 仅用于调试
```

### 4.2 优点

**Layer 3 贡献:**

1. **PSRO 算法在 Layer 3 中是最具"学习"深度的**。MCTS 是纯搜索（无学习），CFR 是离线学习（不支持在线增量），PPO 是策略梯度（需要大量交互）。PSRO 的策略池 + 纳什均衡方法在理论上是"自适应学习"的最佳候选——新策略持续加入、旧策略持续评估、均衡不断逼近。

2. **`exploitability` 监控**是评估求解器质量的重要指标。这是其他三个求解器都不具备的能力。

3. **线性规划纳什求解**虽然在实现上有瑕疵，但方向正确——当部署 Layer 1 (Translator) 后，对于任何零和博弈，这个求解器都可以直接使用。

### 4.3 缺点

**Layer 2 缺口:**

1. **自建 Gym 环境不与 Engine 对接**。这意味着如果 Layer 1 生成 rules.json，PSRO 无法直接消费它。需要额外的适配工作。

**Layer 3 缺口:**

2. **`Agent` 类缺失**、`MoonChessEnv` 缺失——两个关键依赖不在仓库中，代码不可运行。

3. **表格方法无法扩展到更大的游戏**。如果 Layer 1 翻译了一个 9×9 游戏，PSRO 的表格 Q 会直接内存溢出。

**架构缺口:**

4. **完全不支持 Layer 4**。没有考虑从 VLM 识别出的输入启动。

---

## 5. Project C: Vision + PPO (Layer 3 + Layer 4)

### 5.1 在四层架构中的位置

```
Layer 1: Translator   ─── ❌
Layer 2: Env/Engine   ─── ❌ MockMoonEnv 是独立实现
Layer 3: Solver        ─── ✅ PPO 实现完整
Layer 4: Interface    ─── ✅ 唯一实现了 Layer 4 的代码
```

### 5.2 优点

**Layer 4 贡献 (这是项目中最独特的价值):**

1. **`ImageBinding` 是 Interface Layer 的雏形**：OpenCV 模板匹配实现了从截图到 `Observation` 的转化。虽然不是最终的 VLM 方案，但验证了整个管线是可行的。

2. **`VisionLLMBinding` + `QwenVisionClient`** 展示了 VLM 接入路径，代码结构便于替换为任何视觉大模型。

3. **`StateTracker`** 解决了关键问题——单帧无法得知棋子顺序，需要跨帧推断。这对于 Interface Layer 来说是核心能力。

4. **`app_server.py` + 前端** 提供了 Web 界面，是整个项目中唯一有图形界面的部分。

**Layer 3 贡献:**

5. **PPO 实现质量高**：GAE、PPO-clip、mini-batch、梯度裁剪、action mask——这些组件在在线自学习场景中都是必须的。

### 5.3 缺点

**Layer 2 缺口:**

1. **`MockMoonEnv` 不是 Engine**。它包含了写死的月亮棋规则，没有使用 rules.json。如果 Layer 1 翻译了别的游戏，现有的 Binding + PPO 管线无法运行。

**Layer 3 缺口:**

2. **训练量严重不足** (20 episodes)。PPO 的潜力完全没有释放。

3. **对手只有 RandomAgent**。PPO 从未与强策略对弈，产生的策略质量低。

**Layer 4 缺口:**

4. **ImageBinding 硬编码 3×3**。无法处理任意大小的棋盘，离"通用策略游戏 AI"的目标有距离。

5. **VisionLLMBinding 依赖外部 API key**。硬编码在 README 中有安全风险。

**架构缺口:**

6. **Interface 与 Solver 没有打通**。`app_server.py` 只做识别，识别结果没有送入 Solver 做决策。项目目前只有"看"的能力，没有"想"的能力。

---

## 6. 整体架构缺口分析

### 6.1 各层完成度

```
Layer 1 (Translator):      ░░░░░░░░░░   0%  完全未实现
Layer 2 (Env/Engine):      ██████░░░░  60%  引擎有了，但只有一种游戏验证过
Layer 3 (Solver):          ████░░░░░░  40%  四种算法各缺一部分
Layer 4 (Interface):       ████░░░░░░  40%  视觉有了，但没打通 Solver
在线自学习闭环:             ░░░░░░░░░░   0%  完全未实现
```

### 6.2 各代码对四层的贡献

| 代码 | Layer 1 | Layer 2 | Layer 3 | Layer 4 |
|------|:-------:|:-------:|:-------:|:-------:|
| Gavis Engine | — | ✅ GameEngine | ✅ MCTS, CFR | — |
| Moon PSRO | — | — | ✅ PSRO (缺依赖) | — |
| Vision+PPO | — | — | ✅ PPO (训练不足) | ✅ Binding |
| **整体缺口** | **LLM 翻译** | **月亮棋引擎支持** | **统一接口 + 训练量** | **打通 Solver** |

### 6.3 合并的首要任务

从四层架构视角重新排列合并优先级：

```
P0 (必须合并):
  统一 Layer 2 → Layer 3 的接口 (SolverAdapter)
  让三种求解器都能消费 GameEngine 的输出
  修复 PSRO 的缺失依赖

P1 (应该合并):
  补充月亮棋 rules.json
  让 Interface 层打通 Solver 层 (识别 → 决策)
  统一四份代码的目录结构

P2 (可以合并):
  PPO 训练量提升到有意义水平
  Layer 1 Translator 的探索
  在线自学习闭环的架构预留
```
