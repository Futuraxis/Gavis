# Gavis 合并架构设计与迁移方案文档

> 文档日期: 2026-07-28
> 版本: v2.0 (更正版)
> 项目整体架构: 四层自适应策略游戏 AI Agent

---

## 目录

1. [合并目标与原则](#1-合并目标与原则)
2. [四层架构总览](#2-四层架构总览)
3. [架构方案 A: 四层水平集成 (Horizontal Layers)](#3-架构方案-a-四层水平集成-horizontal-layers)
4. [架构方案 B: 桥接模式 (Bridge Pattern)](#4-架构方案-b-桥接模式-bridge-pattern)
5. [架构方案 C: 微服务式 (Microservice)](#5-架构方案-c-微服务式-microservice)
6. [推荐方案与理由](#6-推荐方案与理由)
7. [详细迁移计划](#7-详细迁移计划)
8. [特殊挑战: online learning 闭环](#8-特殊挑战-online-learning-闭环)

---

## 1. 合并目标与原则

### 1.1 认清现状：三个代码库对应四层架构的不同层

```
原始架构愿景:                        当前代码实际覆盖:

┌──────────────────┐              ┌──────────────────┐
│  Layer 1:        │              │  Layer 1:        │
│  Translator (LLM)│              │  ❌ 完全缺失     │
│  → DSL/JSON      │              │                  │
└────────┬─────────┘              └──────────────────┘
         │
┌────────▼─────────┐              ┌──────────────────┐
│  Layer 2:        │              │  Layer 2:        │
│  Env/Engine      │              │  gavis/core/ ✅  │
│  GameEngine      │              │  但只验证了1种游戏│
└────────┬─────────┘              └──────────────────┘
         │
┌────────▼─────────┐              ┌──────────────────┐
│  Layer 3:        │              │  Layer 3:        │
│  Solver          │              │  分散在三份代码:  │
│  MCTS/CFR/PSRO   │              │  gavis/ ✅  MCTS+CFR│
│  /PPO            │              │  moon_chess ⚠️ PSRO│
└────────┬─────────┘              │  vision_ppo ⚠️ PPO│
         │                        └──────────────────┘
┌────────▼─────────┐              ┌──────────────────┐
│  Layer 4:        │              │  Layer 4:        │
│  Interface (VLM) │              │  vision_ppo ✅  │
│  → 实时对局建议   │              │  但未打通 Solver  │
└──────────────────┘              └──────────────────┘
```

### 1.2 合并六个目标 (按四层视角重排)

| # | 目标 | 对应层 | 优先级 |
|---|------|--------|--------|
| 1 | Layer 2→3 接口统一：所有 Solver 消费同一个 Engine | L2↔L3 | P0 |
| 2 | Layer 4→3 打通：识别结果送入 Solver 做决策 | L3↔L4 | P0 |
| 3 | 保留算法多样性：MCTS/CFR/PSRO/PPO 并存 | L3 | P0 |
| 4 | 修复 PSRO 缺失依赖，让所有求解器可运行 | L3 | P0 |
| 5 | 补充月亮棋 rules.json 验证 Engine 的通用性 | L2 | P1 |
| 6 | 架构预留 Layer 1 (Translator) 和在线学习接口 | L1 / 闭环 | P2 |

### 1.3 约束条件

- 团队三人不同分工，合并不能阻塞各自独立开发
- 成员 A 做 PSRO（Layer 3），成员 B 做视觉管线（Layer 4），你维护引擎（Layer 2）
- Layer 1 (Translator) 尚未开发，合并架构需为其预留接口
- 禁止引入大型新依赖

---

## 2. 四层架构总览

合并后的目标架构：

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: Translator                    [预留/未来扩展]          │
│  LLM (规则描述 → rules.json)                                    │
│  接口: TranslatorProtocol                                         │
│     def translate(rule_text: str) -> RulesJSON                    │
│     def validate(rules: RulesJSON) -> bool                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ rules.json
┌──────────────────────────▼──────────────────────────────────────┐
│  Layer 2: Env/Engine                    [gavis/games/ + core/]  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  GameEngine (SolverAdapter 实现)                         │   │
│  │  Games: stochastic_gomoku/  moon_chess/  (未来: 围棋/…)  │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ SolverAdapter Protocol
┌──────────────────────────▼──────────────────────────────────────┐
│  Layer 3: Solver                          [gavis/solvers/]     │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐            │
│  │ MCTS │ │ CFR  │ │ PPO  │ │ PSRO │ │ Auto-    │            │
│  │      │ │      │ │      │ │      │ │ Selector │            │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────────┘            │
│  统一接口: Solver.select_action(state) → action                │
│  统一训练: Solver.train(episodes, callback) → metrics          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ action / state
┌──────────────────────────▼──────────────────────────────────────┐
│  Layer 4: Interface                        [gavis/binding/]    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐  │
│  │ ImageBinding│ │VLM Binding  │ │ 前端 (Web/Mobile/CLI)   │  │
│  │ (快速识别)   │ │(精确识别)   │ │ 实时对局建议 + 反馈    │  │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘  │
│  StateTracker: 跨帧追踪 + 用户反馈收集                         │
│  OnlineLearner: 收集体验数据 → 异步送回 Solver                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 架构方案 A: 四层水平集成 (Horizontal Layers)

### 3.1 核心理念

**将四层严格分离为四个物理目录，层与层之间只通过 Protocol 通信。** 每层可以独立开发、独立测试、独立替换。

### 3.2 目录结构

```
gavis/
├── __init__.py
│
├── layer1_translator/              # (全新编写)
│   ├── __init__.py
│   ├── protocol.py                 # Translator Protocol
│   ├── llm_translator.py           # LLM 实现
│   ├── template_translator.py      # 模板填充实现 (无 LLM 时可用)
│   └── validators/                 # rules.json 校验器
│       ├── schema_validator.py
│       └── game_simulator.py       # 用 Engine 模拟验证规则完整性
│
├── layer2_engine/                  # (从 gavis/core/ 迁移)
│   ├── core/
│   │   ├── engine.py               # GameEngine
│   │   ├── state_graph.py
│   │   └── expr_eval.py
│   ├── games/
│   │   ├── stochastic_gomoku/
│   │   └── moon_chess/             # (新增) 月亮棋 rules.json
│   └── interfaces/
│       └── solver_adapter.py       # SolverAdapter Protocol
│
├── layer3_solvers/                 # (三个来源合并)
│   ├── base.py                     # SolverBase (或 Protocol)
│   ├── mcts/
│   │   └── solver.py               # (来自 gavis/solvers/mcts.py)
│   ├── cfr/
│   │   └── solver.py               # (来自 gavis/solvers/cfr.py)
│   ├── ppo/
│   │   ├── agent.py                # (来自 未命名文件夹/algorithms/)
│   │   ├── networks.py
│   │   └── rollout_buffer.py
│   ├── psro/
│   │   ├── solver.py               # (来自 moon_chess_ai/PSRO/)
│   │   ├── nash_solver.py
│   │   ├── meta_game.py
│   │   └── gym_adapter.py          # (新增) SolverAdapter → Gym
│   └── auto_selector/              # (新增) 自动选择求解器
│       ├── rules_analyzer.py
│       └── solver_picker.py
│
├── layer4_interface/               # (从 未命名文件夹/ 迁移)
│   ├── binding/
│   │   ├── image_binding.py
│   │   ├── vision_binding.py
│   │   ├── mock_binding.py
│   │   ├── qwen_vision.py
│   │   ├── state_tracker.py
│   │   └── schemas.py
│   ├── encoding/
│   │   ├── game_state_adapter.py
│   │   └── moon_state_encoder.py
│   ├── frontend/
│   │   ├── app_server.py
│   │   └── moon_chess_frontend.html
│   └── online_learning/            # (新增) 在线学习入口
│       ├── feedback_collector.py
│       └── async_trainer.py
│
├── tests/
│   ├── test_layer2_engine/
│   ├── test_layer3_solvers/
│   └── test_layer4_interface/
│
└── demos/
    ├── full_pipeline_demo.py       # (新增) 端到端: 规则→训练→UI
    └── benchmark_all.py
```

### 3.3 层间接口设计

```python
# layer2_engine/interfaces/solver_adapter.py
class SolverAdapter(Protocol):
    """Layer 2 → Layer 3 的契约。所有求解器只能通过此接口访问游戏。"""
    def create_initial_state(self) -> State: ...
    def get_node_type(self, state: State) -> NodeType: ...
    def get_current_player(self, state: State) -> str | None: ...
    def get_legal_actions(self, state: State) -> list[ActionInstance]: ...
    def apply_action(self, state: State, action: ActionInstance) -> State: ...
    def get_chance_outcomes(self, state: State) -> list[ChanceOutcome]: ...
    def apply_chance(self, state: State, outcome: ChanceOutcome) -> State: ...
    def is_terminal(self, state: State) -> bool: ...
    def get_utility(self, state: State, player: str) -> float: ...
    def get_observation(self, state: State, player: str) -> Obs: ...
    def get_info_set_key(self, state: State, player: str) -> str: ...


# layer3_solvers/base.py
class SolverBase(ABC):
    """Layer 3 求解器的统一基类。"""
    @abstractmethod
    def select_action(self, state: State) -> ActionInstance: ...
    
    @abstractmethod
    def train(self, episodes: int, **kwargs) -> dict: ...
    
    @abstractmethod
    def save(self, path: str) -> None: ...
    
    @abstractmethod
    def load(self, path: str) -> None: ...


# layer4_interface/binding/schemas.py
class GameObservation:
    """Layer 4 → Layer 3 的契约。VLM 识别结果。"""
    board: list[list[str | None]]
    current_player: str
    legal_actions: list[str]
    piece_order: dict | None        # 月球棋需要
    metadata: dict                  # 置信度、帧号等


# layer4_interface/online_learning/feedback_collector.py
class LearningSignal:
    """Layer 4 → Layer 3 的在线学习信号。"""
    game_state_sequence: list[State]  # 整局的 state 序列
    actions_taken: list[ActionInstance]  # 实际执行的动作
    final_outcome: float               # +1(赢) / 0(平) / -1(输)
    solver_suggestions: list[ActionInstance | None]  # Solver 给出的建议
```

### 3.5 方案优缺点

| 优点 | 缺点 |
|------|------|
| ✅ 架构最干净，层职责明确 | ❌ 初始工作量最大 (需要拆 4 个目录) |
| ✅ Layer 1 独立，未来可添加而不影响下层 | ❌ 新成员需要理解四层架构 |
| ✅ 每层可独立测试、替换、扩展 | ❌ 层间 interface 设计需要审慎协商 |
| ✅ 最符合项目长远目标 | ❌ 短期看不到"合并后能跑"的成果 |

---

## 4. 架构方案 B: 桥接模式 (Bridge Pattern)

### 4.1 核心理念

**保留当前三个项目的物理独立性，通过桥接接口让它们可以联动。** 不在文件系统上强制合并，而是在接口层面打通 Layer 2 / Layer 3 / Layer 4。

### 4.2 目录结构

```
gavis/                             ← (基本不变，你继续维护引擎)
├── core/
├── games/
├── solvers/
└── ...

moon_chess_ai/                     ← (成员 A 继续独立维护)
├── PSRO/
│   ├── train.py                   (修复缺失依赖)
│   └── bridge_adapter.py          ← (新增) 桥接到 SolverAdapter
└── ...

vision_ppo/                        ← (成员 B 继续独立维护)
├── algorithms/
├── binding/
├── encoding/
├── bridge_adapter.py              ← (新增) 桥接到 SolverAdapter
└── ...

bridges/                           ← (新增) 桥接层
├── __init__.py
├── solver_bridge.py               # 统一调用所有求解器的入口
├── engine_bridge.py               # MockMoonEnv → SolverAdapter
├── vision_bridge.py               # Observation → GameEngine input
├── gym_adapter.py                 # SolverAdapter → Gym.Env
└── schemas.py                     # 跨项目共享的数据结构

demos/
├── benchmark_all.py               # 通过 bridge 调所有求解器
└── unified_frontend/              # 统一 Web 前端
    ├── app.py
    └── templates/
```

### 4.3 桥接模式工作方式

```python
# bridges/solver_bridge.py — 不移动代码，只做路由
class SolverBridge:
    """
    不关心求解器代码在哪里，只关心它实现了 SolverAdapter 接口。
    每个现有项目写一个薄适配器，注册到 Bridge 即可。
    """
    
    _solvers: dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str, factory: Callable):
        cls._solvers[name] = factory
    
    @classmethod
    def create(cls, name: str, engine, **kwargs):
        factory = cls._solvers[name]
        return factory(engine, **kwargs)
    
    @classmethod  
    def benchmark(cls, game_engine, episodes=100) -> dict:
        """一键横评所有已注册的求解器。"""
        results = {}
        for name, factory in cls._solvers.items():
            solver = factory(game_engine)
            results[name] = solver.train(episodes)
        return results

# 注册 (在各自的 main.py 或统一的注册脚本中)
SolverBridge.register('mcts', lambda e, **kw: MCTS(e, **kw))
SolverBridge.register('cfr', lambda e, **kw: CFR(e, **kw))
SolverBridge.register('ppo', lambda e, **kw: PPOAgent(e, **kw))
SolverBridge.register('psro', lambda e, **kw: PSROSolver(e, **kw))
```

```python
# bridges/vision_bridge.py — Interface → Engine
class VisionBridge:
    """
    Layer 4 识别出的 Observation → Layer 2 的 state。
    这个桥接是关键——它让"截图"可以直接作为 Engine 的初始状态。
    """
    
    @staticmethod
    def observation_to_state(
        engine: GameEngine,
        obs: Observation,
    ) -> dict:
        """把 VLM 识别结果加载到 Engine 的状态中。"""
        state = engine.create_initial_state()
        board = obs.boardObservation
        for row in range(len(board)):
            for col in range(len(board[row])):
                cell = board[row][col]
                if cell is not None:
                    player = 'p_black' if cell in ('X', '●') else 'p_white'
                    # 直接在 state 中设置棋子
        return state
    
    @staticmethod
    def integrate_with_solver(
        engine: GameEngine,
        solver: SolverBase,
        obs: Observation,
    ) -> ActionInstance:
        """端到端: 截图 → 决策 → 返回建议动作。"""
        state = VisionBridge.observation_to_state(engine, obs)
        return solver.select_action(state)
```

### 4.4 方案优缺点

| 优点 | 缺点 |
|------|------|
| ✅ 零侵入，各项目完全不动 | ❌ 三个独立 repo 维护三套环境 |
| ✅ 最短时间让所有求解器一起跑 | ❌ 桥接层可能越来越厚 |
| ✅ 团队成员零学习成本 | ❌ 没有消除重复代码 (三套状态表示) |
| ✅ 适合过渡期 (3-6 个月) | ❌ 长期维护桥接层的心智负担大 |

---

## 5. 架构方案 C: 微服务式 (Microservice)

### 5.1 核心理念

**每层独立为一个服务/进程**，通过 REST / IPC 通信。Layer 1/2/3/4 各是一个独立的 Python 进程或容器。

### 5.2 架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Translator  │     │  Engine      │     │  Solver      │     │  Interface  │
│  Service     │────▶│  Service     │────▶│  Service     │◀───▶│  Service    │
│  (Layer 1)   │     │  (Layer 2)   │     │  (Layer 3)   │     │  (Layer 4)  │
│              │     │              │     │              │     │             │
│  ports:      │     │  ports:      │     │  ports:      │     │  ports:     │
│  10001       │     │  10002       │     │  10003       │     │  10004      │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │                                        │
                           ▼                                        ▼
                    ┌──────────────────────────────────────────────┐
                    │              Message Queue (Redis/nats)       │
                    │        在线学习信号异步传递                     │
                    └──────────────────────────────────────────────┘
```

### 5.3 通信协议

```python
# 跨层通信全部通过 pydantic 模型定义

# Layer 1 → Layer 2
class TranslateRequest:
    rule_text: str
class TranslateResponse:
    rules_json: dict
    confidence: float

# Layer 2 → Layer 3
class SolverRequest:
    game: str
    state: dict
    solver_type: str
class SolverResponse:
    action: dict
    stats: dict

# Layer 4 → Layer 2 (via VisionBridge)
class RecognitionRequest:
    image_base64: str
    game: str
class RecognitionResponse:
    state: dict
    confidence: float

# Layer 4 → Layer 3 (online learning feedback)
class LearningFeedback:
    session_id: str
    state_sequence: list[dict]
    final_result: float
```

### 5.4 方案优缺点

| 优点 | 缺点 |
|------|------|
| ✅ 四层完全解耦，独立部署 | ❌ 严重过度设计 (对于 3 人团队) |
| ✅ 可以独立扩缩容 (GPU 给 Solver) | ❌ 引入网络延迟 |
| ✅ Layer 1 (LLM) 天然适合独立服务 | ❌ 需要管理容器 / 进程生命周期 |
| ✅ 在线学习回路天然支持异步 | ❌ 调试困难，需要端到端追踪 |

---

## 6. 推荐方案与理由

### 6.1 方案对比

| 维度 | 权重 | A: 四层水平集成 | B: 桥接模式 | C: 微服务 |
|------|:----:|:---:|:---:|:---:|
| 架构清晰度 | 15% | ★★★★★ | ★★★☆☆ | ★★★★★ |
| 消除重复代码 | 20% | ★★★★★ | ★★☆☆☆ | ★★☆☆☆ |
| 渐进迁移可行性 | 15% | ★★★☆☆ | ★★★★★ | ★★☆☆☆ |
| 保留算法多样性 | 10% | ★★★★★ | ★★★★★ | ★★★★★ |
| 团队独立开发 | 15% | ★★★★☆ | ★★★★★ | ★★★★★ |
| 适配长远愿景 | 15% | ★★★★★ | ★★★☆☆ | ★★★★★ |
| 实施工作量 | 10% | ★★★☆☆ (4周) | ★★★★★ (1.5周) | ★☆☆☆☆ (8周+) |
| **总分** | 100% | **4.35/5** | **4.00/5** | **3.55/5** |

### 6.2 推荐

```
推荐方案: 方案 A (四层水平集成)
理由:
  - 最符合"自适应策略游戏 AI Agent"的愿景
  - 消除重复代码，为 Layer 1 (Translator) 预留接口
  - 四层独立，后续团队扩展不会互相阻塞

实施策略:
  - 先做 Layer 2→3 接口统一 (SolverAdapter) [2周]
  - 再做 Layer 4 整合 (Binding + Solver 打通) [1周]  
  - 最后目录重组 + 文档 [1周]
  - Layer 1 (Translator) 作为独立阶段后续实施
```

---

## 7. 详细迁移计划

### 7.1 Phase 1: Layer 2→3 接口统一 (Day 1-10)

```
目标: SolverAdapter Protocol + 所有求解器适配

Day 1-2: 定义 SolverAdapter
  输出: gavis/layer2_engine/interfaces/solver_adapter.py
  - TypedState / TypedAction / SolverAdapter Protocol
  - GameEngine 实现 Protocol (从现有 engine.py 适配)

Day 3-5: MCTS + CFR 适配到 SolverBase
  输入: gavis/solvers/mcts.py + cfr.py
  输出: gavis/layer3_solvers/mcts/solver.py
        gavis/layer3_solvers/cfr/solver.py
  - 包装为统一的 SolverBase.select_action() / train()
  - 提取公共 rollout 函数

Day 6-8: PPO 适配到 SolverAdapter
  输入: 未命名文件夹/algorithms/
  输出: gavis/layer3_solvers/ppo/
  - PPOAgent 的依赖从 MockMoonEnv 改为 SolverAdapter
  - 确认 MoonStateEncoder 仍然可用

Day 9-10: PSRO 适配 + 修复依赖
  输入: moon_chess_ai/PSRO/
  输出: gavis/layer3_solvers/psro/
  - 补全缺失的 Agent 类 (从 tabular_Q 代码提取)
  - GymAdapter: SolverAdapter → Gym 接口
  - 验证 PSRO 可运行

验证: all_solvers_can_play_gomoku.py
  - 4 个求解器都通过 SolverAdapter 下同一盘棋
```

### 7.2 Phase 2: Layer 4 整合 + 打通 Solver (Day 11-15)

```
目标: Interface 层识别结果 → Solver 决策

Day 11-12: Binding 迁入
  输入: 未命名文件夹/binding/
  输出: gavis/layer4_interface/binding/
  - 保持原结构不变
  - 删除 train_mnist_classifier.py (无关文件)
  - 整理 pyproject.toml 依赖

Day 13: VisionBridge
  输出: gavis/layer4_interface/vision_bridge.py
  - Observation → SolverAdapter state
  - 端到端: 截图 → 决策 → 返回动作建议

Day 14: Web 前端对接
  输入: app_server.py + moon_chess_frontend.html
  输出: gavis/layer4_interface/frontend/
  - 前端增加"AI 建议"按钮
  - 调用 Solver 返回建议并高亮显示

Day 15: 新增月亮棋 rules.json
  输出: gavis/layer2_engine/games/moon_chess/rules.json
  - 验证 Engine 能跑月亮棋
  - 验证所有 Solver 能下月亮棋 (通过 SolverAdapter)
```

### 7.3 Phase 3: 目录重组 + 清理 (Day 16-20)

```
目标: 完成文件迁移，旧目录归档

Day 16-17: 文件迁移
  - 按方案 A 的目录结构移动所有文件
  - 更新所有 import 路径
  - 写 __init__.py 导出符号

Day 18: 旧代码归档
  输出: archive/
  - 原始三个项目作为只读存档
  - README 注明"合并前的原始代码"

Day 19-20: 基准评测 + 文档
  输出: demos/benchmark_all.py
  - 一键运行四种求解器对比
  - 更新 README.md 描述四层架构
```

---

## 8. 特殊挑战: Online Learning 闭环

### 8.1 架构上的预留

在线自学习闭环是项目的终极形态，当前阶段不需要实现全部，但架构需要预留接口：

```python
# gavis/layer4_interface/online_learning/
class OnlineLearningSignal:
    """
    Layer 4 收集的"真实对局经验"，可以送入 Layer 3 的求解器。
    当前只是预留——具体实现在后续版本。
    """
    game_state_sequence: list[State]
    actions_taken: list[ActionInstance]
    final_outcome: Literal[-1, 0, 1]
    
    # 元数据: 用户信息、对局时长、Solver 建议

class OnlineLearner:
    """负责异步将经验送回 Solver。"""
    
    def collect(self, signal: OnlineLearningSignal) -> None:
        """收集一次对局经验 (同步，快)。"""
        # 当前: 写入队列或文件
        # 未来: 异步送入 Solver.update()
    
    def batch_update(self, solver: SolverBase) -> None:
        """批量将积累的经验喂给求解器 (异步，慢)。"""
        # PPO: 添加到 replay buffer, 跑几个 epoch update
        # CFR: 追加迭代
        # PSRO: 评估新策略是否加入池
```

### 8.2 当前阶段与终极形态的映射

| 阶段 | Layer 1 | Layer 2 | Layer 3 | Layer 4 | 闭环 |
|:----:|:-------:|:-------:|:-------:|:-------:|:----:|
| **现在** (4个孤立的代码库) | ❌ | ⚠️ | ⚠️ | ⚠️ | ❌ |
| **合并后** (Phase 1-3) | 预留接口 | ✅ 统一 | ✅ 统一 | ✅ 打通 | 预留接口 |
| **v1.0** (下个里程碑) | 基础实现 | ✅ | ✅ + 横评 | ✅ + 反馈收集 | 异步训练 |
| **终极** | LLM 翻译 | 多游戏 | Auto-Selector | 全平台 | 实时闭环 |
