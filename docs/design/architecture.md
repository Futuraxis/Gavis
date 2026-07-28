# Gavis — 自适应策略游戏 AI Agent 架构设计

> 版本: v0.2 | 状态: 实现中 (四层水平集成) | 日期: 2026-07-28

---

## 1. 项目定位

**任何一个策略类游戏**——无论是桌游、电子游戏、还是直播中的对局——只要你能**描述规则**或**给它截图**，Gavis 就能学会下它，并且能**实时给出对局建议**。

---

## 2. 四层架构总览

```
┌────────────────────────────────────────────────────────────┐
│ Layer 1: Translator        [LLM 规则翻译]                  │
│  输入: 自然语言规则 → 输出: rules.json (v4.1)             │
│  状态: 预留 (Protocol + SchemaValidator 已定义)            │
├────────────────────────────────────────────────────────────┤
│ Layer 2: Env/Engine        [游戏引擎]                      │
│  rules.json → GameEngine → SolverAdapter                  │
│  求解器通过 SolverAdapter Protocol 消费游戏                │
├────────────────────────────────────────────────────────────┤
│ Layer 3: Solver            [求解器]                        │
│  MCTS / CFR / PPO / PSRO → 统一 SolverBase 接口           │
│  + AutoSelector (预留)                                     │
├────────────────────────────────────────────────────────────┤
│ Layer 4: Interface         [交互界面]                      │
│  Binding: ImageBinding / VisionLLMBinding / StateTracker   │
│  Encoding: MoonStateEncoder → 特征向量                     │
│  VisionBridge: Observation → Engine State (不通 Solver)    │
│  OnlineLearning: 反馈收集 (预留)                           │
└────────────────────────────────────────────────────────────┘
```

---

## 3. 当前目录结构

```
rules/                              # 游戏规则 JSON (v4.1)
├── stochastic_gomoku.json           # 随机五子棋 (9×9, 50% 消失)
└── moon_chess.json                  # 月亮棋 (3×3, FIFO 淘汰)

layer1_translator/                  # (预留) LLM 规则翻译
├── protocol.py                      # TranslatorProtocol
└── schema_validator.py              # rules.json 格式校验

layer2_engine/                      # 游戏引擎
├── core/
│   ├── engine.py                    # GameEngine — 实现 SolverAdapter
│   ├── state_graph.py               # State / ActionInstance / ChanceOutcome
│   └── expr_eval.py                 # 表达式求值器
├── interfaces/
│   └── solver_adapter.py            # SolverAdapter Protocol (Layer 2→3 契约)
└── games/
    └── moon_chess/
        └── moon_env_adapter.py      # 月亮棋薄适配器 (仅 RL 辅助方法)

layer3_solvers/                     # 求解器
├── base.py                          # SolverBase 抽象类
├── mcts/solver.py                   # MCTS (带 chance 节点)
├── cfr/solver.py                    # CFR (External Sampling MC-CFR)
├── ppo/                             # PPO (PyTorch, GAE, PPO-clip)
│   ├── solver.py
│   ├── networks.py
│   └── rollout_buffer.py
└── psro/                            # PSRO (表格 Q + 纳什均衡)
    ├── solver.py
    ├── nash_solver.py
    ├── meta_game.py
    ├── tabular_q.py
    ├── agent.py
    └── gym_adapter.py              # SolverAdapter → Gym.Env 桥

layer4_interface/                   # 交互界面
├── binding/                         # 视觉识别管线
│   ├── image_binding.py             # OpenCV 模板匹配
│   ├── vision_binding.py            # VLM 大模型识别
│   ├── qwen_vision.py               # 通义千问 VL 客户端
│   ├── state_tracker.py             # 跨帧追踪
│   └── schemas.py                   # Observation pydantic 模型
├── encoding/                        # 状态编码
│   ├── moon_state_encoder.py        # 38 维特征向量
│   └── game_state_adapter.py
├── frontend/                        # Web 服务
│   └── app_server.py
├── online_learning/                 # (预留) 在线学习
└── vision_bridge.py                 # Observation → Engine State

demos/                              # 演示入口
├── demo_mcts.py                     # MCTS 随机五子棋
├── demo_cfr.py                      # CFR 随机五子棋
└── benchmark_all.py                 # 统一基准评测

tests/                              # 测试 (89 个用例)
├── test_layer2_engine/
├── test_layer3_solvers/
├── test_layer4_interface/
├── test_integration.py
└── test_layer1_translator.py
```

---

## 4. 核心契约

### 4.1 SolverAdapter Protocol (Layer 2 → 3)

```python
class SolverAdapter(Protocol):
    def create_initial_state(self) -> State
    def get_node_type(self, state) -> Literal['player','chance','terminal']
    def get_current_player(self, state) -> str | None
    def get_legal_actions(self, state) -> list[ActionInstance]
    def apply_action(self, state, action) -> State
    def get_chance_outcomes(self, state) -> list[ChanceOutcome]
    def apply_chance(self, state, outcome) -> State
    def is_terminal(self, state) -> bool
    def get_utility(self, state, player) -> float
    def get_observation(self, state, player) -> Obs
    def get_info_set_key(self, state, player) -> str
```

### 4.2 SolverBase (Layer 3)

```python
class SolverBase(ABC):
    def select_action(self, state) -> ActionInstance | None
    def train(self, episodes, **kwargs) -> SolverMetrics
    def save(self, path)
    def load(self, path)
```

### 4.3 VisionBridge (Layer 4 → 2)

```python
def observation_to_state(observation: Observation, engine: SolverAdapter) -> State
```

纯函数，不依赖 Layer 3。Solver 集成在应用层完成。

---

## 5. 数据流

### 5.1 离线训练

```
rules.json → GameEngine → SolverAdapter → SolverBase.train() → 策略
```

### 5.2 在线决策

```
截图 → ImageBinding/VisionLLMBinding → Observation
  → vision_bridge.observation_to_state() → State
  → SolverBase.select_action() → ActionInstance
  → 前端展示建议
```

### 5.3 端到端 (Future)

```
用户文本规则 → Translator → rules.json → Engine → Solver → Interface → 对局建议
```

---

## 6. 求解器对比

| 求解器 | 类型 | 适用场景 | 训练方式 | 当前状态 |
|--------|------|---------|---------|---------|
| MCTS | 搜索 | 完美信息、随机博弈 | 纯搜索 (无需训练) | ✅ 可用 |
| CFR | 离线学习 | 小棋盘、需要均衡解 | External Sampling | ✅ 可用 |
| PPO | 在线学习 | 大状态空间、视觉输入 | 策略梯度 + GAE | ✅ 可用 (需 torch) |
| PSRO | 元博弈 | 策略多样性、开放型博弈 | 策略池 + 纳什均衡 | ✅ 可用 |

---

## 7. 效果路线图

```
现在:     四层结构已建立, 4 种求解器可跑, 89 个测试通过
下阶段:   Layer 1 (Translator) 实现, 在线学习闭环, 更多游戏
```

详见 `docs/merge/` 下的六篇架构分析文档：
1. [架构设计](docs/merge/01_architecture_design.md)
2. [架构优劣分析](docs/merge/02_architecture_pros_cons.md)
3. [架构对比](docs/merge/03_architecture_comparison.md)
4. [代码风格分析](docs/merge/04_code_style_analysis.md)
5. [算法分析](docs/merge/05_algorithm_analysis.md)
6. [合并方案与迁移](docs/merge/06_merge_architecture_and_migration.md)
