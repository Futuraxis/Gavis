# Gavis — 自适应策略游戏 AI Agent 架构设计

> 版本: v0.4 | 状态: 实现中 (四层水平集成) | 日期: 2026-08-22
>
> 规则语言: **v5.2 零 BUILTIN + variants 声明式**（`rules["functions"]` alias
> 规则自足；变种/人数/配比在 JSON `variants` 声明，无 per-game 适配器）。
> 架构分层硬约束：Layer N 只依赖 Layer N-1；层间通信只走契约
> （L2→L3: `GameEngine`；L3: `SolverBase`；L4: `BaseBinding`）。

---

## 1. 项目定位

**任何一个策略类游戏**——无论是桌游、电子游戏、还是直播中的对局——只要你能**描述规则**或**给它截图**，Gavis 就能学会下它，并且能**实时给出对局建议**。

---

## 2. 四层架构总览

```
┌────────────────────────────────────────────────────────────┐
│ Layer 1: Translator        [LLM 规则翻译]                  │
│  输入: 自然语言规则 → 输出: rules.json (v5.1)             │
│  状态: 已实现 (确定性模板 + 规则族生成 + LLM 编排 +       │
│         schema 校验, 见 review_layer1_translator)          │
│  层间: L1→L2 单一授权通道 (validate() = schema +           │
│         L2 smoke 校验, 引擎冒烟验证下沉到 L2 服务)         │
├────────────────────────────────────────────────────────────┤
│ Layer 2: Env/Engine        [游戏引擎核心]                  │
│  rules.json → GameEngine（单一契约，无 per-game 适配器）   │
│  求解器通过 GameEngine 公开 API 消费游戏                   │
├────────────────────────────────────────────────────────────┤
│ Layer 3: Solver            [求解器]                        │
│  MCTS / CFR / Hybrid / PPO / PSRO / MARL / LLM / 启发式    │
│  → 统一 SolverBase 接口                                    │
├────────────────────────────────────────────────────────────┤
│ Layer 4: Interface         [交互界面]                      │
│  Binding: ImageBinding / VisionLLMBinding / StateTracker   │
│  Encoding: MoonStateEncoder → 特征向量                     │
│  VisionBridge: Observation → Engine State                  │
│  Frontend: play_* 单应用服务 + platform 平台前端           │
│   (React 前端: 大厅/对战/评测/历史/在线学习,               │
│    见 frontend/platform/)                                 │
│  OnlineLearning: 捕获+持久化+经验表+门禁发布 (已实现,      │
│    见 docs/design/online-learning.md)                     │
│  层间: 不 import L3 — 求解器经 SolverProvider 协议注入     │
│         (L4 定义协议, train-cli/games.py 注册表装配实现)   │
└────────────────────────────────────────────────────────────┘
```

---

## 3. 当前目录结构

```
rules/                              # 游戏规则 JSON (v5.2, 零 BUILTIN + variants)
├── stochastic_gomoku.json           # 随机五子棋 (9×9, 50% 消失)
├── moon_chess.json                  # 月亮棋 (3×3, FIFO 淘汰)
├── texas_holdem.json                # 德州扑克 (双人/多人)
├── mahjong.json                     # 麻将 (由 _gen_mahjong.py 生成)
└── werewolf.json                    # 狼人杀 (由 _gen_werewolf.py 生成)

layer1_translator/                  # LLM 规则翻译 (已实现)
├── protocol.py                      # TranslatorProtocol + LLMTranslatorError
├── rule_parser.py                   # 标记化规则解析
├── prompt_builder.py                # LLM 提示构造 (系统提示硬化)
├── llm_translator.py                # LLM 编排 + 修复循环 + 模板兜底
├── natural_language_translator.py   # 确定性模板翻译 (5 游戏)
├── rule_family_builder.py           # 规则族生成
├── schema_validator.py              # rules.json 格式校验
├── datasets.py / local_client.py    # 训练数据与本地 LLM 客户端
└── engine_validator.py              # validate() = schema + L2 smoke 校验
                                    # (L1→L2 单一授权通道; 冒烟服务在 L2)

layer2_engine/                      # 游戏引擎核心 (无 per-game 适配器)
├── core/
│   ├── engine.py                    # GameEngine — 规则解释器 + 编译器（L2→L3 单一契约）
│   ├── state_graph.py               # State / ActionInstance / ChanceOutcome / NodeType
│   ├── expr_eval.py                 # 表达式求值器 + codegen 编译
│   ├── rules_compiler.py            # v5.2 alias 编译管线
│   ├── api_key.py                   # L1 通道的 LLM API key 解析（自 interfaces/ 迁入）
│   └── smoke_validator.py           # L2 冒烟校验服务 (L1 validate 下沉)

layer3_solvers/                     # 求解器
├── base.py                          # SolverBase 抽象类
├── mcts/solver.py                   # MCTS (带 chance 节点)
├── cfr/solver.py                    # CFR (External Sampling MC-CFR)
├── hybrid/solver.py                 # HybridSolver (CFR 表 + MCTS/策略)
├── llm/ollama_solver.py             # Ollama 大模型求解器
├── mahjong/heuristic.py             # MahjongHeuristicAI (启发式)
├── ppo/                             # PPO (PyTorch, GAE, PPO-clip)
├── psro/                            # PSRO (表格 Q + 纳什均衡)
└── marl/                            # 多智能体 (QMix/HAPPO/MAAC, 需 torch)

layer4_interface/                   # 交互界面
├── binding/                         # 视觉识别管线 (BaseBinding Protocol)
│   ├── image_binding.py             # OpenCV 模板匹配
│   ├── dom_binding.py               # DOM 界面读取
│   ├── vision_binding.py            # VLM 大模型识别
│   ├── mock_binding.py              # 测试用
│   ├── qwen_vision.py               # 通义千问 VL 客户端
│   ├── state_tracker.py             # 跨帧追踪
│   └── schemas.py                   # Observation pydantic 模型
├── encoding/                        # 状态编码
│   ├── moon_state_encoder.py        # 38 维特征向量
│   └── game_state_adapter.py
├── solver_provider.py               # SolverHandle / SolverProvider 协议
├── frontend/                        # Web 服务 (按应用分目录)
│   ├── common/http_utils.py         # send_json / read_json_body
│   ├── play_moon_chess/             # 月亮棋人机对弈 (8765)
│   ├── play_gomoku/                 # 随机五子棋人机对弈 (8767)
│   ├── play_texas_holdem/           # 德州扑克人机对弈 (8768)
│   ├── play_werewolf/               # 狼人杀人机对弈 (8771, 需 ollama)
│   ├── vision/                      # 视觉识别应用 (8766)
│   └── platform/                    # 平台前端服务 (8770)
│       ├── server.py                # HTTP 路由 + 静态服务 dist/
│       ├── games.py                 # GameSpec 游戏注册表 (六游戏)
│       ├── session.py               # 通用 GameSession + PlayManager
│       ├── history.py               # 对局记录 (data/matches/*.json)
│       └── benchmark.py             # 求解器评测 (后台 job)
│       # 前端: platform-frontend/ (React+Vite+TS, npm run build → dist/)
├── online_learning/                 # 在线学习 (已实现, 见 §2/§7)
│   ├── recorder.py                  # 逐决策捕获 (RecordingHandle 包装)
│   ├── store.py                     # JSONL 持久化 (data/online_learning/)
│   ├── signals.py                   # 轨迹 → OnlineLearningSignal
│   ├── feedback_collector.py        # 信号数据类 + 内存采集器
│   ├── models.py                    # OnlineModelStore 已发布模型/回滚
│   └── manager.py                   # LearningManager (建表/门禁/发布/auto)
└── vision_bridge.py                 # Observation → Engine State

train-cli/                          # 训练 CLI + 求解器装配（游戏注册制）
├── games.py                        # 游戏注册表 (7 游戏: 引擎/座位/训练管线/运行时求解器)
│                                   #   + DefaultSolverProvider (L3 装配点, 数据驱动)
└── train.py                        # 统一抽象训练脚本 (只读注册表, 无 per-game 分支)
train_cli.py                        # 根目录导入桥 → train-cli/ (连字符目录模块化别名)

tests/                              # 测试 (558 用例, pytest)
├── test_layer1_translator/
├── test_layer2_engine/
├── test_layer3_solvers/
├── test_layer4_interface/
├── test_integration.py
└── test_security_fixes.py
```

---

## 4. 核心契约

### 4.1 GameEngine 契约 (Layer 2 → 3)

```python
engine = GameEngine(rules, seed=None, variant=None, player_count=None)
state = engine.create_initial_state()
engine.get_node_type(state)              # Literal['player','chance','terminal']
engine.get_current_player(state)         # str | None（env.turn 推导）
engine.get_legal_actions(state)          # list[ActionInstance]
engine.apply_action(state, action)       # -> State
engine.sample_chance(state)              # chance 采样（get_chance_outcomes + rng）
engine.apply_chance(state, outcome)      # -> State
engine.is_terminal(state) / engine.get_utility(state, player)
engine.project_observation(state, player)  # -> Obs（visibility 声明式投影）
engine.get_info_set_key(state, player)
engine.eval_expr(expr, extra_ctx=None)   # 规则表达式求值（前端显示助手用）
```

部分可观测、变体/人数/配比、轮转全部由规则 JSON 声明（`visibility` /
`variants` / env.`turn` / `chance`+`effectMap`），引擎不做任何游戏特化。

### 4.2 SolverBase (Layer 3)

```python
class SolverBase(ABC):
    def select_action(self, state) -> ActionInstance | None
    def train(self, episodes, **kwargs) -> SolverMetrics
    def save(self, path)
    def load(self, path)
```

### 4.3 SolverProvider (Layer 4 定义、应用层实现 — L4→L3 依赖倒转)

```python
class SolverProvider(Protocol):
    def create_solver(self, game_id, name, engine, seed, budget, **kwargs) -> SolverHandle

# 实现与装配在 train-cli/games.py（唯一允许 import layer3_solvers 的
# 装配点，全部由注册表数据驱动）；layer4_interface/ 内不存在
# layer3_solvers 的 import。
# 启动注入: play_* / platform 的 main() 将 default_provider 注入
# PlayManager / BenchmarkRunner。
```

### 4.4 VisionBridge (Layer 4 → 2)

```python
def observation_to_state(observation: Observation, engine: GameEngine) -> State
```

纯函数，不依赖 Layer 3。Solver 集成在应用层完成。

### 4.5 Layer 1 → Layer 2（授权依赖）

`layer1_translator/engine_validator.py` 的 `validate()` = schema 校验 +
`layer2_engine/core/smoke_validator.py` 的引擎冒烟校验（启动 GameEngine
并探针 player/chance/terminal 节点）——L1→L2 是单一授权通道，L1 需
`layer2_engine` 内容本质上就是把自然语言翻译成 L2 可执行规则（契约）。

---

## 5. 数据流

### 5.1 离线训练

```
rules.json → GameEngine → SolverBase.train() → 策略
```

### 5.2 在线决策

```
截图 → ImageBinding/VisionLLMBinding → Observation
  → vision_bridge.observation_to_state() → State
  → SolverHandle.select_action() (经 SolverProvider 注入的求解器) → ActionInstance
  → 前端展示建议
```

### 5.3 端到端

```
用户文本规则 → TranslateService(LLM) → engine_validator.validate() → 可执行 rules.json
  → Engine → Solver → Interface → 对局建议
```

---

## 6. 求解器对比

| 求解器 | 适用场景 | 训练方式 | 当前状态 |
|--------|---------|---------|---------|
| MCTS | 完美信息、随机博弈 | 纯搜索 (无需训练) | ✅ 可用 |
| CFR | 小棋盘、需要均衡解 | External Sampling | ✅ 可用 |
| Hybrid | 不完全信息（德州扑克） | CFR 表 + MCTS/策略 + 经验对手模型（在线学习） | ✅ 可用 |
| OllamaSolver | 自由文本游戏（狼人杀） | 本地 LLM 推理 | ✅ 可用 (需 ollama) |
| MahjongHeuristicAI | 麻将 | 启发式 | ✅ 可用 |
| PPO | 大状态空间、视觉输入 | 策略梯度 + GAE | ✅ 可用 (需 torch) |
| PSRO | 策略多样性、开放型博弈 | 策略池 + 纳什均衡 | ✅ 可用 |
| MARL (QMix/HAPPO/MAAC) | 多智能体 | CTDE | ✅ 可用 (需 torch) |

---

## 7. 效果路线图

```
现在:     四层结构已建立, 8 类求解器 + 六游戏可跑, 610 测试通过,
          在线学习 MVP 已上线 (德州扑克经验对手模型 + 门禁发布,
          见 docs/design/online-learning.md)
下阶段:   Layer 1 LLM 路由全量启用, 在线学习扩展到 MCTS/PPO/CFR,
          平台鉴权/工程化
```