# Gavis 项目架构对比文档

> 文档日期: 2026-07-28
> 对比范围: 四个代码库的架构、设计哲学、实现质量横向比较

---

## 目录

1. [总体对比矩阵](#1-总体对比矩阵)
2. [设计哲学对比](#2-设计哲学对比)
3. [接口设计对比](#3-接口设计对比)
4. [状态表示对比](#4-状态表示对比)
5. [动作表示对比](#5-动作表示对比)
6. [策略表示对比](#6-策略表示对比)
7. [数据流对比](#7-数据流对比)
8. [错误处理对比](#8-错误处理对比)
9. [扩展性对比](#9-扩展性对比)
10. [代码规模统计](#10-代码规模统计)

---

## 1. 总体对比矩阵

| 维度 | A: Gavis Engine | B: Moon PSRO | C: Vision+PPO |
|------|:---:|:---:|:---:|
| **主语言** | Python 3.11+ | Python 3.x | Python 3.11+ |
| **框架依赖** | 无 (纯标准库) | scipy, numpy, gymnasium | torch, numpy, opencv-python, pydantic |
| **设计范式** | 声明式规则引擎 | 过程式训练循环 | 面向对象 + Protocol |
| **总代码行数** | ~950 | ~400 | ~3500 |
| **测试覆盖率** | 0% | 0% | ~40% |
| **文档覆盖** | 注释充分, 无独立文档 | README 有, 但简短 | README 详细 |
| **可运行性** | ✅ 可直接运行 | ⚠️ 缺依赖文件 | ✅ 可运行 (mock 模式) |
| **Python 类型提示** | 部分 | 无 | 完整 (from __future__) |
| **异常层次** | ValueError / 无自定义 | 无自定义 | 7 种自定义异常 |
| **依赖数量** | 0 | 4+ | 5+ |

## 2. 设计哲学对比

### 2.1 核心原则

| | A: Gavis Engine | B: Moon PSRO | C: Vision+PPO |
|--|---|---|---|
| **核心理念** | 声明式规则 + 通用引擎 | 元博弈均衡计算 | 完整 AI 管线: 视觉→编码→决策 |
| **设计驱动力** | 求解器与游戏解耦 | 算法实验 | 端到端可演示 |
| **抽象层次** | 高 (规则引擎 + 表达式求值) | 低 (过程式脚本) | 中 (类 + 接口分离) |
| **扩展方向** | 多游戏多求解器 | 更强的元博弈算法 | 更好的视觉识别 + 更深的策略网络 |

### 2.2 优缺点一句话

```
A: "可以跑任何声明式博弈"        — 但只实装了一种游戏
B: "理论上可以收敛到纳什均衡"    — 但代码跑不起来  
C: "从截图到落子一条龙"          — 但训练严重不充分
```

## 3. 接口设计对比

### 3.1 游戏环境接口

| 特性 | A: GameEngine | B: Gym Wrapper | C: MockMoonEnv |
|------|:---:|:---:|:---:|
| 创建状态 | `create_initial_state()` | `reset()` | `reset()` |
| 获取动作 | `get_legal_actions()` | 隐含在观测量 | `get_state()['legalActions']` |
| 执行动作 | `apply_action(state, action)` | `step(action)` | `step(action_dict)` |
| 判断终局 | `is_terminal(state)` | `done` 返回值 | `is_terminal()` |
| 获取收益 | `get_utility(state, player)` | `reward` 返回值 | 未独立提供 |
| 随机节点 | ✅ 一等公民 (chance) | ❌ | ❌ |
| 完美信息 | ✅ 完整状态 | ✅ 完整 3^9 编码 | ✅ board + pieceOrder |

**结论：** B 和 C 使用了 Gym 风格接口 (或类似风格)，A 使用了一个更宽松的方法集。三者**完全不兼容**。

### 3.2 求解器接口

| 特性 | A: MCTS | A: CFR | B: PSRO | C: PPOAgent |
|------|:---:|:---:|:---:|:---:|
| 选择动作 | `search(state)` | `get_action(state)` | `Agent.step(obs)` | `select_action(state, mask)` |
| 训练 | 无 (纯搜索) | `solve(state)` | `PSRO_Q(env)` | `update()` |
| 保存/加载 | ❌ | ❌ | `np.save` Q 表 | `save()` / `load()` PyTorch |
| 批量推理 | ❌ | ❌ | ❌ | ✅ (通过 buffer) |
| 动作掩码 | 隐式 (legal_actions) | 隐式 (legal_actions) | 隐式 (available_actions) | ✅ 显式 action_mask |

## 4. 状态表示对比

### 4.1 表示方式

```
A: dict {
    'board_size': 9,
    '_board': [None, None, ..., 'black', ...],   # 81 elements
    'nodes': {'cell_0_0': {...}, ...},            # 81 nodes
    'env': { 'phase': 'playing', 'turn': {...}, ... }
}

B: int (3-base encoding)
   每个格子: 0=空, 1=黑, 2=白
   编码: sum(digit_i * 3^i)  ∈ [0, 19682]

C: dict {
    'board': [[None, 'X', None], ...],            # 3×3 list
    'pieceOrder': {'player_x': [{cellId, placedSeq}, ...]},
    'currentPlayerId': 'player_x',
    'legalActions': ['cell_0_0', ...],
    'stepCount': 5,
    'status': 'running',
    'playerSymbols': {'X': 'player_x', 'O': 'player_o'}
}
```

### 4.2 表示能力对比

| 能力 | A | B | C |
|------|:---:|:---:|:---:|
| 编码时间信息 | ✅ env.turn.round | ❌ | ✅ pieceOrder |
| 编码玩家身份 | ✅ p_black/p_white | ❌ (3-base 不分颜色) | ✅ player_x/player_o |
| 支持任意棋盘 | ✅ board_size 参数化 | ❌ 固定 3×3 | ❌ 固定 3×3 |
| 深拷贝效率 | ✅ 浅 copy | ✅ int 拷贝 | ❌ list of list 拷贝 |
| 人类可读 | ✅ 部分 (nodes 字段多) | ❌ 纯整数 | ✅ |

### 4.3 特征向量 (给 RL 的输入)

```
A: 无 (MCTS/CFR 直接用 state dict)

B: int (3-base code, 相当于 1 维离散观测)

C: np.ndarray[38] — MoonStateEncoder:
    [0:27)   9 cells × 3 one-hot (empty/self/opp)
    [27:36)  9 cells × age (1=latest…3=oldest)
    [36]     whose turn (0/1)
    [37]     normalized step count
```

## 5. 动作表示对比

```
A: ActionInstance(template_id='place_stone', actor_id='p_black',
                  params={'cell': {id:'cell_3_5', props:{...}}},
                  canonical_key='place:3,5')

B: int ∈ [0, 8]   (单元格索引)

C: dict {
    'actorId': 'player_x',
    'actionType': 'place_piece',
    'parameters': {'targetCellId': 'cell_1_2'}
}
```

| 维度 | A | B | C |
|------|:---:|:---:|:---:|
| 动作空间大小 | 棋盘空格数 (可变的) | 固定 9 | 固定 9 |
| 携带上下文 | ✅ 完整 node 数据 | ❌ 纯索引 | ❌ cellId 字符串 |
| 类型安全 | ✅ Enum 风格的 template_id | ❌ 纯 int | ✅ actionType 字符串 |
| 跨游戏通用 | ✅ | ❌ | ❌ |

## 6. 策略表示对比

```
A-MCTS:  树搜索策略 (无存储)，隐式策略: π(s) = argmax UCB1(s,a)

A-CFR:   info_sets dict: 
         {'info_set_key': {'regrets': {action_key: float}, 
                           'strategy_sum': {action_key: float}}}

B-PSRO:  策略池: pi[N][19683][9] → one-hot
         纳什混合: nash_pi = Σ(w_i * pi_i)

C-PPO:   神经网络参数: ActorCriticNetwork
         actor: Linear(128→9) + Softmax
         critic: Linear(128→1)
```

| 维度 | A-MCTS | A-CFR | B-PSRO | C-PPO |
|------|:---:|:---:|:---:|:---:|
| 存储大小 | 0 (运行时树) | O(state_visits) | O(N×19683×9) | O(128×38 + 128×9) |
| 泛化到未见状态 | ❌ 需重新搜索 | ✅ 策略会泛化 | ❌ 查表 | ✅ 网络前向 |
| 支持更新 | ❌ 无学习 | ✅ 增量 regret | ✅ PSRO 迭代 | ✅ 梯度下降 |
| 支持热加载 | ❌ | ❌ | ✅ np.load | ✅ torch.load |
| 确定性 | ❌ UCB 含随机 | ❌ 采样 | ❌ 纳什混合 | ✅ (无采样时) |

## 7. 数据流对比

### 7.1 A: MCTS 数据流

```
用户请求 → demo.py
  → GameEngine.create_initial_state() → state dict
  → MCTS.search(state)
    → 循环 budget 次:
      → GameEngine.get_legal_actions(state)
      → GameEngine.apply_action(state, action)
      → GameEngine.get_chance_outcomes(state)  
      → GameEngine.apply_chance(state, outcome)
      → GameEngine.is_terminal(state) → GameEngine.get_utility(state, player)
  → best ActionInstance
  → GameEngine.apply_action(state, best)
  → 渲染 Board → 终端输出
```

### 7.2 B: PSRO 数据流

```
用户请求 → PSRO/train.py
  → MoonEnvWrapper (Gym)
  → PSRO_Q(env):
    1. 初始化随机策略 pi[0]
    2. gamescape() → 两两对战
       → Agent.step(obs) → env.step(action) → reward
    3. solve_nash(R) → nash distribution
    4. tabular_Q() → new policy
    5. 策略已收敛? → 结束 / 继续
  → np.save('Qh.npy', data)
```

### 7.3 C: PPO + Vision 数据流

```
视觉路径:
  浏览器截图 → base64 → HTTP POST /api/recognize
    → VisionLLMBinding.parse_bytes()
      → QwenVisionClient.infer() → LLM 识别
      → StateTracker.update() → 跨帧追踪
    → Observation (pydantic) → JSON response

训练路径:
  train_ppo.py
    → MockMoonEnv.reset()
    → MoonStateEncoder.encode() → 38-dim vector
    → PPOAgent.select_action(vector, mask) → action index
    → agent.build_action(player, index) → action dict
    → MockMoonEnv.step(action dict) → (obs, reward, done)
    → PPOAgent.record_transition(...) → buffer
    → 每 episode 结束: PPOAgent.update() → policy gradient
```

## 8. 错误处理对比

| 维度 | A: Gavis Engine | B: Moon PSRO | C: Vision+PPO |
|------|:---:|:---:|:---:|
| 自定义异常 | ❌ | ❌ | ✅ 7 种异常 |
| 合法动作校验 | ✅ 引擎层面 | ✅ mask 机制 | ✅ PPO 内 check |
| 参数校验 | ❌ 异常来自 dict 访问 | ❌ | ✅ pydantic ValidationError |
| 边界处理 | ❌ 假设状态合法 | ❌ 假设代码正确 | ✅ 多种 if-not 守卫 |
| 资源清理 | ❌ 无状态泄漏风险 | ❌ 无外部资源 | ❌ HTTP 服务有 (通过 try/finally) |

## 9. 扩展性对比

### 9.1 添加新游戏

| 步骤 | A | B | C |
|------|:---:|:---:|:---:|
| 1 | 写 rules.json | 实现 Gym env | 实现对应环境 |
| 2 | (不需要) | 实现 wrapper | 实现特征编码 |
| 3 | 求解器不变 | 修改 PSRO 中的动作映射 | 绑定到新环境 |
| 难度 | **低** | **高** | **中** |

### 9.2 添加新求解器

| 步骤 | A | B | C |
|------|:---:|:---:|:---:|
| 1 | 实现 GameEngine 接口的方法 | 实现 Agent 接口 | 实现 solver 接口 |
| 2 | 直接使用 | 调整 to/from 框架 | 接入编码器 |
| 难度 | **低** | **中** | **中** |

### 9.3 扩展到更大棋盘

| 项目 | 当前最大 | 瓶颈 | 扩展可能性 |
|------|:--------|:----|:----------|
| A-MCTS | 9×9 | budget 不够 | ✅ 增加 budget + GPU 并行 |
| A-CFR | 5×5 | info_set 爆炸 | ❌ 指数级 |
| B-PSRO | 3×3 | 表格 3^N | ❌ 必须换神经网络 |
| C-PPO | 3×3 | 编码器固定 3×3 | ⚠️ 需要重构编码器 |

## 10. 代码规模统计

### 10.1 行数

```
Project A: Gavis Engine
  core/engine.py         505 行
  core/state_graph.py     56 行
  core/expr_eval.py      257 行
  solvers/mcts.py        309 行
  solvers/cfr.py         317 行
  demo.py                297 行
  demo_cfr.py            168 行
  总计:                 ~1,909 行

Project B: Moon PSRO
  train.py (root)        22 行  (不可用)
  PSRO/train.py          166 行
  PSRO/moon_env_wrapper  54 行
  PSRO/watch_psro.py     33 行
  总计:                 ~275 行 (可用 ~253 行)

Project C: Vision+PPO
  algorithms/             390 行
  binding/               1,867 行  (含 qwen_vision.py)
  encoding/               95 行
  training/               275 行
  tests/                  ~600 行
  app_server.py          125 行
  总计:                 ~3,352 行
```

### 10.2 依赖树复杂度

```
A:  无外部依赖
    ├── json           (标准库)
    ├── random         (标准库)  
    └── dataclasses    (标准库)

B:  外部依赖 4+
    ├── numpy
    ├── scipy
    ├── gymnasium
    ├── tqdm
    └── stable-baselines3 (root train.py)

C:  外部依赖 5+
    ├── torch
    ├── numpy
    ├── opencv-python
    ├── pydantic
    └── httpx / requests (qwen_vision)
```
