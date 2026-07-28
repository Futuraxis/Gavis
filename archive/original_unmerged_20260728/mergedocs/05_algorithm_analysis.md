# Gavis 项目算法分析文档

> 文档日期: 2026-07-28
> 覆盖范围: 项目中使用的四种博弈求解算法

---

## 目录

1. [MCTS (Monte Carlo Tree Search)](#1-mcts-monte-carlo-tree-search)
2. [CFR (Counterfactual Regret Minimization)](#2-cfr-counterfactual-regret-minimization)
3. [PSRO (Policy-Space Response Oracles)](#3-psro-policy-space-response-oracles)
4. [PPO (Proximal Policy Optimization)](#4-ppo-proximal-policy-optimization)
5. [算法适用性对比](#5-算法适用性对比)
6. [月亮棋的算法匹配分析](#6-月亮棋的算法匹配分析)

---

## 1. MCTS (Monte Carlo Tree Search)

**位置：** `gavis/solvers/mcts.py`
**使用场景：** `demo.py` — 随机五子棋 9×9

### 1.1 算法概述

MCTS 是一种基于模拟的搜索算法，通过重复执行四个阶段逐步构建搜索树：

```
重复 budget 次:
  1. 选择 (Selection): 从根到叶子，使用 UCB1 选择子节点
  2. 扩展 (Expansion): 展开一个未探索的动作/outcome
  3. 模拟 (Simulation): 随机走棋到终局
  4. 回溯 (Backpropagation): 沿路径更新 value
```

### 1.2 实现细节

```python
class MCTS:
    budget: int = 5000        # 每步搜索迭代数
    ucb_c: float = 1.414      # UCB1 探索常数 (√2，经典值)
    rollout_depth: int = 20   # 模拟深度上限
```

**UCB1 公式：**
```
UCB = Q(s,a) + c * √(ln(N_parent + 1) / N_child)

其中:
  Q(s,a) = child.total_value / child.visits  (平均回报)
  c = 1.414 (ucb_c)
  N = 访问次数
```

**Chance 节点处理：**
- 完全展开所有 chance outcome（因为概率分布已知）
- 然后按概率随机选择一个进行模拟

**回溯 (Backpropagation)：**
- Player 节点: 符号翻转 (零和博弈: +1 对黑方 = -1 对白方)
- Chance 节点: 直接传递 (不反转)

### 1.3 复杂度

| 维度 | 值 |
|------|------|
| 每次搜索复杂度 | O(budget × (树深度 + rollout_depth)) |
| 每步最坏情况 | O(5000 × (棋盘大小² + 20)) |
| 空间复杂度 | O(budget) 个节点 (每次 search 后树丢弃) |
| 对 9×9 五子棋 | 每步约 0.5-2 秒 (Python 纯 CPU) |

### 1.4 优点

- **不需要训练**：纯搜索算法，即用即搜
- **支持随机博弈**：chance 节点处根据已知概率分布准确建模
- **任意时间算法**：随时可停止，返回当前最佳动作
- **完美信息游戏最优选择**：在足够预算下收敛到 Minimax 值

### 1.5 缺点

- **每次从头搜索**：不积累跨局知识，每步都要重复搜索
- **对搜索预算高度敏感**：5000 次迭代对 9×9 五子棋不够
- **Rollout 策略太弱**：随机 rollout 的评估方差大，需要更多迭代补偿
- **无泛化能力**：见过 100 次同一局面后仍会重新搜索

### 1.6 改进空间 (当前实现)

| 问题 | 改进方向 |
|------|---------|
| 随机 rollout 方差大 | 用神经网络评估替代随机 rollout (→ AlphaGo 式 MCTS) |
| 树每步丢弃 | 复用上一步的树 (Root Parallelization) |
| 单线程 | 根节点并行化 (Root Parallelization) 或叶节点并行 |
| 无先验知识 | 用 PPO 的网络输出作为先验概率 (→ Expert Iteration) |

---

## 2. CFR (Counterfactual Regret Minimization)

**位置：** `gavis/solvers/cfr.py`
**使用场景：** `demo_cfr.py` — 随机五子棋 5×5

### 2.1 算法概述

CFR 通过自博弈迭代最小化反事实遗憾值，逐渐接近纳什均衡策略。

```
External Sampling MC-CFR:

对每次迭代 (1..iterations):
  对每个玩家 (p_black, p_white):
    walk(root_state, updating_player, reach, depth=0)

walk:
  - 终端节点: 返回 utility
  - Chance 节点: 采样一个 outcome
  - 对手节点: 按当前策略采样一个动作
  - 更新方节点: 遍历所有动作 → 计算后悔值 → 更新存储
```

### 2.2 实现细节

```python
class CFR:
    iterations: int = 1000        # 训练迭代次数
    depth_limit: int = 8          # 最大递归深度
    rollout_depth: int = 15       # 超出深度后的 rollout
    use_cfr_plus: bool = True     # CFR+ 无符号遗憾截断
```

**遗憾匹配 (Regret Matching)：**
```
σ(a|I) = max(0, R⁺(I,a)) / Σ max(0, R⁺(I,b))
  其中 R⁺(I,a) 是 CFR+ 截断后的累积遗憾
```

**CFR+ 特性：**
- 所有 `regrets = max(0, regrets)` 在每个迭代后裁剪
- 没有负遗憾 → 更快收敛
- 平均策略仍然跟踪

### 2.3 复杂度

| 维度 | 值 |
|------|------|
| 每次迭代复杂度 | O(信息集数 × 分支因子) |
| 信息集数量 | 指数级于棋盘大小 |
| 5×5 棋盘 | ~10⁵-10⁶ 信息集 (可管理) |
| 9×9 棋盘 | ~10²⁰+ 信息集 (完全不可行) |
| 收敛速度 | O(1/√T) (普通 CFR), O(1/T) (CFR+) |

### 2.4 优点

- **理论保证收敛到纳什均衡**（在零和完美信息博弈中）
- **策略可解释**：可以直接查看每个信息集的策略分布
- **离线训练，在线使用快**：训练完成后 `get_action()` 是 O(|A|)
- **支持不完美信息扩展**（虽然这里只用完美信息）

### 2.5 缺点

- **状态空间爆炸**：仅能处理极小棋盘（5×5 已经是极限）
- **对 chance 节点处理粗糙**：外部采样只展开一个 chance outcome
- **rollout 只做随机**：与 MCTS 相同的随机 rollout 问题
- **没有 true online 玩法**：必须先跑完所有迭代再下棋
- **完美信息下 MCTS 通常优于 CFR**：CFR 为不完美信息设计，完美信息下 MCTS 更高效

### 2.6 与 MCTS 的对比 (在五子棋场景)

| 指标 | MCTS | CFR |
|------|:---:|:---:|
| 收敛目标 | 最优动作 | 纳什均衡 |
| 棋盘扩展 | ✅ 9×9 可运行 | ❌ 限制 5×5 |
| 在线/离线 | 在线搜索 | 离线训练 |
| 每次决策时间 | 0.5-2s | < 1ms (训练后) |
| 学习跨局知识 | ❌ | ✅ 策略网络 |

---

## 3. PSRO (Policy-Space Response Oracles)

**位置：** `moon_chess_ai/PSRO/train.py`
**使用场景：** 月亮棋 3×3

### 3.1 算法概述

PSRO 是 Double Oracle 算法的多智能体扩展，通过迭代扩展策略池来逼近纳什均衡。

```
初始化: 策略池 pi = [π₀]  (随机策略)

重复直到收敛:
  1. 构建收益矩阵 R[i,j] = U(πᵢ, πⱼ)   (gamescape)
  2. 求解纳什均衡: σ = Nash(R)           (solve_nash)  
  3. 计算最佳响应: π' = BR(σ)            (tabular_Q)
  4. 如果 π' ∉ pi: pi = pi ∪ {π'}
```

### 3.2 实现细节

```python
def PSRO_Q(env, num_iters=20, num_steps_per_iter=5000, eps=0.1, alpha=0.1):
    # 1. 初始化: 随机策略
    pi = random_policy_matrix(19683, 9)
    
    # 2. 主循环
    for niter in range(num_iters):
        R = gamescape(env, pi, Ne=10)       # 两两对战
        nash_p = solve_nash(R)              # 线性规划
        expl = exploitability(env, nash_pi, pi, Ne=300)
        
        # 训练最佳响应
        Q = tabular_Q(env, num_steps_per_iter, Q, epsilon=eps, alpha=alpha)
        beta = greedy_policy(Q)
        
        # 策略去重
        if beta not in pi:
            pi = concat(pi, beta)
```

**策略表示：** 每个策略是一个 `[19683, 9]` 的 one-hot 矩阵。

**纳什均衡：** 通过线性规划求解：
```
max  v
 s.t. R·σ ≥ v·1
      Σσ = 1, σ ≥ 0
```

### 3.3 复杂度

| 维度 | 值 |
|------|------|
| 每轮 gamescape | O(N² × Ne × rollout_length) | 
| 每轮 最佳响应 | O(num_steps_per_iter × 19683) |
| 总迭代 | 10-100 轮 (取决于收敛) |
| 3×3 月亮棋 | 完全可行 (19683 状态 × 9 动作) |
| N×N 月亮棋 | ❌ 不可行 (3^(N²) 爆炸) |

### 3.4 优点

- **纳什均衡有理论保证**：PSRO 收敛到博弈的纳什均衡
- **自动发现多样策略**：策略池逐步覆盖博弈的不同区域
- **可监控策略多样性**：通过 `exploitability` 和 `diversity` 指标
- **训练后推理极快**：纳什策略是查表操作

### 3.5 缺点

- **表格方法完全不可扩展**：`[19683, 9]` 已经是上限
- **gamescape 计算成本高**：O(N²) 次对战，N 增长后成本平方增长
- **对模拟噪声敏感**：`estimate_reward` 有高方差，需要大量 episodes
- **线性规划实现有 Bug**（详见代码风格分析）
- **策略去重仅靠"完全相同"**：不会识别功能上等价的策略

### 3.6 PSRO vs MCTS vs CFR 对比

| | MCTS | CFR | PSRO |
|---|:---:|:---:|:---:|
| 决策时搜索 | ✅ | ❌ | ❌ |
| 离线学习 | ❌ | ✅ | ✅ |
| 纳什均衡 | ❌ (最优动作) | ✅ | ✅ |
| 策略多样性 | ❌ | ❌ | ✅ |
| 可扩展性 | 中 | 低 | 极低 (表格) |
| 与 RL 结合 | ❌ | ❌ | ✅ (BR 可用任何 RL) |

---

## 4. PPO (Proximal Policy Optimization)

**位置：** `未命名文件夹/algorithms/ppo_agent.py`
**使用场景：** 月亮棋 3×3

### 4.1 算法概述

PPO 是一种策略梯度方法，通过剪切（clip）策略更新幅度来保证训练的稳定性。

```
在每轮训练中:
  1. 用当前策略与环境交互收集轨迹
  2. 计算 GAE (Generalized Advantage Estimation)
  3. 在经验上做 K 个 epoch 的 minibatch 更新
  4. 使用 PPO-clip 目标函数
```

### 4.2 实现细节

```python
class PPOConfig:
    gamma: float = 0.99           # 折扣因子
    gae_lambda: float = 0.95      # GAE λ 参数
    clip_epsilon: float = 0.2     # PPO 剪辑范围
    value_coef: float = 0.5       # Value loss 权重
    entropy_coef: float = 0.01    # 熵正则权重
    learning_rate: float = 3e-4   # Adam 学习率
    max_grad_norm: float = 0.5    # 梯度裁剪阈值
    update_epochs: int = 4        # 每批更新次数
    minibatch_size: int = 32      # mini-batch 大小
```

**网络结构：**
```
ActorCriticNetwork(
  backbone: Linear(38→128) → ReLU → Linear(128→128) → ReLU
  actor_head:  Linear(128→9)    → logits (masked)
  critic_head: Linear(128→1)    → value
)
```

**PPO-Clip 目标函数：**
```
L(θ) = -min( r(θ)Â, clip(r(θ), 1-ε, 1+ε)Â )
  + 0.5 * (V_θ - R)²
  - 0.01 * H(π_θ)

其中:
  r(θ) = π_θ(a|s) / π_old(a|s)  (重要性采样比)
  Â = 标准化后的 GAE
  H = 策略熵
```

**训练循环 (当前实现)：**
```
每 20 episodes:
  1. 随机选择 controlled_player (player_x 或 player_o)
  2. 对手为 RandomAgent
  3. 收集整局轨迹到 RolloutBuffer
  4. Episode 结束 → compute GAE → 4 epoch mini-batch 更新
```

### 4.3 复杂度

| 维度 | 值 |
|------|------|
| 网络参数量 | ~19,000 |
| 每步复杂度 | O(38×128 + 128×9) ≈ 5k FLOPS |
| 当前训练量 | 20 episodes (严重不足) |
| 合理训练量 | >100,000 episodes |
| 收敛时间 | 取决于棋盘复杂度 |

### 4.4 优点

- **神经网络泛化**：可以泛化到未见过的状态
- **随机策略**：动作选择有探索（通过高斯噪声/分类分布）
- **Action Mask**：利用编码器提供的合法动作掩码，避免无效探索
- **对问题规模不敏感**：相同算法可扩展到 5×5、9×9
- **持续学习潜力**：可以持续与环境交互改善

### 4.5 缺点

- **当前训练深度太浅**：20 episodes 远不能收敛
- **对手始终是 Random**：PPO 学到的策略只在对抗随机时有意义
- **自博弈缺失**：没有使用自对弈(self-play)或对手抽样来逐步提升
- **无经验回放**：只有 on-policy 数据，样本效率低
- **特征工程固定**：38 维编码针对 3×3 的手工设计

### 4.6 PPO 与表格方法的深度对比

| 维度 | 表格 Q (PSRO) | PPO 神经网络 |
|------|:---:|:---:|
| 状态泛化 | ❌ 查表 | ✅ 连续特征 |
| 样本效率 | 高 (直接 Q 更新) | 低 (on-policy) |
| 扩展性 | 3×3 上限 | ✅ 任何大小 |
| 收敛保证 | ✅ Q-learning 收敛性 | ❌ 策略梯度无保证 |
| 需要环境交互 | 是 | 是 |
| 能处理连续/大状态 | ❌ | ✅ |

---

## 5. 算法适用性对比

### 5.1 按游戏类型

| 游戏类型 | 最佳算法 | 理由 |
|---------|---------|------|
| 完全信息、小棋盘 (3×3) | MCTS / 表格方法 | 状态空间枚举可行 |
| 完全信息、中棋盘 (5×5-9×9) | MCTS | CFR 状态爆炸，PSRO 表格不可行 |
| 完全信息、大棋盘 (15×15+) | MCTS + 神经网络 | AlphaGo 式方案 |
| 随机博弈 (chance node) | MCTS / CFR (但不推荐) | MCTS 自然支持 |
| 需要纳什均衡 | CFR / PSRO | 理论保证 |
| 需要端到端视觉输入 | PPO + 视觉前端 | 完整学习管线 |

### 5.2 按项目需求

```
当前游戏:
  - 随机五子棋 (9×9, chance):              → MCTS 最合适
  - 月亮棋 (3×3, pieceOrder, 无 chance):   → 四种均可

求解目标:
  - 展示 AI 能力 vs 人类:                  → MCTS 或 PPO (有前端)
  - 学术比较/收敛性验证:                   → CFR 或 PSRO
  - 产品级可用:                            → PPO (可扩展) + MCTS (搜索精调)
  - 视觉输入 + AI 决策:                    → PPO + Vision Binding (唯一选择)
```

---

## 6. 月亮棋的算法匹配分析

### 6.1 月亮棋关键特征

1. **3×3 棋盘** — 极小状态空间
2. **先落先消 (pieceOrder FIFO)** — 最多 3 子/人
3. **三子连珠获胜** — 简单胜负条件
4. **完美信息** — 无隐藏信息
5. **确定性** — 无随机元素

### 6.2 各算法的表现预期

| 算法 | 预期表现 | 说明 |
|------|---------|------|
| **MCTS** | ⭐⭐⭐⭐⭐ | 3×3 极小搜索空间，几乎是秒算必胜策略 |
| **CFR** | ⭐⭐⭐⭐ | 3×3 信息集很少，快速收敛到纳什均衡 |
| **PSRO (表格)** | ⭐⭐⭐⭐ | 19683 状态全量枚举，均衡策略质量高 |
| **PPO (当前)** | ⭐⭐ | 20 episodes 未充分训练；如果有 100k+ 可到 ⭐⭐⭐⭐ |

### 6.3 推荐组合

对于月亮棋 3×3，最有价值的算法路线：

```
生产用途:  MCTS (保证必胜/必不败) → 演示效果最好
研究用途:  PSRO (追踪均衡收敛过程) → 学术价值最高
视觉产品:  PPO + Binding → 唯一可行的端到端管线
框架验证:  所有算法跑同一种游戏 → 验证 Gavis 引擎通用性
```
