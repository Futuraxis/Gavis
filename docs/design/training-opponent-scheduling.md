# 训练对手编排机制（Training Opponent Scheduling）

> 版本: v0.1 | 状态: 已实现 | 日期: 2026-10
> 范围: `layer3_solvers/marl/opponent_pool.py` + QMix / HAPPO / MAAC 三求解器
> 训练循环；注册表 `train-cli/games.py` 麻将管线默认开启。

## 问题：纯自博弈的震荡

MARL 三求解器（QMix/HAPPO/MAAC）原先在麻将 2 人局上做**纯自博弈**：每局
两个座位的决策都由"当前的自己"产生。两个网络互相追赶、共同漂移——
学习器每局的对手就是自己的即时镜像，胜率（尤其 vs-random 固定基线）在
噪声附近震荡，长期看就是一条**锯齿曲线**（逼近陷阱 / approximation trap，
网络为了赢"当前的自己"而不断推翻刚学会的行为）。

## 机制设计

对手编排把"这一局与谁打"从隐式（永远是当前镜像）变成显式决策，
全部由 `OpponentScheduleConfig` 扁平字段（求解器 config 上 `opponent_*`）
驱动，三个求解器共用同一套机制：

```
每局（episode）:
  learner := RoleScheduler.learner_for(ep)   # 2 人局逐局轮换学习器座位
  若 ep 是 checkpoint_interval 的倍数（且过了 warmup）:
     把 p0 / p1 各自当前策略冻结成 OpponentSnapshot，分别入各自对手池
     （两个池对称增长，避免座位轮换造成只有单边有池条目）
  opp := 学习器对手池按模式采样（池空 → None → 退化为纯自博弈）
  run_episode(selectors):
    学习器座位 → 当前策略（带探索，其 transitions 进入学习器）
    对手座位   → 冻结快照贪心（不带探索，其 transitions 不进入学习器）
  pool.record_win(opp.id, 学习器 payoff)   # 滚动胜负窗口，供 PFSP 加权
  每 eval_interval 局: eval_vs_random 采样一次曲线点（vs-random 固定基线）
```

### 采样模式（`opponent_mode`）

| 模式 | 采样权重 | 作用 |
|------|----------|------|
| `self` | 无池，纯自博弈 | 基线/回归对照（与旧行为逐字节一致） |
| `uniform` | `p_i ∝ 1` | 虚构自博弈（FSP）：均匀面对自身过去 |
| `pfsp` | `p_i ∝ max(win_rate_i, floor)^α`（`priority="win"`）或 `p_i ∝ max(1−win_rate_i, floor)^α`（`priority="lose"`） | 优先虚构自博弈（OpenAI Five 论文公式）：把训练时间稳定投向"当前能赢"（或打不赢）的对手 |
| `curriculum` | `p_i ∝ decay^age`（越新越快照权重越高） | 课程：从旧弱对手平滑过渡到新强对手 |

`pfsp_floor` 保证池内每个对手都有非零采样概率；`win_memory` 是每个对手的
滚动胜负窗口（胜=payoff>0，平=0.5，负=0）。

### 为什么能平滑曲线

1. **对手分布稳定**：学习器面对的是自身过去的**加权混合**，而不是随时
   变化的即时镜像——对手"难度分布"不再逐局剧烈跳动。
2. **反共同漂移**：两座位的策略不再互相追逐（一个变强另一个立刻变弱），
   vs-random 的评估曲线因此单调平滑上升而非锯齿震荡。
3. **资源定向**：`pfsp` 把训练时间集中投向稳定可强化的对手，避免把大量
   局数浪费在"刚被自己打败的新镜像"上。

### on-policy 性质

对手座位执行冻结快照时产生的 transitions **不进入**学习器的 replay
buffer / HAPPO 轨迹——Q 学习与策略梯度只用学习器自身采样数据，冻结对手
数据不会污染 TD 目标或优势估计（每个求解器的 `train()` 里
`tracked = learner if scheduled else None` 实现）。

## 配置入口

- 麻将登记表默认开启（`train-cli/games.py` `_MAHJONG_OPPONENT_CFG`）：
  `opponent_mode="pfsp"`、池容量 32、每 25 局入池一次、warmup 100 局、
  每 50 局做一次 vs-random 曲线采样（`eval_interval=50, eval_episodes=5`）。
- CLI 大参数/编排覆盖：`python train-cli/train.py --game mahjong_guangdong
  --solver qmix --episodes 1500 --config-override hidden_dim=512,opponent_mode=pfsp,
  opponent_checkpoint_interval=25,eval_interval=100`

## 产物与可验证性

- `metrics.json` 的每个求解器 `extra` 里新增：
  - `opponent_enabled`：是否编排模式；
  - `curve_roll`：训练中学习器视角滚动胜率（每 25 局采样，窗口 50）；
  - `curve_eval`：vs-random 固定基线采样点（`eval_interval` 控制），
    即"训练曲线平滑度"的直接证据。
- 单元测试：`tests/test_layer3_solvers/test_marl_opponents.py`
  （池机制 + 采样权重 + 淘汰 + 座位轮换 + 三求解器集成冒烟）。
- 运行结果：`docs/design/training-opponent-scheduling-results.md`
  （大参数麻将 1200 局 ×2 + 月亮棋平滑实证）。

## 运行与对比（大参数）

同一 seed、同一网络规模、同一局数下跑两条对照命令（输出目录必须分开，
否则后完成者会覆盖前者的 `metrics.json`/`qmix.pt`）：

```bash
# 基线：纯自博弈（大参数）
python train-cli/train.py --game mahjong_guangdong --solver qmix \
  --episodes 1200 --seed 42 \
  --config-override hidden_dim=512,opponent_enabled=false, \
    eval_interval=100,eval_episodes=8 \
  --out-dir models/train_baseline

# 编排：PFSP 对手池（大参数）
python train-cli/train.py --game mahjong_guangdong --solver qmix \
  --episodes 1200 --seed 42 \
  --config-override hidden_dim=512,opponent_enabled=true, \
    opponent_mode=pfsp,opponent_checkpoint_interval=25, \
    opponent_pool_capacity=32,opponent_warmup=100, \
    opponent_pfsp_priority=win,eval_interval=100,eval_episodes=8 \
  --out-dir models/train_scheduled
```

对比方法：两条运行的 `metrics.extra['curve_eval']`（vs-random 固定基线
胜率点列）直接可比——计算滑动窗口标准差（越小越平滑）、方向反转次数
（震荡次数）、首尾三采样均值差（提升幅度）；`curve_roll`（学习器滚动
胜率）同理。`--config-override` 支持任意求解器字段（含 `hidden_dim`、
`opponent_*`、`eval_*`），bool/int/float/None/str 自动转换。