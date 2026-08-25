# 训练对手编排 — 运行结果（大参数麻将 + 月亮棋平滑实证）

> 版本: v0.1 | 日期: 2026-08 | 前置设计: `docs/design/training-opponent-scheduling.md`
> 范围: QMix（QMixSolver）+ `layer3_solvers/marl/opponent_pool.py` 编排机制

## 1. 大参数麻将（2 人广东鸡胡，hidden=512，1200 局）

同一 seed 42、同一网络规模的两条对照（输出目录分开）：

| 运行 | 命令要点 | 耗时 | 终局评估 vs-random |
|------|----------|------|--------------------|
| 基线（纯自博弈） | `opponent_enabled=false` | 3565 s ≈ 59.4 min | win_rate 0.000, avg +0.000（8 局全流局） |
| 编排（PFSP 池） | `opponent_enabled=true, opponent_mode=pfsp, checkpoint 25, pool 32, warmup 100` | 5172 s ≈ 86.2 min（并发负载） | win_rate 0.000, avg +0.000（8 局全流局） |

`curve_roll`（学习器滚动胜率，每 25 局采样，窗口 50）：两条运行各 48 个点，
**46 点为 0.0，仅 2 点为 0.02**（1200 局中学习器合计仅胜 ~1 局）；
`curve_eval`（vs-random 采样）：全部 0.0（每次采样 8 局全流局）。

**结论（重要发现）**：2 人广东鸡胡的终局奖励极其稀疏——弱/随机策略下游戏几乎
必然流局（`layer2_engine` 胡牌路径经 `tests/test_layer2_engine/test_mahjong.py`
26 项测试验证正常，但策略要完成 4 面子+1 对子的鸡胡手牌在 2p 短墙内概率极低），
因此 QMix 几乎收不到非零 TD 信号，两条曲线都平坦。**这解释了该仓库 QMix
麻将训练一贯"评估全平局"的现状** —— 是奖励稀疏性，不是编排机制失效；
在此信号上无法演示"曲线平滑"差异。

## 2. 月亮棋平滑实证（同机制，奖励密集信号的对照）

同 seed 42、600 局、hidden=128，基线 vs PFSP 编排（机制与游戏无关）：

| 指标 | 基线（自博弈） | PFSP 编排 |
|------|----------------|-----------|
| 终局 CLI 评估 vs-random win_rate / avg_utility | 0.300 / −0.400 | **0.750 / +0.500** |
| curve_eval 双座平均胜率（12 采样点） | 0.448 | **0.604** |
| 方向反转次数（两座合计） | 10 | **6** |
| 振荡 std（滑窗 W3，黑/白） | 0.161 / 0.151 | 0.154 / **0.102** |
| curve_roll std（滑窗 W10） | 0.0461 | **0.0427** |
| 白座终局段走势 | 0.5 → **0.0**（塌缩，自博弈共同漂移典型症状） | 0.5 → **1.0**（平滑上升） |

要点：

- **更强**：PFSP 编排终局 vs-random 胜率 0.75（基线 0.30，avg 从 −0.40 翻到
  +0.50）；curve_eval 双座平均 0.604 > 0.448。
- **更平滑**：方向反转 10 → 6、白座振荡 std 0.151 → 0.102、黑座 0.161 → 0.154。
- **反塌缩**：基线白座被自博弈共同漂移拖垮（0.5→0.0）；编排版白座被池对手
  支撑，平滑升至 1.0 —— 正是 PFSP 让学习者面对"自身过去的加权混合"、
  资源定向投给可稳定强化的对手的直接效果。

## 3. 可复现命令

```bash
# 月亮棋对照（各 ~26 s）
python train-cli/train.py --game moon_chess --solver qmix --episodes 600 --seed 42 \
  --config-override hidden_dim=128,opponent_enabled=false,eval_interval=50,eval_episodes=8 \
  --out-dir models/train_moon_baseline
python train-cli/train.py --game moon_chess --solver qmix --episodes 600 --seed 42 \
  --config-override hidden_dim=128,opponent_enabled=true,opponent_mode=pfsp, \
    opponent_checkpoint_interval=25,opponent_pool_capacity=24,opponent_warmup=50, \
    opponent_pfsp_priority=win,eval_interval=50,eval_episodes=8 \
  --out-dir models/train_moon_scheduled

# 大参数麻将（各 60–90 min；产物在 models/train_baseline / models/train_scheduled）
python train-cli/train.py --game mahjong_guangdong --solver qmix --episodes 1200 --seed 42 \
  --config-override hidden_dim=512,opponent_enabled=false,eval_interval=100,eval_episodes=8 \
  --out-dir models/train_baseline
python train-cli/train.py --game mahjong_guangdong --solver qmix --episodes 1200 --seed 42 \
  --config-override hidden_dim=512,opponent_enabled=true,opponent_mode=pfsp, \
    opponent_checkpoint_interval=25,opponent_pool_capacity=32,opponent_warmup=100, \
    opponent_pfsp_priority=win,eval_interval=100,eval_episodes=8 \
  --out-dir models/train_scheduled
```

对比脚本：`.scratch/moon_curves.py`（曲线序列 + 滑窗 std + 反转次数 + 首尾增益）。
平滑度指标定义见 `docs/design/training-opponent-scheduling.md` 末尾。