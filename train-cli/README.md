# train-cli — Gavis 统一训练 CLI（游戏注册制）

`train-cli/` 是唯一的训练入口，采用**游戏注册制**：所有游戏在 `games.py`
的 `GAMES` 注册表中登记，训练脚本 `train.py` 只读注册表数据，**不含任何
per-game 分支**。接入新游戏 = 在注册表新增一个 `GameSpec` 条目，训练与
运行时装配逻辑零改动。

## 文件

| 文件 | 作用 |
|------|------|
| `games.py` | 游戏注册表（配置数据）+ 通用求解器工厂（`create_solver`）与 `default_provider` |
| `train.py` | 统一抽象训练脚本（引擎构造/训练/评估全部由注册表驱动） |
| `README.md` | 本文件 |
| `../train_cli.py` | 根目录导入桥：使连字符目录可 `import train_cli` / `python -m train_cli` |

## 用法

```bash
# 注册表一览（游戏 × 训练管线）
python train-cli/train.py --list

# 训练全部已登记游戏（产物 models/train/<game>/：config.json + metrics.json + 求解器产物）
python train-cli/train.py --game all

# 按游戏 / 按求解器过滤（求解器支持逗号分隔）
python train-cli/train.py --game moon_chess --solver hybrid
python train-cli/train.py --game texas_holdem --solver qmix,happo,maac
python train-cli/train.py --game mahjong_guangdong --solver all

# 常用参数
python train-cli/train.py --game moon_chess --solver ppo --episodes 100 --seed 42 \
                          --device auto --out-dir models/train --skip-eval --verbose

# 等价桥接入口（模块化）
python -m train_cli --game all --solver cfr
```

### CLI 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--game` | `all` | 游戏 id（逗号分隔）或 `all` |
| `--solver` | `all` | 求解器名（逗号分隔）或 `all`（只取该游戏已登记的交集） |
| `--episodes` | 注册表默认 | 覆盖各管线训练局数（solve 模式忽略） |
| `--seed` | `42` | 全链路种子 |
| `--device` | `auto` | `auto`/`cpu`/`cuda` |
| `--out-dir` | `models/train` | 产物根目录（每个游戏一个子目录） |
| `--eval-episodes` | 注册表默认 | 每个评估对手的局数（座位轮换） |
| `--eval-opponents` | 注册表 `eval_opponents` | 评估对手（逗号分隔：`random`/`self` 或任何已登记 `runtime_solvers` 名字，如 `mcts,mahjong`；未登记时该列自动跳过） |
| `--skip-eval` | — | 跳过评估 |
| `--preset` | `full` | 训练预设：`full`（完整训练）/ `quick`（快速演示校准，训练局数 0.2 缩放） |
| `--config-override` | — | `KEY=VALUE`（可多次）：覆盖求解器 config 的字段（如 `--config-override budget=500`） |
| `--list` | — | 打印注册表一览并退出 |
| `--verbose` | — | 训练过程详细输出 |

## 注册表数据结构

```python
# train-cli/games.py
GAMES: dict[str, GameSpec] = {
    "moon_chess": GameSpec(
        game_id="moon_chess",
        display_name="月亮棋",
        engine=EngineSpec(rules="moon_chess.json"),          # v5.0.0 无 variants；需变种/人数的游戏在 EngineSpec 里选 variant/player_count
        players=("p_black", "p_white"),                      # 座位顺序（先手在前）
        solvers={
            "hybrid": SolverPipeline(
                "hybrid",
                episodes=1,                                   # train() 局数
                config={...},                                 # config-class kwargs
                # 路径可用 $OUTDIR 占位，由 train.py 展开
            ),
            "cfr": SolverPipeline("cfr", entry="solve", config={...}),
            "qmix": SolverPipeline("qmix", episodes=600, save="qmix.pt"),
        },
        eval_episodes=20,
        runtime_solvers=("mcts", "cfr", "hybrid", "random"),  # 运行时装配（数据驱动）
        runtime_configs={...},                                 # 运行时配置覆盖
    ),
    ...
}
```

`SolverPipeline` 字段：

- `solver` — `SOLVER_FACTORY` 中的求解器名。
- `entry` — `"train"`（`solver.train(episodes)`）/ `"solve"`（`solver.solve(initial)`，CFR）。
- `episodes` — 训练局数（`solve` 模式忽略；PSRO 的 `episodes` 是迭代数）。
- `config` — 传给 config-class 的 kwargs；`"$OUTDIR"` 占位符在训练开始时展开为
  该游戏输出目录（如 `cfr_table_path="$OUTDIR/cfr_table.json"`）。
- `save` — 产物文件名（写入 `<out-dir>/<game>/`；未实现 `save()` 或未声明则跳过）。
- `per_player` — `True` → 每个座位一个实例（`player_id=座位`，贝叶斯狼人杀用）。
- `eval` — 训练后是否运行评估（对手由 `eval_opponents` 数据驱动声明）。
- `eval_opponents`（GameSpec）— 评估对手列，数据驱动：
  - `random` 均匀随机（内置下限基准）
  - `self` 自博弈镜像（内置；per_player 时各座位用自身实例互博）
  - 其余任意名字 = 必须在该游戏 `runtime_solvers` 里已登记（如 `mcts` 搜索基线、
    `mahjong` 启发式基线、`ollama`），经 `create_solver` 通用装配（预算统一
    `EVAL_MCTS_BUDGET=300` 规模）；**未登记则该列自动跳过并提示**——接入新
    基线只需在注册表登记，评估代码零改动

## 已登记游戏

| 游戏 | rules | 变种/人数 | 训练管线 |
|------|-------|-----------|----------|
| `moon_chess` | moon_chess.json | — | hybrid / cfr / ppo / psro / qmix / happo / maac |
| `stochastic_gomoku` | stochastic_gomoku.json | — | hybrid(CFR 先验) / cfr |
| `texas_holdem` | texas_holdem.json | — | hybrid(不完全信息) / qmix / happo / maac |
| `mahjong_guangdong` | mahjong.json | guangdong × 4p | qmix / happo / maac |
| `mahjong_hongzhong` | mahjong.json | hongzhong × 4p | qmix / happo / maac |
| `mahjong_blood` | mahjong.json | blood × 4p | qmix / happo / maac |
| `mahjong_sichuan` | mahjong.json | sichuan × 4p | qmix / happo / maac |
| `mahjong_changsha` | mahjong.json | changsha × 4p | qmix / happo / maac |
| `mahjong_taiwan` | mahjong.json | taiwan × 4p | qmix / happo / maac |
| `mahjong_international` | mahjong.json | international × 4p | qmix / happo / maac |
| `werewolf` | werewolf.json | 默认 9 人（3 狼+3 村+预言家+女巫+猎人） | bayes(per_player，训练 no-op，仅评估) |
| `undercover` | undercover.json | fruit_normal × 8p（4..12 人） | —（`solvers={}` 无训练管线；运行时 ollama/random） |
| `uno` | uno.json | classic × 4p（2..10 人） | hybrid(不完全信息, `imperfect_information=True`) |
| `uno_seven_zero` | uno.json | seven_zero × 4p（2..10 人） | hybrid(不完全信息, `imperfect_information=True`) |
| `uno_jump_in` | uno.json | jump_in × 4p（2..10 人） | hybrid(不完全信息, `imperfect_information=True`) |
| `uno_stacking` | uno.json | stacking × 4p（2..10 人） | hybrid(不完全信息, `imperfect_information=True`) |
| `uno_draw_until` | uno.json | draw_until × 4p（2..10 人） | hybrid(不完全信息, `imperfect_information=True`) |
| `uno_strict_wild4` | uno.json | strict_wild4 × 4p（2..10 人） | hybrid(不完全信息, `imperfect_information=True`) |

## 运行时装配（前端/基准共用）

`games.py` 同时提供通用运行时工厂（供 `layer4_interface` 各 server 与
benchmark 注入，遵守“L4 不 import L3”的层间约定）：

```python
from train_cli import create_solver, default_provider

solver = create_solver("moon_chess", "mcts", engine, seed=42, budget=2000)
# 或通过 provider 对象（同上，另支持 online_models 经验表注入）
solver = default_provider.create_solver("moon_chess", "hybrid", engine, 42, 2000)
```

- 求解器对该游戏的可用性（`runtime_solvers`）、运行时配置（`runtime_configs`，
  如德州 Hybrid 的 `imperfect_information`）全部由注册表数据驱动，工厂内零分支。
- 未登记游戏 / 不适用求解器 → `ValueError`（列出可选值）。
- 可选依赖（torch/psro）缺失的求解器 → 实例化时报可执行错误信息。

## 约定

- 训练脚本不得出现 `if game == ...` 之类的 per-game 分支；一切游戏差异来自注册表。
- 新游戏接入流程：`rules/<game>.json`（v5.2 variants 声明式）→ 在 `GAMES` 登记
  `GameSpec` → `train.py --list` 确认管线 → 训练。无需改动 `train.py`。
- 层间契约：训练只通过 `GameEngine`（L2→L3）与 `SolverBase`（L3）交互。
