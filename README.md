# Gavis — 自适应策略游戏 AI Agent

> **G**ame **A**dapti**v**e **I**ntelligence **S**ystem

## 架构概览

```
┌─────────────────────────────────────────────────┐
│  Layer 1: Translator      [LLM 规则翻译]        │
│  Layer 2: Env/Engine      [游戏引擎 + 规则引擎]  │
│  Layer 3: Solver           [MCTS/CFR/PPO/PSRO]  │
│  Layer 4: Interface       [VLM 识别 + 前端]     │
└─────────────────────────────────────────────────┘
```

四层架构，详见 [`docs/merge/`](docs/merge/)。

## 快速开始

```bash
# 安装基础依赖
pip install numpy

# 安装可选依赖
pip install "gavis[torch]"     # PPO 求解器
pip install "gavis[binding]"   # 视觉识别
pip install "gavis[psro]"      # PSRO 求解器
pip install "gavis[all]"       # 全部
```

## 玩一局（平台前端）

```bash
# 1. 构建前端（首次需要 Node.js ≥ 18）
cd platform-frontend && npm install && npm run build && cd ..

# 2. 启动平台服务（默认 http://127.0.0.1:8770/）
python -m layer4_interface.frontend.platform.server

# 3. 浏览器打开 http://127.0.0.1:8770/ → 游戏大厅 → 选游戏开战
```

平台内置 15 款游戏（月亮棋 / 随机五子棋 / 德州扑克 / 麻将六变种 / UNO 六变种），
支持聊天开局、人机对战、复盘回放、在线学习开关；自定义游戏经「创建游戏」页
接入（见 `docs/user/custom_games.md`）。前端开发模式：`cd platform-frontend && npm run dev`
（5173 端口，`/api` 自动代理到 8770）。

## 训练 CLI（游戏注册制）

所有游戏在 `train-cli/games.py` 注册表中登记（引擎构造、座位、可训练求解器
管线与默认超参全部是配置数据）。统一训练脚本 `train-cli/train.py` 只读注册表，
**不含任何 per-game 逻辑**：新游戏接入 = 新增一个登记条目。

```bash
# 查看注册表一览（7 个游戏 × 训练管线）
python train-cli/train.py --list

# 训练全部已登记游戏的默认管线（产物在 models/train/<game>/）
python train-cli/train.py --game all

# 单游戏 × 单求解器：月亮棋 Hybrid（CFR+PSRO 先验 + MCTS 搜索）
python train-cli/train.py --game moon_chess --solver hybrid

# 麻将变种 × MARL 求解器
python train-cli/train.py --game mahjong_guangdong --solver qmix,happo,maac

# 等价桥接入口（连字符目录的导入桥）
python -m train_cli --game texas_holdem --solver hybrid
```

## 项目结构

```
layer1_translator/      # LLM 规则翻译层（模板 + LLM 编排 + schema 校验）
layer2_engine/          # 游戏引擎核心（无 per-game 适配器）
  core/                 # GameEngine + state_graph + expr_eval
layer3_solvers/         # 求解器
  base.py               # SolverBase 抽象类
  mcts/                 # 蒙特卡洛树搜索
  cfr/                  # 反事实遗憾最小化
  ppo/                  # 近端策略优化
  psro/                 # 策略空间响应 Oracle
  auto_selector/        # (预留) 自动选择求解器
layer4_interface/       # 交互界面
  binding/              # VLM 图片识别
  encoding/             # 状态特征编码
  frontend/             # Web 服务 + 平台前端
  online_learning/      # 在线学习（轨迹捕获 + 经验对手模型 + 门禁发布）
  agent/                # 陪伴 Agent（性格/场景对话/局势评估）
  vision_bridge.py      # 识别→求解器的桥梁
train-cli/              # 训练 CLI：games.py 游戏注册表 + train.py 统一训练脚本
train_cli.py            # 根目录导入桥（train-cli/ 的模块化别名）
platform-frontend/      # 平台前端（React + Vite + TS，构建产物 dist/）
rules/                  # 游戏规则 JSON（v5.2 零 BUILTIN + variants 声明式）
tests/                  # 测试
archive/                # 原始旧代码存档
docs/                   # 架构设计 + 用户文档 + 审计
```

## 四层详解

详见 `docs/merge/` 下的架构文档：

1. [`01_architecture_design.md`](docs/merge/01_architecture_design.md)
2. [`02_architecture_pros_cons.md`](docs/merge/02_architecture_pros_cons.md)
3. [`03_architecture_comparison.md`](docs/merge/03_architecture_comparison.md)
4. [`04_code_style_analysis.md`](docs/merge/04_code_style_analysis.md)
5. [`05_algorithm_analysis.md`](docs/merge/05_algorithm_analysis.md)
6. [`06_merge_architecture_and_migration.md`](docs/merge/06_merge_architecture_and_migration.md)

## 许可证

MIT
