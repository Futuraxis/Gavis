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

四层架构，详见 [`CLAUDE.md`](CLAUDE.md)。

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

平台内置 18 款游戏（月亮棋 / 随机五子棋 / 德州扑克 / 麻将七变种 / UNO 六变种
/ 狼人杀 / 谁是卧底），支持聊天开局、人机对战、复盘回放、在线学习开关；
自定义游戏经「创建游戏」页接入。
前端开发模式：`cd platform-frontend && npm run dev`
（5173 端口，`/api` 自动代理到 8770）。

## 训练 CLI（游戏注册制）

所有游戏在 `train-cli/games.py` 注册表中登记（引擎构造、座位、可训练求解器
管线与默认超参全部是配置数据）。统一训练脚本 `train-cli/train.py` 只读注册表，
**不含任何 per-game 逻辑**：新游戏接入 = 新增一个登记条目。

```bash
# 查看注册表一览（18 个游戏 × 训练管线）
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
layer1_translator/      # LLM 规则翻译层（模板 + LLM 编排 + schema/冒烟校验）
layer2_engine/          # 游戏引擎核心（无 per-game 适配器，无 interfaces/）
  core/                 # GameEngine + state_graph + expr_eval + rules_compiler
layer3_solvers/         # 求解器
  base.py               # SolverBase 抽象类
  mcts/ cfr/ hybrid/    # 搜索/遗憾/混合求解器
  ppo/ psro/            # 策略梯度 / 策略空间响应 Oracle
  marl/                 # 多智能体（QMix/HAPPO/MAAC + PFSP 对手池）
  mahjong/ uno/ werewolf/ social/ llm/   # 游戏启发式 + 社交/LLM 求解器
  auto_selector/        # （占位）规则分析器（粗略猜测，未完整实现）
layer4_interface/       # 交互界面
  binding/              # 视觉识别（Image/DOM/Vision/Mock Binding）
  encoding/             # 状态特征编码
  frontend/             # Web 服务（platform 平台 + vision 视觉）
  online_learning/      # 在线学习（轨迹捕获 + 经验对手模型 + 门禁发布）
  agent/                # 陪伴 Agent（教练/对手/人格/对话/隐藏信息守卫）
  difficulty/ profile/ review/   # 自适应难度 / 偏好档案 / 复盘
  botzone/ aifight/     # Botzone 接入 / AIFight OpenAI 兼容桥
  vision_bridge.py      # 识别→求解器的桥梁
train-cli/              # 训练 CLI：games.py 游戏注册表 + train.py 统一训练脚本
train_cli.py            # 根目录导入桥（train-cli/ 的模块化别名）
scripts/                # 规则生成器（_gen_*.py）与训练/同步脚本
platform-frontend/      # 平台前端（React + Vite + TS，构建产物 dist/）
rules/                  # 游戏规则 JSON（零 BUILTIN；mahjong/werewolf/undercover/uno 为 v5.2 variants 声明式）
tests/                  # 测试（1435 用例）
archive/                # 原始旧代码存档
docs/                   # 架构设计 + 规则语言 + 用户文档
```

## 许可证

MIT
