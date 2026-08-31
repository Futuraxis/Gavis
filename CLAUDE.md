# Gavis 项目指南

## 项目概述

自适应策略游戏 AI Agent — 四层水平集成架构。

## 目录结构

```
rules/                 游戏规则 JSON（零 BUILTIN；7 个文件：
                       moon_chess / stochastic_gomoku 为 v5.0.0，
                       texas_holdem 为 v5.1.0（无 variants 节），
                       mahjong / werewolf / undercover / uno 为 v5.2
                       variants 声明式）
layer1_translator/    LLM 规则翻译层（确定性模板 TemplateTranslator +
                       LLM 编排/修复循环 + v5.5 增量补丁 rule_patch +
                       schema 校验 + L2 冒烟校验下沉）
layer2_engine/        游戏引擎核心（GameEngine 单一契约；无 per-game
                       适配器、无 interfaces/）
layer3_solvers/       求解器（MCTS/CFR/Hybrid/PPO/PSRO + MARL(QMix/HAPPO/
                       MAAC) + mahjong/uno 启发式 + werewolf 贝叶斯 +
                       social/LLM + auto_selector(占位)）
layer4_interface/     交互界面（Binding/Encoding/Frontend/OnlineLearning +
                       agent/difficulty/profile/review + botzone/aifight +
                       vision_bridge）
train-cli/            训练 CLI：games.py 游戏注册表（配置驱动，18 游戏）
                       + train.py 统一训练脚本
train_cli.py          根目录导入桥 → train-cli/（python -m train_cli）
scripts/              规则生成器（_gen_mahjong / _gen_werewolf / _gen_uno /
                       _gen_undercover.py）+ 训练/同步脚本（sync_maac_models /
                       train_maac_resume / eval_mahjong / build_botzone_zip）
tests/                测试（1435 collected；UNO 引擎 39 例；9 处 torch skipif）
platform-frontend/    平台前端（React + Vite + TS，构建产物 dist/ 已 gitignore）
data/                 运行时数据（对局 data/matches/、在线学习
                       data/online_learning/、档案/LLM 配置/会话/自定义游戏；
                       已 gitignore）
docs/                 架构设计 + 规则语言 + 用户文档
docs/design/          architecture / game-model / online-learning /
                       security-notes / frontend-contract / gamerule(v4.1~v5.2)
docs/user/            使用说明（play_* / custom_games / teaching /
                       llm_config / aifight / botzone）
.docs/                本地文档存档（已 gitignore；设计/审计/历史文档的
                       本地副本，不入库）
archive/              原始旧代码只读存档
```

自定义游戏工作流：`layer1_translator/variant_translator.py`（确定性参数路径 +
LLM 全量改写 / v5.5 增量补丁修复循环，输出必过 `engine_validator`
= schema + L2 冒烟校验）+ `layer4_interface/frontend/platform/families/`
（规则族：grid/poker/mahjong/social，detect+build_spec 自动发现）+
`custom_games.py`（注册表）+ 前端 `/create` 创建游戏页；引擎一律
`allow_codegen=False` 纯解释器路径（见 `docs/design/security-notes.md`）。

## 代码规范

完整的代码规范、格式化规则、命名约定和层间契约详见 **[docs/coding-standards.md](docs/coding-standards.md)**。

核心要点：
- **Python 3.11+**, `from __future__ import annotations`
- `ruff format` + `ruff check` (line-length=120)
- 全覆盖类型标注，优先 `X | None` 而非 `Optional[X]`
- 标准库 → 第三方 → 项目内部导入分组
- 层内相对导入 (`..base`)，跨层绝对导入 (`layer2_engine.*`)
- Google 风格文档字符串（`---` 分隔章节标题）
- Dataclass 配置 + Protocol 契约 + ABC 抽象基类
- 自定义异常层次（每层一个 Base 类）
- pytest 测试（固定 seed、fixture、可选依赖 skipif）

## 关键约定

- **Python 3.11+**, 类型标注全覆盖 (`from __future__ import annotations`)
- `ruff` 格式化 + lint (line-length=120)
- `pytest` 运行测试: `python -m pytest`
- 层间通信只能通过契约（Layer2→3: `GameEngine`；Layer3: `SolverBase`；
  Layer4: `BaseBinding`）
- **禁止循环依赖**：Layer N 只能依赖 Layer N-1
- Layer 4 原则上不依赖 Layer 3 — **唯一例外**：`botzone/mahjong_format.py`
  直引 `SolverConfig` + `MahjongHeuristicAI`（Botzone 薄适配边界）；
  其余装配一律经 `train-cli/games.py` 的 `create_solver` / `default_provider`
- 规则 JSON 放在 `rules/` 顶层，不在层目录内
- **v5.1/v5.2 零 BUILTIN**：规则自足，`BUILTIN_FUNCTIONS` 已退役；引擎从
  `rules["functions"]` 读取 alias 定义（`{"params": [...], "expr": {...}}`，
  构造时 `expr.set_functions(...)` 注册，禁递归、解释器 32 层深度上限）
- 语言原语只从数学操作角度增加（`choose/range/sort/group/at/add/sub/...`），
  禁止游戏特供原语
- 判定遵循增量局部原则：围绕 `lastPlacedCell` / `lastDiscard` /
  `last_action` 做 O(1) 局部判定
- **v5.2 variants 声明式**：变种/人数/配比都在单个 JSON 的 `variants`
  节声明（`variant` + `player_count` + `options: {<变体名>: {constants 补丁}}`
  —— **dict 形，非列表**；`$variant/$player_count/$constants/$players`
  上下文 + `trim_players/trim_utility`），引擎做纯数据解析，
  **没有任何 constants 注入 API**；未知 variant → ValueError
- 存量规则版本混用：moon_chess / stochastic_gomoku = 5.0.0（无 variants、
  无 functions）、texas_holdem = 5.1.0（有 functions + hiddenWorld，
  无 variants）；"v5.2 声明式"覆盖 mahjong / werewolf / undercover / uno
- `rules/mahjong.json` 由 `scripts/_gen_mahjong.py` 生成（**7 变种**：
  guangdong / hongzhong / blood / sichuan / changsha / taiwan /
  **international(国标)**；默认 4 人；136 张基础 = guangdong/hongzhong/
  taiwan/international，108 张补丁 = blood/sichuan/changsha），改规则改
  生成器再重新生成；**平台默认 AI=已训练 MAAC**（`models/train/<game_id>/maac.pt`，
  运行时工厂按游戏注入该路径，产物缺失回退启发式，平台不崩）；共享产物
  同步见 `scripts/sync_maac_models.py`：136 组 guangdong→(hongzhong,taiwan)、
  108 组 sichuan→(blood,changsha)（international 无共享产物）
- `rules/werewolf.json` 由 `scripts/_gen_werewolf.py` 生成（默认 9 人/3 狼，
  配比本体在 `constants.role_pool`、由 variants 的 player_ids 公式引用）；
  结算阶段（夜晚/放逐）用 `chance` 模板 + `effectMap` 表达（explicit
  概率 1.0）；部分可观测由 `visibility` 声明（`my_role`/`dead_roles` 视图 +
  `env` 字段投影 → `seerResult` 仅预言家可见）；轮转由 env/`turn` 驱动；
  `speak` 用 text 参数预制能力
- `rules/uno.json` 由 `scripts/_gen_uno.py` 生成（108 张牌 × 2-10 人 ×
  六变体 classic/seven_zero/jump_in/stacking/draw_until/strict_wild4
  全在 variants 声明，默认 4 人）；出牌/罚牌/抢牌/叠加全部用 `env` +
  `chance` 表达；牌堆无实体数组（`deck = 108 − 手牌 − 弃牌`，
  `undrawn_cards` 查询 + `deck_count` 别名）；手牌部分可观测
  （`visibility` 隐藏他人牌面但保留张数）；`hand_of` 别名（10 分支 switch）
  可内联进查询/合法条件——编译器对 switch 生成无 walrus 的首匹配 if/elif
  链（`:=` 在 comprehension 迭代表达式里是 SyntaxError，曾经的
  switch-in-comprehension 会让整套规则回退纯解释器；现可正常编译）
- **text 参数预制能力**：动作参数声明 `"type": "text"` 时不参与合法
  动作枚举（占位 `""`），solver 把文本放进 `ActionInstance.params`，
  effector 经 `$text` 读取（`_build_context` 自动平铺 params）——
  自由文本发言游戏（狼人杀/谁是卧底等）用，规则语言通用能力非游戏特供
- **v5.5 增量补丁协议**（L1）：`layer1_translator/rule_patch.py`
  （RFC-6902 风格子集：`{"patch":[{"op":"replace|add|remove","path":"...",
  "value":...}]}`，`apply_patch` 深拷贝按序应用、失败抛 PatchError 绝不
  半份应用）——大模板变体（mahjong ≈87k 字符）走补丁而非全量改写
- 平台注册表 `layer4_interface/frontend/platform/games.py` 与训练注册表
  `train-cli/games.py` 均登记 **18 游戏**（月亮棋/随机五子棋/德州扑克 +
  麻将 7 变种 + UNO 6 变体 + 狼人杀 + 谁是卧底；undercover 无训练管线）

## 求解器注册

新求解器 → `layer3_solvers/base.py` 实现 `SolverBase` → 注册到 `__init__.py`
（PPO/MARL 依赖 torch、PSRO 依赖 gymnasium/scipy/tqdm，缺失时对应 class
为 `None`，`train-cli/games.py` 的工厂实例化时给出明确报错）。

## 文档

| 文件 | 内容 |
|------|------|
| `docs/coding-standards.md` | 代码规范（格式化/命名/类型/层间契约） |
| `docs/design/architecture.md` | 当前四层架构设计 (v0.4) |
| `docs/design/game-model.md` | 随机博弈模型（POSG → 声明式 rules） |
| `docs/design/frontend-contract.md` | 平台前端信封契约与教训 |
| `docs/design/multi-agent.md` | 多智能体架构补充（多座位/MARL/PSRO/陪伴 Agent/人机闭环） |
| `docs/design/online-learning.md` | 在线学习设计（捕获/经验对手模型/门禁发布） |
| `docs/design/security-notes.md` | 安全与性能决策记录（审计 3.6 修复项与暂缓项） |
| `docs/design/gamerule/v5.2.md` | 现行规则语言 v5.2（variants 声明式 + visibility） |
| `docs/design/gamerule/v5.1-reference.md` | v5.1/v5.2 语法参考 |
| `docs/design/gamerule/v5.1.md` / `v5.0.md` | 规则语言历史版本（已被 v5.2 取代） |
| `docs/design/gamerule/v4.1.md` / `v4.1-compact.md` | v4.1 图模型时代（已废弃，仅存档） |
| `docs/user/play_moon_chess.md` | 月亮棋人机对弈使用说明 |
| `docs/user/play_gomoku.md` | 随机五子棋人机对弈使用说明 |
| `docs/user/play_texas_holdem.md` | 德州扑克人机对弈使用说明 |
| `docs/user/play_mahjong.md` | 麻将人机对弈使用说明（七变种 × 默认 4 人） |
| `docs/user/play_uno.md` | UNO 使用说明（六变体 × 2-10 人，rules/uno.json） |
| `docs/user/play_undercover.md` | 谁是卧底使用说明（6 主题 × 3 难度 × 4-12 人，text 发言桌游） |
| `docs/user/play_werewolf.md` | 狼人杀使用说明（9 人/3 狼，text 发言桌游） |
| `docs/user/custom_games.md` | 自定义游戏使用说明（自然语言规则 / 模板变体 → 平台可对弈；四规则族） |
| `docs/user/teaching.md` | 教学对局使用说明（教练看玩家自己的牌推理：agent/coach.py 三红线——玩家投影/双脑分离/raw_solver 防录制污染） |
| `docs/user/llm_config.md` | LLM 服务配置（端点与模型） |
| `docs/user/aifight.md` | AIFight 接入（OpenAI 兼容桥） |
| `docs/user/botzone.md` | Botzone 接入（本地 AI / 远程接口 / 上传包） |

## 常用命令

```bash
python -m pytest tests/ -v                          # 跑测试
python train-cli/train.py --list                    # 查看游戏注册表一览（18 游戏）
python train-cli/train.py --game all                # 训练所有已登记游戏的默认管线 (完整训练)
python train-cli/train.py --game moon_chess --solver hybrid   # Hybrid 训练 (产物 models/train/<game>/)
python train-cli/train.py --game stochastic_gomoku --solver hybrid --preset quick  # 快速演示校准 (quick 预设: 0.2 缩放)
python train-cli/train.py --game mahjong_guangdong --solver qmix,happo,maac  # MARL 训练
python -m train_cli --game texas_holdem --solver hybrid         # 等价桥接入口
python -m layer4_interface.frontend.vision.server            # 视觉识别应用 (8766, 独立服务)
python -m layer4_interface.frontend.platform.server          # 平台前端服务 (8770, 需先 npm run build; --learning-interval N 开启在线学习 auto-apply)
cd platform-frontend && npm install && npm run build         # 构建平台前端
cd platform-frontend && npm run dev                          # 平台前端开发模式 (5173, /api 代理到 8770)
python scripts/_gen_mahjong.py                               # 重新生成 rules/mahjong.json (改 scripts/_gen_mahjong.py 后)
python scripts/_gen_werewolf.py                               # 重新生成 rules/werewolf.json
python scripts/_gen_undercover.py                             # 重新生成 rules/undercover.json
python scripts/_gen_uno.py                                    # 重新生成 rules/uno.json (六变体 × 2-10 人)
python scripts/sync_maac_models.py --dry-run                  # 预览 MAAC 共享产物同步
python scripts/train_maac_resume.py --game mahjong_guangdong --episodes 400  # 从已有 maac.pt 续训
python scripts/eval_mahjong.py --mode 1v3                     # 麻将独立对局评估
python -m layer4_interface.botzone.localai                    # Botzone Local AI 长轮询
python -m layer4_interface.botzone.server --host 0.0.0.0 --port 8788 --token TOKEN  # Botzone 远程接口
python -m layer4_interface.aifight.openai_compat --host 127.0.0.1 --port 8789 --token TOKEN --model gavis-local  # AIFight 桥
python scripts/build_botzone_zip.py --remote-url https://.../botzone/decide --remote-token TOKEN  # 构建 Botzone 上传包
```

## 层间契约速查

| 层 | 对外契约 | 说明 |
|----|---------|------|
| L1 → L2 | `TranslateResponse` + `EngineValidator.validate()` | 翻译产物必须过 schema + L2 冒烟校验（`smoke_validate(variants="all")`） |
| L2 → L3 | `GameEngine`（13 方法契约） | 求解器只经 GameEngine 消费游戏；`project_observation` 提供部分可观测投影，`get_info_set_key` 输出 sha256 64 字符信息集键 |
| L3 | `SolverBase`（select_action / train） | 所有求解器统一接口；可选依赖缺失时 class=None |
| L4 | `BaseBinding` Protocol、（内部）`SolverProvider` 注入 | 求解器经 `train-cli` 的 `default_provider` 装配（L4 不直接构造 L3） |