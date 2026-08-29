# Gavis 项目审查报告 — 功能 / Bug / 产品意见

> 审查对象：Gavis「陪你玩 Agent」自适应策略游戏 AI（四层水平集成架构）
> 审查日期：2026-09-21
> 审查方式：1 名主审 + 7 个并行子代理逐文件阅读源码（每条结论均带 `file:line` 证据），主审对关键 Bug 与既有审查修复项独立复核源码；含 961 用例收集统计、`ruff` 实跑、`npm run build` 实跑、生成器幂等性实跑、BayesSolver 实跑复现。
> 审查范围：`layer1_translator/` `layer2_engine/` `layer3_solvers/` `layer4_interface/` `rules/` `_gen_*.py` `train-cli/` `platform-frontend/src/` `tests/` `docs/`。
> 对比基线：`.docs/review/`（2026-08-22 L2/L3 逐行审查，17×P1 + 38×P2 + 48×P3 全部声明已修复）、`docs/product/product.md`（PRD v0.1）、`docs/product/AGENTS.md`（C1–C6 执行计划）。

---

## 0. 执行摘要（TL;DR）

**Gavis 的"陪你玩"愿景是真的、且端到端接通了，不是空壳 UI 套在一个未完成内核上。** 四层架构硬约束被遵守（L4 全仓零 `layer3_solvers` import —— 主审与两个子代理交叉确认）；Agent 对话/人格/自适应难度/偏好记忆/赛后复盘/在线学习六个陪伴模块全部落地并接入 `PlayManager`；既有审查（2026-08-22）的 17 个 P1 **确实已修复且带回归测试**（主审独立复核了其中关键的 P1-10/11/12 引擎三分叉、P1-13 MARL 发散、P2-25 可选依赖安全、两个 P0 阻断 Bug）。两个文档记录的 P0 阻断 Bug（麻将无法开局、在线学习发布不生效）**已修复且有 E2E 回归测试**。

**但存在 6 个新 P1 与一批 P2，集中在三条线上：**

1. **隐藏信息红线在"规则可见性声明"和"平台快照"两层被击穿**（既有审查和陪伴层子代理判"隐藏信息安全 HIGH"仅对 `agent/` 层成立）。新发现：德州弃牌局泄露 AI 牌型类别（P1）、UNO `env.handsSnapshot` 全手明文（P1）、麻将暗杠牌面与 `last_drawn` 实时泄露、狼人杀夜间行动目标公开。
2. **UNO 与狼人杀两条"旗舰新游戏"链路端到端不可用**：UNO = 规则泄密 + 前端无棋盘 + L1 翻译静默丢参 + 不在平台 9 游戏注册表；狼人杀 = BayesSolver 每次决策必崩（新 P1）+ 非真多变种 + 不在平台注册表 + 夜间目标规则泄密。
3. **工程信任基建失灵且无 CI 强制**：`ruff` 25 个 lint 错误 + 16 个待格式化文件（CLAUDE.md "ruff clean" 不实）；测试 961 例（声称 902，过期）；无 `pytest-timeout`，全套件超 600s；文档计数过期（平台 9 非 11、train-cli 17、测试 961）；所有 AGENTS.md "验收 ruff/pytest 全绿"均为自证且当前为假。

**产品意见（详见 §5）**：PRD §8 五条差距中 #1/#2/#3/#4 已 CLOSED，#5（狼人杀/视觉依赖外部服务）OPEN。最大产品风险：① 默认无 LLM 时陪伴"像念台词"、差异化体验欠交付；② 狼人杀/视觉仍依赖外部服务、无引导式首跑；③ 工程信任基建失灵、回归静默入库。陪伴评估（好棋/失误/提示/复盘）**只对网格类游戏有效**，对作为卖点的牌/麻将/狼人杀/UNO/卧底 5 类游戏全部失效或空转——这是与"陪你玩"承诺最实质的功能落差。

### 0.1 修复进展（2026-09 修复轮，全部带回归测试）

| # | 修复 | 证据 |
|---|------|------|
| P1-1 ✅ | BayesSolver `TypeError`：`_ensure_tracker()` 删除未用的 `obs` 形参 | `bayes_solver.py`；`test_bayes_werewolf.py` +2（无参可调、`select_action` 实跑 p0 回合） |
| P1-2 ✅ | 德州弃牌泄露 AI 牌型：`_hand_name(pid, gate)` 拆分，AI 侧以 `revealed` 为门 | `games.py` + `families/poker.py`；`test_platform_session.py` 弃牌局 `ai_hand_name is None` 回归 |
| P1-3 ✅ | UNO `handsSnapshot` 泄露：visibility 加 `env.handsSnapshot filter:false` + 换手/轮转后清空 | `_gen_uno.py` 重生成 `rules/uno.json`；`test_uno.py` +1（7/0 换手后原始 state 与观察均无泄露） |
| P1-4+P1-6 ✅ | **UNO 完整接入平台**：6 变体进注册表（平台 9→15 游戏）、族映射、前端 `UnoTable` + `UnoSnapshot` + uno 族分发、npm build 通过、全 6 变体 E2E 可玩到终局 | `games.py`（6 个 GameSpec）、`session.py`、`UnoTable.tsx`、`familyBoards.tsx`、`types.ts`；`test_platform_session.py::TestUno` 6 例 |
| P1-5 ✅ | L1 变体翻译崩溃：`_parse_rules` try/except + uno/mahjong 声明式 `variants` 参数分支 | `variant_translator.py`/`template_translator.py`；`test_layer1_translator.py` +4 |
| P2-18 ✅ | 五子棋聊天坐标：`spec.board_size` 优先（字典仅兜底、值改 9） | `chat.py`；`test_chat.py` +2（(2,3)→11 格、第 10 行澄清、天元 40） |
| P2-19 ✅ | `do_GET` 双重响应：except 路径只发 500 JSON，dist 检查只在 else 分支 | `server.py`（并行修复轮落地） |
| P2-22 ✅ | L1 温度非确定：`RULE_LLM_TEMPERATURE=0.0`，两个翻译器默认客户端固定 | `local_client.py`；温度捕获回归测试 |
| P2-23 ✅ | 传输/冷启动不重试：`complete_with_retry` 立即重试一次再兜底 | `local_client.py`；`test_layer1_translator.py` +2（首败重试成功 / 持久失败回退） |

**修复轮新发现（UNO 编译↔解释器分叉的现实样本）**：UNO 引擎在 `draw_result` 等阶段编译快路径与解释器分叉（首步即 compiled=1 vs interp=2 合法动作），牌堆耗尽后编译路径返回 0 合法动作导致对局卡死——P2-6~10 分叉组的已证实危害。平台 UNO 引擎固定 `allow_codegen=False`（`games.py` 带说明 docstring + `TestUno::test_engine_uses_interpreter_path` 守卫）。

**修复轮验证**：L1 162 通过（含新增 44 例变体翻译套件）/ chat 23 / bayes 10 / UNO 引擎 37 / 平台 session+server 55 / train-cli 23 / 前端 `npm run build` exit 0 / 修改文件 `ruff check` + `format` clean。

---

## 1. 审查方法

1. **主审一手复核**：主审直接阅读并验证了下列关键路径——引擎三分叉（`rules_compiler.py:239-260` / `engine.py:261-267,278-293`）、MARL `next_mask` 修复（`marl/env.py:206-227`）、可选依赖守卫（`layer3_solvers/__init__.py:15-35`）、两个 P0 修复（`games.py:375-379` / `train-cli/games.py:513`）、德州弃牌泄露（`games.py:320-322,351`）、五子棋聊天坐标 Bug（`chat.py:50` vs `rules/stochastic_gomoku.json:99`）、陪伴六模块（`agent/`、`difficulty/adaptive.py`、`profile/store.py`、`review/analyzer.py`、`online_learning/manager.py,recorder.py`）、`chat.py` 工具调用管线、`session.py` 集成接线、`--preset quick`（`train.py:555-561`）、L1→L2 校验闸门（`engine_validator.py` + `smoke_validator.py`）、平台 9 游戏注册表（实跑 `--list` 与 `grep`）。对 L4→L3 零 import 做了全仓 `grep` 复核。
2. **7 路并行子代理逐文件审查**：L1 翻译层 / L2 引擎 / L3 求解器 / L4 陪伴模块 / L4 平台后端 / 规则+前端 / 测试+文档+产品。每代理产出带 `file:line` 的功能、Bug、产品意见报告。子代理结论与主审一手复核**高度一致**，跨代理间无矛盾。
3. **实跑验证**：`python train-cli/train.py --list`（17 游戏）、`python -m pytest --collect-only`（961 例）、`ruff check/format`（25+16）、`npm run build`（exit 0）、4 个生成器幂等（SHA-256 字节一致）、BayesSolver 复现（p0 行动 TypeError）。
4. **对比基线**：逐项核对 `.docs/review/issues.md` 的 17×P1 / 38×P2 / 48×P3 是否在当前源码中已修复。

---

## 2. 功能审查

### 2.1 架构与契约（健康）

- **四层硬约束被遵守**：全仓 `grep` 确认 `layer4_interface/` 内**零** `layer3_solvers` import（所有 `layer3_solvers` 字样都是"声明其不存在"的 docstring）。L4→L3 经 `SolverProvider` 协议注入（`solver_provider.py:40-57`），实现与装配在 `train-cli/games.py`（唯一允许 import L3 的装配点）。L4→L2 import（`GameEngine`/`ActionInstance`/`LLMClient`）合法。
- **`SolverBase` 契约**：`select_action(state) -> ActionInstance | None`。主审与 L3 子代理确认：MCTS/CFR/Hybrid/MahjongHeuristic/OllamaSolver 对合法 player 状态从不返回 None（有 uniform/random 兜底）；**PPO 对非网格游戏返回 None（契约违例）**；**BayesSolver 每次必崩（新 P1）**。
- **L1→L2 单一授权通道**：`engine_validator.validate()` = schema 校验 → 早退 → L2 `smoke_validate`（启动引擎 + 探针一次初始转移）。结构正确。但 L1 在 4 处直接 import 了 L2 的具体类 `LLMClient`（`llm_translator.py:8`、`variant_translator.py:34`、`engine_validator.py:14`、`local_client.py:16`），"单一授权通道"不变量已松动（具体类依赖，非 Protocol）。

### 2.2 Layer 1 翻译层（已实现，非"预留"）

- **13 个模块**，README 称"(预留)"过期。管线**结构健全**：确定性模板 → LLM 编排（含修复循环）→ schema 校验 → L2 冒烟校验；任何路径失败返回 `rules_json={}` + 中文原因，**绝不返回未校验产物**。
- 既有审查的 L1 相关项 4 项中 **3 项已修复**：JSON 提取改用 `JSONDecoder.raw_decode` 花括号扫描（非贪心正则）、`max_tokens=8192` + 30s 超时 + 512K 回复上限、`rule_text` 经 `sanitize_rule_text` 清洗 + 注入防御系统提示。**1 项部分修复**：修复循环只覆盖"校验失败"，**传输失败/冷启动仍不重试**直接回退。
- 确定性模板覆盖 5 游戏（moon_chess/gomoku/texas/mahjong/werewolf），UNO 解析器 `_parse_uno` 存在但**应用层无 uno 分支**（见 P1-5）。

### 2.3 Layer 2 引擎（核心正确，编译快路径有静默分叉）

- **解释器路径正确且健壮**：既有审查的 P3 修复（触发器级联、None 守卫、trimByKey 非方阵、笛卡尔积上限、视图缓存）均在场。`sample_chance` 空表 → ValueError（非 IndexError 崩溃）+ 概率归一化（主审复核 `engine.py:278-293`）。
- **三个引擎 P1 全部修复**（主审复核）：P1-10 switch 编译为首匹配 if/elif/elif（`rules_compiler.py:239-260`，注释说明旧 `or` 链 falsy 穿透已修）；P1-11 chance 多模板解释器首匹配（`engine.py:261-267`）与编译器 if/elif 对齐（`rules_compiler.py:690`）；P1-12 sample_chance 见上。
- **`project_observation` 可见性修复**（P2-12）：`visibility.env` 逐字段按 viewer 过滤（`engine.py:322-393`），隐藏字段实体 strip `value`。狼人杀/麻将测试覆盖。
- **编译快路径残留 5 个非崩溃分叉**（P2，见 §3.2）：grid `$i`、query 原始数组 vs 派生字段、legal_actions 缺节点类型检查、append `{"count":N}` 启发式误判、add/sub/mul 吞异常。这些**不崩溃**故绕过"编译失败即回退解释器"的安全网，仅靠**仍很薄的 3 步探针**（`rules_compiler.py:732-747`）兜底。`smoke_validator.py` 更薄（只探针 1 次初始转移，不测 `sample_chance`/`project_observation`/`get_utility`/`is_terminal`）。
- **关键判断**：对**内置游戏**用编译路径（暴露于上述静默分叉）；对**自定义游戏**用 `allow_codegen=False` 纯解释器路径（安全）。所以"创建游戏"功能反而比内置游戏更安全——但 smoke 校验薄意味着自定义游戏第 4 步后的潜在 Bug 不会在创建时被发现。

### 2.4 Layer 3 求解器（核心算法已修复，一个导出求解器是坏的）

| 求解器 | 状态 | 说明 |
|--------|------|------|
| MCTS | ✅ demo-ready | 完美信息 + chance，stdlib+numpy |
| CFR | ✅ demo-ready | External-Sampling MC-CFR+，uniform 兜底；**但 save/load 是 no-op（P2-21），训练成果不持久化**；评估只测首玩家 vs uniform（P2-20） |
| Hybrid | ✅ demo-ready（主路径） | MCTS+CFR+PSRO+经验对手模型，PIMC；PSRO import 守卫 |
| MahjongHeuristic | ✅ demo-ready | 无状态，`legal[0]` 兜底 |
| OllamaSolver | ✅ demo-ready | JSON 契约 LLM，传输/解析失败随机兜底；女巫崩溃/解毒歧义已修（`ollama_solver.py:194-232`） |
| PPO | ⚠️ 仅网格 | 硬编码 `place:r,c` 键，非网格游戏全零 mask 返回 None（契约违例） |
| PSRO | ⚠️ 仅 3×3 月亮棋 | `_validate_board` 对非 9 格硬报错（正确的拒绝，但意味着 PSRO 非通用求解器）；exploitability 用 `-mean` 而非 `-min`（信号弱，P1-3 部分） |
| QMix/HAPPO/MAAC | ⚠️ 实验 | 发散级 Bug 已修（`env.py:206-227` 卷机会节点，主审复核；MAAC detach+REINFORCE，`maac.py:402-404`）；但"next_mask 属对手回合"语义未根本解决，重定义为 CTDE 近似 |
| **BayesSolver** | ❌ **坏的** | `werewolf/bayes_solver.py:68` 调 `_ensure_tracker()` 缺 `obs` 实参 → 每次绑定玩家决策必 TypeError（已实跑复现，见 P1-1） |
| auto_selector | ❌ 桩 | "future work"，非 SolverBase |

- **可选依赖安全（P2-25）已修**（主审复核 `__init__.py:15-35`）：PSRO/PPO/MARL 包在 `try/except ImportError`，最小安装（仅 numpy）可 import。
- **可复现性**：PSRO/HAPPO/MAAC/QMix/MCTS/PPO 均种 RNG；CFR `_play_vs_random` 已种（P2-19 修）。**CFR save/load no-op 是唯一复现性硬伤**。

### 2.5 Layer 4 陪伴模块（全部落地并接线）

`agent/`（7 文件）、`difficulty/`、`profile/`、`review/`、`online_learning/`（6 文件）全部实现并经 `session.py:20-27` 接入 `PlayManager`。

- **人格/场景**：4 性格（gentle/teacher/banter/cold）× 9 场景，`_assert_coverage()` 导入期自检全覆盖（`persona.py:107-115`）。PRD §4.2.1 要求 ≥4，**满足且超出 P0 计划的 2 个**。
- **`Skills.build` 红线入口**（`skills.py:46-64`）：只调 `project_observation` + `get_legal_actions` + `evaluate` + `assert_no_hidden`。结构正确。
- **`DialogueEngine.reply` 管线**（主审复核 `dialogue_engine.py:80-97`）：mute → LLM 成文（失败回退 `fallback_lines`）→ `_clean`（长度≤100 + 控制字符）→ `hidden_guard.scan` → 去重（5 分钟窗，键 `(scenario, persona.key, state_hash)`）。与冻结契约一致。
- **`AdaptiveController`**（主审复核 `adaptive.py:141-174`）：胜率<0.40 降一档、>0.60 升一档、夹带、锁定返回原档；麻将启发式无预算旋钮如实返回"展示档位"。PACING ≤1s/≤5s/≤15s 符合 PRD §4.3.2。
- **在线学习端到端**（主审复核 `manager.py` + `recorder.py`）：捕获（`RecordingHandle` 包装求解器，记录人类每决策的 info_key+canonical_key）→ 持久化（`store.append_match` 原子写）→ 聚合（`build_empirical_table` 只计人类决策）→ 门禁（hybrid-vs-hybrid 固定种子、换座、容差 0.03）→ 发布（`OnlineModelStore` 版本化 + 回滚）→ 自动应用（守护线程，幂等）→ 下次开局注入（`train-cli/games.py:507-513` `empirical_table` + `opponent_model="empirical"`）。**#2 P0 修复确认有效**。
- **隐藏信息守卫**（agent 层）：`assert_no_hidden` 黑名单键扫描观测（`hidden_guard.py:109-121`）；`scan` 后置令牌扫描按游戏分派。主审确认 `infer_game_id` 的视图名键与各规则 JSON `visibility` 声明一致，后置扫描未被静默禁用。**但 `assert_no_hidden` 只扫 `ctx.observation`，不扫 `ctx.evaluation`（见 P2-3）。**

### 2.6 Layer 4 平台后端（API 完整，快照层有隐藏信息泄露）

- **HTTP API**：GET/POST/PUT/DELETE 路由齐全（`/api/games`、`/api/chat`、`/api/match/{start,move,state,hint}`、`/api/agent/say`、`/api/profile[ /clear]`、`/api/review/:id`、`/api/history`、`/api/benchmark[/start,/status]`、`/api/learning/{status,apply,config}`、`/api/custom/games`、`/api/rules/translate`）。错误信封一致，内部异常文本**不**泄露给客户端。
- **会话可靠性**：每会话 `threading.Lock` 保护 `move()`；刷新恢复经 `/api/match/active` + `/match/state` 可用；**会话纯内存，服务器重启丢失进行中对局**（本地工具可接受，对外暴露前需持久化）。
- **chat-first 管线**（主审复核 `chat.py`）：OpenAI 风格工具调用 + 确定性正则回退；`_legal_context` 只从快照取 `legal/board/turn`（投影后，不碰原始隐藏数组）；客户端 `history` 清洗（仅 user/assistant、24 条/6000 字、system 每轮后端现构）；`make_move` 经 `parse_human_action` 预校验。LLM 调用同步在请求线程，但 `ThreadingHTTPServer` 使其只阻塞本线程（缓解，非全局卡死）。
- **自定义游戏流**（`custom_games.py`）：翻译 → 校验 → 族识别 → `build_spec` → 原子持久化 + id 白名单。结构健全。

### 2.7 前端（构建健康，缺 UNO 棋盘）

- `npm run build` **通过**（exit 0，81 模块，269.74KB JS / 86.99KB gzip）。`tsconfig` strict + noUnusedLocals + noUnusedParameters + noFallthroughCasesInSwitch；**全 `src` 零 `any`、零 `@ts-ignore`**（grep 确认）。
- PRD §5 页面齐全：Home/Lobby/Create/Battle/Benchmark/Learning/History/Profile/Settings/Review(+Replay 别名)。`ChatPage`+`useChatRuntime` 是"对话即一切"主界面，棋盘点击走 LLM 旁路快路径直发 `/match/move`。
- 陪伴感：人格开场、头像 mood（happy/thinking/sorry/neutral）、复盘时间线（turning_point/blunder/winning_move 标签 + 导出）已交付。
- 刷新恢复（`?game=` + `/match/state` + `localStorage` 活跃会话）健壮。

### 2.8 规则与生成器（确定性优秀，可见性声明有缺口）

- **4 个生成器幂等**（SHA-256 字节一致，无时间戳/RNG/环境泄漏）。
- 麻将/卧底/UNO 的 `variants` 声明式正确（变种/人数/配比在单个 JSON 声明，引擎纯数据解析）。
- **狼人杀 `variants` 是桩**：9 人/3 狼烤进顶层 `constants.role_pool`，`variants={default:{}}` 无 `player_count`；6/12 人需重新生成 JSON 而非引擎选择（与 CLAUDE.md "声明在 variants"矛盾，见 P2-12）。
- `chance`+`effectMap` 结算、`text` 参数预制能力（狼人杀/卧底发言）声明正确。

---

## 3. Bug 审查

> 严重度：**P0**=致命必崩 ｜ **P1**=严重逻辑错/契约违例/红线击穿 ｜ **P2**=一般问题/边界/泄露 ｜ **P3**=代码质量/文档漂移。
> 既有审查（2026-08-22）17×P1 **全部已在源码中确认修复**（见 §3.0），下表只列**当前仍存在**的问题与新发现。

### 3.0 既有审查 P1 修复确认（17/17 已修，主审与子代理交叉复核）

| 既有 # | 位置 | 状态 | 修复证据 |
|--------|------|------|---------|
| P1-1/2/3 | psro | ✅修/✅修/⚠部分 | Q-update 移到对手动作后（`tabular_q.py:86-110`）；mask 来自传入 state（`solver.py:81-88`）；exploitability 去掉 `max(v,0)` 但用 `-mean` 非 `-min`（信号弱） |
| P1-4~9 | 适配器/生成器 | ✅修 | 麻将 15 张判和、狼人杀轮转（规则层推进 env.turn + 女巫存活门）、witch_self_save 死配置等 |
| P1-10/11/12 | 引擎 | ✅修（主审复核） | switch if/elif（`rules_compiler.py:239-260`）；chance 首匹配（`engine.py:261-267`/`rules_compiler.py:690`）；sample_chance 归一化+ValueError（`engine.py:278-293`） |
| P1-13 | MARL next_mask | ✅修发散/⚠语义 | 卷机会节点使 next_mask 非零（`env.py:206-227`，主审复核），QMix 不再 -1e9 发散；"对手回合 mask"重定义为 CTDE 近似 |
| P1-14 | MAAC | ✅修 | `q_online.detach()` + REINFORCE（`maac.py:402-404`） |
| P1-15/16/17 | LLM/规则 | ✅修 | 女巫 `_target_of` 类型守卫 + `except (TimeoutError,OSError)`（`ollama_solver.py:222-232`）；do_kill 单守卫分支+女巫存活（`rules/werewolf.json:1999-2355`）；heal/poison 双字段格式 |

### 3.1 P1（严重，6 项新发现）

| # | 位置 | 问题 | 根因 |
|---|------|------|------|
| P1-1 | `layer3_solvers/werewolf/bayes_solver.py:68` vs `:104` | **BayesSolver 每次绑定玩家决策必 TypeError**：`select_action` 调 `self._ensure_tracker()` 无实参，但签名要求 `obs`。已实跑复现（p0 行动 step 9 → TypeError）。狼人杀唯一非 LLM 求解器（顶层导出符号）不可用。两个测试都绕开了 `select_action` 路径故未暴露。 | 调用点未随签名更新；`obs` 实际在方法体内未使用。修复=删除 `obs` 形参或传入 `flat`。 |
| P1-2 | `layer4_interface/frontend/platform/games.py:320-322,351`（+ `families/poker.py:261-267,296`） | **德州弃牌局泄露 AI 牌型类别**：`_hand_name(pid)` 仅以 `over` 为门，而 `ai_hole` 以更严的 `revealed = over and last_action=="showdown"` 为门。弃牌局 `over=True,revealed=False` → `ai_hole=[]`（正确）但 `ai_hand_name` 仍据 AI 隐藏底牌+公共牌计算并返回。翻前弃牌→暴露 AI 是否口袋对；翻后弃牌→暴露 AI 成牌类别。**击穿 §A.3 reveal-gate 红线**（代码自身文档把"仅终局+revealed"作为门）。主审一手复核确认。 | `_hand_name` 复用了 `over` 而非 `revealed`。修复=AI 分支用 `revealed` 作门。 |
| P1-3 | `_gen_uno.py:565-608,1150,297-309`（输出 `rules/uno.json`） | **UNO 全手明文泄露**：`seven_zero` 变体出 `0` 时 `env.handsSnapshot = [hand_of(p) for p in all]`（全手转储），出 `7` 时为两名换牌者手牌；**永不清理**；UNO `visibility` 无 `env` 子段 → 按契约 `handsSnapshot` 对**任何观察者公开**。 | scratch env 字段未标隐藏/未清理。修复=uno visibility 加 `"env":{"handsSnapshot":{"filter":{"const":false}}}` 且/或在轮转后清理。 |
| P1-4 | `platform-frontend/src/components/boards/familyBoards.tsx:16-46` | **前端无 UNO 棋盘/无 uno 族**：`FAMILY_BOARDS` 仅 grid/poker/mahjong/social；`src` 内零 "uno" 引用。UNO 会话在 `BattlePage.tsx:244-267` 落到 `<GomokuBoard>`（崩坏）。`types.ts:153` Snapshot 联合无 `UnoSnapshot`。规则+引擎+生成器+smoke 都过，但 SPA 无法渲染。 | 规则与引擎已交付但缺匹配的前端族。 |
| P1-5 | `layer1_translator/variant_translator.py:407`（+ `template_translator.py:107-122`, `variant_translator.py:253-269`） | **变体 LLM 路径 `_parse_rules` 未 try/except**：LLM 返回非 JSON/散文时抛 `LLMTranslatorError` 直穿 `translate`，使变体翻译崩溃而非回退确定性路径（违反文档契约）。`llm_translator.py:70-75` 正确包裹了同一调用。零 `VariantTranslator` 测试故未暴露。**连带**：`_apply_parameters` 无 `uno` 分支 → "UNO 4人 叠加" 静默返回 classic 默认（比狼人杀更糟，后者至少警告）。 | 调用点遗漏守卫；UNO 应用分支缺失。 |
| P1-6 | （产品级，跨层） | **UNO 端到端不可用**：P1-3（规则泄密）+ P1-4（无前端棋盘）+ P1-5（L1 静默丢参）+ 不在平台 9 游戏注册表（见 §3.3）。一个被 commit `ab5e3fb` 宣称"完整接入"的旗舰新游戏在主 UI 上无法开局且会泄密。 | 多层独立缺陷叠加。 |

### 3.2 P2（一般/边界/泄露，按主题归并）

**A. 隐藏信息泄露（红线，跨"规则可见性"与"平台快照"两层）**

| # | 位置 | 问题 |
|---|------|------|
| P2-1 | `games.py:438-439,465` + `families/mahjong.py:174-175,205`；`rules/mahjong.json:2655-2675` | **麻将暗杠牌面实时泄露**：`_mahjong_snapshot` 绕过 `project_observation` 直读 `melds_pN`，人类实时获得 AI 暗杠 `concealed_gang.tiles`。真麻将对他家暗杠牌面保密。规则 `visibility` 只隐 `hand_view_pN.id`，快照从不查它。"字面合规但精神泄露"。 |
| P2-2 | `_gen_mahjong.py:674,693,1259-1281` | **麻将 `env.last_drawn`（刚摸牌）公开**：每次摸牌置 `last_drawn` 且永不清理；visibility 只过滤 `win_hand` → 对所有观察者公开。他家摸什么是有意义的隐藏信息（手牌推算）。前端只在人类自己回合渲染故今日不用户可见，但规则契约泄露。 |
| P2-3 | `_gen_werewolf.py:893-931` | **狼人杀夜间行动目标公开**：`visibility.env` 只过滤 `seerResult`；`nightKill`/`poisonTarget`/`guardTarget`/`witchSavedTarget`/`hunterShoot`/`lynched` 均为公开 env 字段。狼刀目标在死亡宣布前应保密；被守卫/女巫救下的刀应不暴露被刀者。今日 `SocialSnapshot` 不渲染这些故无用户可见泄露，但规则契约松于游戏语义。 |
| P2-4 | `agent/evaluation.py:53` | **陪伴层唯一的直接 `state["_arrays"]` 读取**：`_board_heuristic` 直读 `state["_arrays"]["board"]`，绕过"输入只许来自 project_observation"红线策略。今日只读公开 `board` 字段、对隐藏信息游戏返回 0.0 故无害，但**未受 `assert_no_hidden` 守卫**（该守卫只扫 `ctx.observation`，不扫 `ctx.evaluation`）。 |
| P2-5 | `types.ts:110,73`；`MahjongTable.tsx:75`；`PokerTable.tsx:98` | **隐藏数组在客户端状态**：麻将快照带 AI 全手 `ai_hand`、德州带 `ai_hole`，即使渲染层正确以 `over`/`revealed` 为门不显示，但数据在 React 状态/网络响应里，好奇用户可经 DevTools/网络检查读到。后端应在发送前按 viewer 清零。 |

**B. 引擎编译↔解释器静默分叉（项目最大正确性风险面，5 项不崩溃故绕过安全网）**

| # | 位置 | 问题 |
|---|------|------|
| P2-6 | `rules_compiler.py:422,472` vs `state_graph.py:227,282` | grid 视图 `$i`：解释器实体无 `_i` 键→`0`；编译器 binder 映射 `i→_i`（数组下标）。分叉。 |
| P2-7 | `rules_compiler.py:376-398` vs `engine.py:559-574` | query 谓词：编译快路径扫**原始**数组值（`$node.occupant`→原始 int）；解释器过滤**物化**实体（派生 `occupant`="black"/"white"）。两者选不同集合。 |
| P2-8 | `rules_compiler.py:560-564,645-646` | 编译 `legal_actions` 只发 `phase` 守卫，**无节点类型检查**；解释器对非 player 节点返回 `[]`。chance 节点 phase 与某动作模板重叠时编译返回动作、解释器返回 `[]`。 |
| P2-9 | `engine.py:688-706`；`rules_compiler.py:207-209` | (a) append 启发式把单键数据字典 `{"count":3}` 误判为表达式→求值成 0 追加（静默数据损坏，apply_action 纯解释器故非分叉而是单路径 Bug）；(b) 编译器对字面 list 逐元素编译但解释器原样返回 list——注释"逐元素求值"事实错误并掩盖分叉。 |
| P2-10 | `expr_eval.py:231-244` vs `737-744` | 编译 add/sub/mul 吞 `TypeError` 返 None；解释器直接抛。非崩溃→探针不命中则生产静默返 None。 |
| P2-11 | `engine.py:447` | **`get_utility` `float(None)` 终局崩溃路径**：value 表达式返 None 时 `float(None)` 抛 TypeError，无守卫、无测试。 |
| P2-12 | `rules_compiler.py:732-747`（探针）/ `smoke_validator.py:25-57` | **探针/冒烟校验覆盖极薄**：编译探针仍只 3 步沿首动作；smoke 只探针 1 次初始转移，不测 `sample_chance`/`project_observation`/`get_utility`/`is_terminal`。这是让上面 5 个静默分叉"静默入库"的缺口。L1 自定义游戏创建时第 4 步后 Bug 不会被发现。 |

**C. 求解器缺口**

| # | 位置 | 问题 |
|---|------|------|
| P2-13 | `cfr/solver.py`（无 save/load 覆盖） | **CFR save/load 是继承 no-op**：训练的 `info_sets` 退出即失，仅经 Hybrid 的 JSON 表间接持久化。复现性硬伤。 |
| P2-14 | `nash_solver.py:16-78`；`meta_game.py:169` | **零和假设无守卫**：无条件建反对称 `M[j,i]=-payoff`；一般和或 >2 人游戏静默产非均衡混合。PSRO 只测 p1 效用且假设 p2=-p1。 |
| P2-15 | `action_space.py:40-51` | **麻将动作槽基址硬编码 34**：`_MAHJONG_BLOCKS` 步长 0/34/68...，加花牌（tile_ids≠34）即槽位错位（弃牌 0-41 与 gang_concealed 基址 34 碰撞）。标准 34 牌变种无事。 |
| P2-16 | `ppo/solver.py:415,433` | **PPO 硬编码 `place:r,c` 网格键**：非网格游戏全零 mask→`select_action` 返 None（SolverBase 契约违例）；`_action_from_index` 抛 ValueError 而非返 None。 |
| P2-17 | `marl/qmix.py:350-353,361` | QMix 对 N 个 per-agent Q-net 全前向再用 acting mask 置零非行动者——浪费 (N-1)/N 前向算力（正确性无碍）。 |

**D. 平台/快照/聊天缺口**

| # | 位置 | 问题 |
|---|------|------|
| P2-18 | `chat.py:50-53` vs `games.py:554`/`rules/stochastic_gomoku.json:99` | **五子棋聊天回退算错格**：`GRID_BOARD_LEN["stochastic_gomoku"]=15` 但棋盘 9×9。"下第2行第3列"→cell 17（应 11）；10-15 行被正则接受再被判非法。主审一手复核确认。修复=改 9 或从 `spec.board_size` 派生。 |
| P2-19 | `server.py:141-154` | **`do_GET` 异常路径发第二个 HTTP 响应**：500 JSON 写完后又跑 SPA-serve 逻辑（`send_json(503)` 或 `super().do_GET()`），同一 socket 第二次响应，破坏流。dist 检查块被复制进 except。 |
| P2-20 | `agent/skills.py:97-128` | **`suggest_hint` specific/demo 是桩**：选合法动作中位（按 canonical_key 排序取中位），非真实求解器走法；真实求解走法延后到会话接线（docstring 自承）。 |
| P2-21 | `agent/skills.py:74-95` + `evaluation.py:53-55` + `review/analyzer.py:181-202` | **陪伴评估/复盘只对网格游戏有效**：`_board_heuristic`/`_board_score` 只对方阵棋盘工作；德州/麻将/狼人杀/UNO/卧底非终局一律返回 0.0"胶着"→`detect_good_move`/`detect_blunder` 永不触发、`turning_point` 退化为"取首步"、`blunder` 永不检出、`improvement` 退化为通用"稳扎稳打"。**对 5 个卖点游戏陪伴智能空转。** |

**E. L1 翻译缺口**

| # | 位置 | 问题 |
|---|------|------|
| P2-22 | `llm_translator.py:62`；`local_client.py:28`；`llm.py:63,187` | **LLM 规则翻译温度 0.2 非确定性**：同一规则文本跨次产出不同 `rules.json`，损害"它就是能用"的可复现性；`RuleLLMClient` 协议无 `temperature` 形参，L1 无法强制 0。 |
| P2-23 | `llm_translator.py:62-69`；`variant_translator.py:399-406` | **传输/冷启动失败不重试**：空 `raw` 与异常都立即退出循环，只"校验失败"进修复循环。冷 Ollama/网络抖动→即时回退。 |
| P2-24 | `prompt_builder.py:85-96`；`external_frontend_reader.py:148-155` | `external_frontend.rule_text` 绕过 `sanitize_rule_text`（仅 `.strip()`）→控制字符/超长载荷进 prompt 未清洗。 |
| P2-25 | `schema_validator.py`（整文件） | **不校验 v5.2 `variants` 段**：malformed variants 过 schema，只在引擎冒烟（更薄）才可能暴露。 |

**F. 规则与前端缺口**

| # | 位置 | 问题 |
|---|------|------|
| P2-26 | `_gen_mahjong.py:1196-1240` | **mahjong.json 735KB 膨胀**：`functions` 占 580KB，`is_win_hand` 单独 512KB——`_standard_win`/`_pair_pool`/`_meld_pool`/`_cover_ok`/`_cover_prefix` 子树被内联 4-5 次（~95KB 字节相同重复）。维护性（改一处需重审 512KB）+ 每次 `GameEngine` 构造的解析成本。可提取命名 alias + `CALL` 缩 ~30×。 |
| P2-27 | `_gen_werewolf.py:1088-1113,1117` | **狼人杀非真多变种**：`variants` 是桩，9p/3w 在 `constants.role_pool`，一 JSON 一配比；6/12 人需重新生成。与麻将/UNO/卧底"一 JSON 多变种"模式不一致，与 CLAUDE.md 措辞矛盾。 |
| P2-28 | `global.css:62-71` | **侧栏不响应式**：210px 固定无移动抽屉；手机上永久占 210px。 |
| P2-29 | `LobbyPage.tsx:50`；`ReviewPage.tsx:127-138`；`InlineBoard.tsx:88`；`Layout.tsx:21` | **可访问性缺口**：可点 `<div>`/`<span>` 无 `tabIndex`/`onKeyDown`/`role`，键盘不可达；`<nav>` 无 `aria-label`；无 `:focus-visible` 样式。 |
| P2-30 | `useChatRuntime.ts:31-48` vs `types.ts:214-226,250-260` | **类型漂移**：本地重声明 `BenchmarkJob`（`status:string` vs 字面量联合）、`LearningItem`（丢字段），后端 shape 变更不会被 TS 捕获。 |
| P2-31 | `online_learning/manager.py:23` | 顶层 `from layer4_interface.frontend.platform.games import GAMES` 把学习管理器与平台前端注册表耦合（`adaptive.py:70` 用 lru_cache 懒加载）。 |

### 3.3 注册表与可玩性落差（产品级）

- **平台注册表 9 游戏**（moon_chess、stochastic_gomoku、texas_holdem + 6 麻将变种），**不含**狼人杀/卧底/6 UNO 变种——尽管这 8 个有完整规则 JSON + 生成器 + 引擎测试 + train-cli 登记（**train-cli 17 游戏**）。主审与 L4-platform 子代理交叉确认（`games.py` grep + `--list`）。
- CLAUDE.md 称平台"11 游戏"、architecture.md 称"11 游戏"——**均过期**（实际 9）。
- 后果：PRD §1.2 "游戏多样性：棋、牌、麻将、狼人杀"中的"狼人杀"与"牌"（UNO）在主 UI 不可玩；train-cli 能训练却平台不能对弈。
- **➡ 修复轮状态**：UNO 落差已闭合（平台 9→15，含 6 UNO 变体全链路可玩，见 §0.1）；狼人杀/卧底仍不在平台注册表（social 族有棋盘与求解器，缺 GameSpec 注册——下一修复轮候选）。文档计数（CLAUDE.md/architecture.md/README）需随之刷新。

### 3.4 P3（代码质量/文档漂移，择要）

- **文档漂移**：README 严重过期（L57 "layer1 (预留)" 与 CLAUDE.md "已实现"直接矛盾；L39 "7 游戏"实际 17；L16/77/82-89 指向不存在的 `docs/merge/`——六篇合并文档实为 `archive/original_unmerged_20260728/mergedocs/`；结构树缺 `agent/`/`difficulty/`/`profile/`/`review/`）。CLAUDE.md L17 "902 cases" 实际 961；L93 `docs/merge/` 同样过期。architecture.md "11 游戏"（平台 9/train 17）/"870+ 测试"（961）/版本 v0.4 vs CLAUDE.md 表 v0.2 不一致。`security-notes.md:15` 仍引 `interfaces/api_key.py`（已迁 `core/api_key.py`）。**用户文档端口漂移已解决**（全指向 8770，标注 play_* 退役）。
- **`ruff` 不净**（主审实跑）：25 lint 错误（11×W292、7×F401、5×I001、2×F841，19 可自动修）+ 16 文件待 `ruff format`。CLAUDE.md "ruff format + ruff check" 声明不实。
- **无 CI 闸门**：无 `ruff` 预提交/CI、无 `pytest --timeout`、`pytest-timeout` 未装；全套件 >600s（慢 CFR，既有审查 P2-17 未解）。所有 AGENTS.md "验收 ruff/pytest 全绿"均自证且当前为假。回归静默入库。
- **`train_cli.py` 桥**：源码树下 `python -m train_cli` 可用，但真正 `pip install gavis` 后失效（`train-cli/` 非合法包名未装）；`pyproject` `[tool.pytest.ini_options]` 与 `pytest.ini` 重复（无害告警）。
- 其余 P3：CFR eval 用首玩家 vs uniform（P2-20）；MAAC 单样本 log_prob² 熵估计（非标准）；auto_selector 是桩；`social/` 双轨未接线死代码；`episodes-0` 无操作残留；单赢家门禁假设；`feedback_collector` 未被 apply 管线消费；UNO `hand_view_*` 与麻将 `infer_game_id` 前缀碰撞（窄模式故无害）；前端无 404 路由；`player_counts[0]` 默认可能为 4 人麻将选 2 人。

---

## 4. 既有审查 vs 现状（对照基线）

| 维度 | 既有审查（2026-08-22） | 现状（2026-09-21） |
|------|----------------------|-------------------|
| P0 | 0 | 0 |
| P1 | 17（声明全修） | **17 确认已修**（P1-3/13 标"部分"）+ **6 新 P1** |
| P2 | 38 | 既有大部分已修，**残留 ~31 项**（§3.2，含编译 5 分叉、CFR save/load、零和守卫、麻将 34 步长、隐藏信息两层泄露等） |
| P3 | 48 | 已处理，但**新增**：ruff 不净、文档计数过期、无 CI、`train_cli` 桥装后失效 |
| 新增陪伴/平台/前端/规则 | 未覆盖 | **新增 ~20 项**（陪伴评估网格局限、平台快照泄露、UNO/狼人杀链路、前端 a11y/响应式/类型漂移、规则可见性声明缺口、mahjong.json 膨胀） |

**净评估**：既有审查的 P1 全部真修且有回归测试（高质量修复）；但审查范围当时只到 L2/L3，**未覆盖 L4 平台快照层、规则 `visibility` 声明层、L1 翻译的 UNO/变体路径、前端**——本次审查补齐了这些，发现的新 P1/P2 集中在彼时盲区。

---

## 5. 产品意见

### 5.1 PRD §8 差距判定

| PRD §8 差距 | 判定 | 证据 |
|------------|------|------|
| #1 无 Agent 对话/人格模块（最大差距） | **CLOSED**（含保留） | `agent/` 7 文件 + `session.py:264` 接入 + 4 性格×9 场景 + 9 场景检测（`session.py:306,376-395`）+ `/api/agent/say`/`/api/chat`/`/api/match/hint` + 前端 ChatPanel/AgentAvatar/dist 已建。**保留**：LLM 半边可选——无 ollama/OpenAI 时 `DialogueEngine` 回退人格模板台词（`dialogue_engine.py:99-110`），陪伴感"像念台词"；C2 计划的 `agent/llm_client.py` 未建、改复用 `layer2_engine.core.llm.LLMClient`（功能等价/更优 DRY+tools，但未循 C2 字面契约）。 |
| #2 麻将开局 bug + 在线学习发布 bug | **CLOSED** | `_make_mahjong_solver(game_id)` + `opponent_model="empirical"` setdefault；回归测试 `test_platform_session.py:214-291` + `test_online_learning.py:616,644`（断言 `opponent_model=="empirical"` + E2E play→learn→publish→consume）。 |
| #3 默认训练管线过慢 | **CLOSED** | `--preset quick`（0.2 缩放 + 下限）已实现。 |
| #4 UI 基础大厅、无聊天/形象/复盘 | **CLOSED**（含保留） | ChatPanel/AgentAvatar/ReviewPage/SettingsPage/HomePage + chat/ 对话 UI 已建；`api/client.ts` 真接线。**保留**：`mock.ts` 被多个页面作空态兜底导入，ReviewPage 有 `mockMode`（端点不可用时显 MOCK_REVIEW）。 |
| #5 狼人杀/视觉依赖外部服务 | **OPEN** | 狼人杀求解器=OllamaSolver（需本地 ollama）；视觉=`qwen_vision.py`（外部 VLM API）。AGENTS.md C6.2 仅归档+文档标注，未做打包/引导首跑/优雅"安装 ollama"UX。PRD §6.4 "离线…除狼人杀 LLM 和视觉识别外"措辞仍准确（这两项被诚实除外），但差距本身未处理。 |

### 5.2 三大产品风险（子代理 + 主审综合）

1. **默认无 LLM 使陪伴"像念台词"，差异化欠交付。** `DialogueEngine` 仅在 `LLMClient.available()` 时调 LLM，否则每场景返回轮换人格兜底台词。PRD 场景 D 与 §4.2.2 各场景"被覆盖"但"不丰富"。无文档化/打包的 LLM 配置路径→多数用户永远见不到 LLM 驱动半边。这与差距 #5（狼人杀 LLM-only）叠加：狼人杀默认不可玩。
2. **UNO 与狼人杀两条旗舰新游戏链路端到端不可用**（见 P1-1/3/4/5/6 + P2-27 + 注册表落差）。UNO 被 commit 宣称"完整接入"却在主 UI 无法开局且泄密；狼人杀的 BayesSolver 坏、非多变种、夜间目标规则泄密。这两类是 PRD "游戏多样性"卖点的支柱，当前构成信誉风险。
3. **工程信任基建失灵且无 CI 强制**（§3.4）：ruff 不净、测试计数过期、无超时、无闸门→"验收全绿"为自证且当前为假。任何上述区域的回归静默入库。

### 5.3 陪伴评估的"网格局限"——最实质功能落差

陪伴层评估（`evaluation._board_heuristic` + `review/analyzer._board_score`）**只对方阵棋盘工作**。对德州/麻将/狼人杀/UNO/卧底 5 个卖点游戏：非终局评估恒 0.0"胶着"→好棋/失误永不触发、提示"方向"恒"优先占住关键位置"（无意义）、复盘 turning_point 退化、blunder 永不检出、改进建议通用化。即 PRD §4.4"教学与复盘"对最需要陪伴的牌/麻将/社交游戏**实质空转**——只有 moon_chess/gomoku 两个网格游戏享受完整陪伴智能。这是"陪你玩"承诺与当前实现间最实质的落差（非 Bug 而是覆盖盲区），建议作为下一阶段最高优先功能项。

### 5.4 安全（对外暴露前必修）

PRD §6.3"对外暴露服务前必须加认证和限流"当前**未满足且明确记录为暂缓**（`security-notes.md:27-28,40`）：`Access-Control-Allow-Origin: *` + 无认证 + 同步 30s LLM 调用。本地 127.0.0.1 工具可接受，但任何非本地暴露前必须：同源 CORS、token/session 认证、请求限流、LLM 调用改作业队列（`/api/chat`、`/api/rules/translate`、`/api/custom/games` 的同步 30s LLM 在负载下会耗尽工作线程）。已修：路径穿越白名单、10MB body 上限、控制字符清洗、sha256 key、gym 克隆独立、jobscape 反对称、job 裁剪——`test_security_fixes.py` 15 项 1:1 覆盖。

---

## 6. 建议（按优先级）

### P1（阻断/红线，应立即修）

1. **P1-1 BayesSolver TypeError**——一行修复（`bayes_solver.py:68` 传 `flat` 或删 `obs` 形参）+ 补一个强制 p0 行动的测试。
2. **P1-2 德州 `ai_hand_name` 弃牌泄露**——AI 分支改用 `revealed` 作门（builtin + 自定义扑克族两处）。
3. **P1-3 UNO `handsSnapshot` 泄露**——uno visibility 加 `env` 过滤 + 轮转后清理。
4. **P1-4 前端 UNO 棋盘**——加 `uno`/`card` 族棋盘 + `UnoSnapshot` 类型，或暂将 UNO 移出 `/games` 注册表直到棋盘就绪。
5. **P1-5 L1 变体 `_parse_rules` 未守卫 + UNO 分支缺失**——`variant_translator.py:407` 包 try/except（仿 `llm_translator.py:70-75`）；两个 `_apply_parameters` 加 `uno` 分支 + 测试。

### P2（高价值）

6. **扩大探针 + smoke**（§3.2-B P2-12）：种子驱动随机 rollout ≥20 状态 + 每 viewer 一次 `project_observation`/`get_info_set_key` 比对 + smoke 加 `sample_chance`/`get_utility`/`is_terminal`。这是把 5 个静默编译分叉（P2-6~10）转为自动检测回退的**最高杠杆单点改动**。
7. **隐藏信息两层收口**：快照层走 `project_observation`（P2-1 麻将暗杠、P2-2 last_drawn、P2-5 客户端隐藏数组）；规则层补 `visibility.env`（P2-3 狼人夜间目标、P1-3 UNO）。
8. **陪伴评估扩展到非网格游戏**（§5.3）：为德州/麻将引入公开特征评估（pot odds/听牌数等公开启发式），否则卖点游戏陪伴空转。
9. **CFR save/load**（P2-13）+ **零和守卫**（P2-14）+ **麻将 34 步长参数化**（P2-15）。
10. **五子棋聊天坐标**（P2-18，一行）+ **`do_GET` 双响应**（P2-19，删 144-154）+ **L1 温度强制 0**（P2-22）+ **传输重试**（P2-23）。
11. **注册表对齐**：把狼人杀/卧底/UNO 接入平台 9 游戏注册表（需先修 P1-1/3/4/5），并修文档计数（9/17/961）。

### P3（工程化）

12. **CI 闸门**：`ruff check/format` + `pytest --timeout`（装 `pytest-timeout`）+ 文档计数校验；先把现有 25 lint + 16 format 一次性 `ruff check --fix && ruff format` 清零。
13. **文档同步**：重写 README（Layer1 已实现、17 游戏、结构树补陪伴模块、`docs/merge/`→`archive/.../mergedocs/`）；修 CLAUDE.md 测试数 961、`docs/merge/` 指针；修 architecture.md 游戏数 9/17、测试数 961、版本号。
14. **mahjong.json 瘦身**（P2-26）：提取 `_standard_win` 等为命名 alias + `CALL`，~512KB→~20KB。
15. 前端 a11y（P2-29）、响应式侧栏（P2-28）、类型漂移（P2-30）；`train_cli.py` 装后失效（改用 `package_dir` 或打包 `train-cli`）。

---

## 7. 结论

Gavis 在架构与既有 Bug 修复上是**扎实的**：四层硬约束被遵守、2026-08-22 审查的 17 个 P1 全部真修且带回归测试、陪伴六模块端到端接通、在线学习闭环有效、前端构建健康、生成器确定性优秀、PRD §8 五条差距关了四条。这是一个**有实质内核、非空壳**的项目。

但本次审查覆盖了既有审查的盲区（L4 平台快照层、规则 `visibility` 声明层、L1 翻译的 UNO/变体路径、前端），发现：**隐藏信息红线在两层被击穿**、**UNO 与狼人杀两条旗舰链路端到端不可用**、**陪伴评估对卖点游戏空转**、**工程信任基建失灵且无 CI**。其中 6 个新 P1 应立即修复，其余 P2 以"扩大探针"和"隐藏信息两层收口"为最高杠杆。

**一句话**：内核与修复记录可信，但"陪你玩"承诺对牌/麻将/社交类卖点游戏的覆盖与隐藏信息完整性尚未兑现，且缺乏 CI 闸门使回归静默入库——建议优先补齐盲区审查发现的 P1，再推进陪伴评估的非网格覆盖。

---

## 附录：审查覆盖文件清单

- L1：`layer1_translator/` 13 模块（含 `external_frontend_reader.py`、`template_translator.py`；`datasets.py` 不存在）
- L2：`layer2_engine/core/` 7 模块（`engine`/`state_graph`/`expr_eval`/`rules_compiler`/`api_key`/`smoke_validator`/`llm`）
- L3：`layer3_solvers/` ~49 文件（`base`/`mcts`/`cfr`/`hybrid`/`ppo`/`psro`/`marl`/`llm`/`mahjong`/`werewolf`/`auto_selector`/`common`/`social`）
- L4 陪伴：`agent/`、`difficulty/`、`profile/`、`review/`、`online_learning/` 全部
- L4 平台：`frontend/platform/{server,session,games,chat,history,benchmark,custom_games,families/*}`、`binding/`、`encoding/`、`solver_provider.py`、`vision_bridge.py`
- 规则：`rules/*.json`（7）+ `_gen_{mahjong,werewolf,undercover,uno}.py` + `_smoke_uno.py`
- 前端：`platform-frontend/src/**`（47 ts/tsx）
- 测试/文档/CLI：`tests/`（55 文件，961 例）、`train-cli/`、`pyproject.toml`、`pytest.ini`、`CLAUDE.md`、`README.md`、`docs/design/*`、`docs/product/*`、`docs/user/*`
- 实跑：`train.py --list`、`pytest --collect-only`、`ruff check/format`、`npm run build`、4 生成器幂等、BayesSolver 复现
