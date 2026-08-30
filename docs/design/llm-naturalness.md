# 传给 LLM 的信息「不过分技术化」——系统性排查与修复记录

> 起因：麻将 `s1`（索子一）在对话/提示/复盘里被显示成「1条」或裸 id `s1`。
> 用户要求系统性排查「传给 LLM 的信息过分技术化」的问题：LLM 直面文本
> 里出现的应当是自然中文（一条/三万/红中、跟注/加注、卧底/平民），
> 机器契约（快照 id、canonical key、工具参数、规则原语）原样保留。

## 1. 设计原则（红线 / 双轨）

- **快照载荷是浏览器契约**：`my_hand: ["s1", ...]`、`discard:s1`、
  `act:call:2`、`play:r7a`、`claim_chi:['m1','m2','m3']`、`place:cell_r_c`
  等 **一字不改** —— 前端 `types.ts` / 回合渲染 / `/match/move` 校验都依赖它们。
- **canonical key 是引擎契约**：`ActionInstance.canonical_key` 仍是
  `discard:s1` / `act:call:2` —— 只在**面向 LLM 的渲染文本**处换成中文，
  并在括号内附机器参数（`打出 一条（tile=s1）`），让模型既能读懂又能
  产出可校验的 `make_move` 参数。双轨：**读得懂 + 用得对**。
- **隐藏信息红线不变**：所有入口仍然是 `spec.build_snapshot`（玩家投影）
  与 `hidden_guard` 扫描；人化只改**读法**，不扩大可见范围。
- **fail-soft**：未知 id/形状直出原值（宁缺毋滥，不崩）。

## 2. 全部 LLM 触面清单（审计结论）

| # | 模块 | 触面 | 修复前 | 修复后 |
|---|------|------|--------|--------|
| 1 | `frontend/engine_helpers.py` | **统一中文名称层**（新增） | — | `mahjong_tile_name`（`s1`→一条）、`uno_card_name`、`poker_card_name`、`social_role_name`、`piece_name(s)`、`seat_label`、`canonical_family_text`（各族 canonical key→中文）、`game_family` |
| 2 | `agent/dialogue_engine.py` | 教学/常规对局 user prompt | `player_hand: ["s1",...]`；`coach_reference: "discard:s1"` | `player_hand: "一条、三万…"`；`coach_reference: "打出 一条"`；机器键 `coach_reference_key: "discard:s1"` 保留（`_payload_family` 做族推断） |
| 3 | `agent/skills.py` | `suggest_hint` 的机械提示 | `演示走法：discard:s1` | `演示走法：打出 一条`（`canonical_family_text`，机器 key 仍留在 `result["action"]`） |
| 4 | `frontend/platform/chat.py` | `get_match_state` / 合法动作上下文 | `my_hand: ["s1",...]`、`合法动作: [{"type":"discard","tile":"s1"}]` | `my_hand: "一条、三万…"`、`合法动作: 打出 一条（tile=s1）; 摸牌`（`_humanize_snap` / `_legal_payload_text`；键名保留保证测试契约） |
| 5 | `frontend/platform/games.py` + `families/*` | `describe_action`（历史日志/教练参考/复盘时间线） | `discard:s1`、`act:call:2`、`play:r7a`、`cell_0_0` | 中文（麻将经 `mahjong_tile_name` 自动修复；poker/uno/grid 走 `canonical_family_text`；social 走 `_ACTION_LABELS` + `seat_label`） |
| 6 | `layer3_solvers/llm/ollama_solver.py` | 狼人杀 LLM prompt | `身份是wolf` | `身份是狼人（wolf）`（本地 `_ROLE_NAMES`，Layer 3 不依赖 Layer 4；输出仍是机器 JSON） |
| 7 | `layer3_solvers/social/llm_policy.py` | 社交推理 LLM prompt context | `role: "undercover"` | `role: "卧底"`（本地 `_ROLE_NAMES`） |
| 8 | `platform-frontend/MahjongTable.tsx` | 前端牌面/文本 | `1条`（阿拉伯数字 + 条） | `一条`（`CN_RANK` 中文点数，`tileLabel` 与 `TileView` 统一） |
| 9 | `layer1_translator` | 规则翻译/修复 prompt | 输入规则 JSON / 输出规则 JSON | **不改** —— 产品本身就是 JSON，技术化是必要的 |
| 10 | `layer1_translator` 拆解 prompt（中文描述/解释） | 已有中文 | 确认无需改 |
| — | `vision` 绑定/图片输入 | 图片进模型 | 超出本文范围（见 §4 残余项） |

## 3. 关键实现地点

- **名称层**：`layer4_interface/frontend/engine_helpers.py`
  - `mahjong_tile_name("s1")` → ``一条``；`"m3"` → ``三万``；`"z5"` → ``中``；
    1-9 全部中文数字（一条…九条）。
  - `uno_card_name("r7a")` → ``红7``；`"gsa"` → ``禁``；`"gra"` → ``反转``；
    `"gda"` → ``+2``；`"wild_1"` → ``万能``；`"wild4_1"` → ``+4``。
  - `poker_card_name("sA")` → ``黑桃A``；`"hT"` → ``红桃10``；`"dQ"` → ``方块Q``。
  - `social_role_name("undercover")` → ``卧底``；`"wolf"` → ``狼人``；
    `"seer"` → ``预言家``；`"witch"` → ``女巫``；`"hunter"` → ``猎人``；
    `"villager"` → ``村民``；`"civilian"` → ``平民``；`"blank"` → ``白板``。
  - `seat_label("p1")` → ``2号玩家``；`seat_label("p0", self_pid="p0")` → ``你``。
  - `canonical_family_text(family, key)`：
    - mahjong：`discard:s1` → ``打出 一条``；`claim_chi:m1,m2,m3` → ``吃 一万、二万、三万``；`win_self` → ``自摸``。
    - poker：`act:call:2` → ``跟注 2``；`act:check` → ``过牌``。
    - uno：`play:r7a` → ``打出 红7``；`play:r7a:红` → ``打出 红7（红）``；`play7:r7a:p1` → ``出7 红7 → 2号玩家``。
    - grid：`place:cell_0_0` → ``落子 第1行第1列``。
    - social：`vote:p1` → ``投票 2号玩家``；`speak:claim` → ``发言（claim）``。
- **会话层**：`chat.py` `_humanize_snap` / `_legal_payload_text` / `_mahjong_melds_text`
  （麻将对子表手牌/牌河/副露/最后打出/阶段、UNO 顶牌/弃牌堆/罚牌目标、
  德州公共牌/手牌/下注动作、社交身份 —— 键名与预算 `_STATE_MAX_CHARS` 不变）。
- **输出侧不人化的字段**（有意保留）：`last_action`（环境动作 id，精确性
  优先）、`raise_amounts` / `payoffs` / `wall_remaining`（数值）——
  它们不是“牌面”，原值对模型更精确；文档在 §4 标注为已知残余。

## 4. 残余项 / 后续（不阻塞本次修复）

- `infer_game_id` 无法区分 UNO 与麻将（两者投影都用 `hand_view_*` 键）。
  调用方应优先走显式 `game_id`/`session.family`（`game_family(game_id)`），
  已知限制留档。
- `last_action` 等环境动作 id 保持原值（见上）。
- 视觉绑定路径（图片直接进模型）不在文本人化范围。
- 狼人杀 speak 意图 id（`claim/accuse/...`）在 canonical 中文里保留原词
  （`发言（claim）`）—— 它是发言内容标签而非牌面，过度翻译反而失真。

## 5. 验证要点（回归）

- 测试契约不变：`test_chat.py` 断言 ``"my_hand" in tool_msgs[0]["content"]``、
  `test_teaching.py` 断言 ``"player_hand" in llm.user`` —— 键名保留，只有
  **值**人化，两个断言天然通过。
- 快照键集契约（`MAHJONG_SNAPSHOT_KEYS` 等）不受影响 —— 人化只作用于
  chat/提示的**渲染层**，`spec.build_snapshot` 本体不动。
- Layer 3 不依赖 Layer 4：身份名映射在 Layer 3 本地维护（`_ROLE_NAMES`）。