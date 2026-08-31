# UNO 使用说明

UNO（经典 108 张牌）已接入 Gavis 规则体系：一套 `rules/uno.json`
（v5.2 零 BUILTIN + variants 声明式）通吃 **六种变体 × 2–10 人**。
引擎装配、可见性（部分可观测）、终局/收益均由纯数据声明驱动，无
per-game 适配器。

## 启动

```bash
python scripts/_gen_uno.py            # 重新生成 rules/uno.json（改了生成器后）
python -m pytest tests/test_layer2_engine/test_uno.py -v   # 引擎级测试（39 例）
python train-cli/train.py --list      # 查看注册表（含 6 个 uno_* 条目 + hybrid 管线）
```

**平台对局**：六款 UNO 已登记平台注册表（游戏大厅可见，徽标 🎴 UNO），
默认 4 人（1 人 + 3 AI），UNO 桌面组件按颜色分组展示手牌、台面顶牌与
罚牌状态。启动平台：`python -m layer4_interface.frontend.platform.server`
（需先 `cd platform-frontend && npm run build`）。

训练 / 运行时装配走 `train-cli/games.py` 注册表：`uno` 与
`uno_seven_zero` / `uno_jump_in` / `uno_stacking` / `uno_draw_until` /
`uno_strict_wild4` 六个条目，默认 4 人，Hybrid 求解器以
`imperfect_information=True` 开部分可观测配置。

## 基础规则

- 每人 7 张手牌，翻一张作为弃牌（首张特殊效果按规则结算）。
- 出牌必须同色或同符号；`万能(wild)` 与 `万能+4(wild4)` 任意时刻可出并选色
  （牌 id 形如 `wild_2` / `wild4_2`——后缀数字是 4 张副本的编号，同 `r7a`/`r7b`）。
- `跳过`：下家被跳过；`反转`：方向反转（2 人局等价跳过）；`+2`/`+4`：
  下家吃对应罚牌并跳过（罚牌经 penalty_pick 逐张结算）。
- 摸牌后若可接可 `play_drawn`（万能牌走 `play_drawn_wild`，同样四选一
  颜色），否则 `pass` 过。
- 终局：手牌清空即胜（`hand_empty`）；牌堆空且当前玩家无可打 → `stuck`
  卡死；回合数上限 2000（`max_turns`）。胜者 = 手牌最少者
  （`least_player`），收益 +1 / 其余 -1。

## 六种变体

| 变体（id） | 规则 |
|---|---|
| 经典 classic | 标准 UNO |
| 7-0 换手 seven_zero | 打出 7 选目标玩家换手；打出 0 全场按方向移交给下一家 |
| 抢牌 jump_in | 同色同数字可抢出（仅数字 1–6、8、9，按座位序排队，可抢出/放弃） |
| 叠加 stacking | 响应同色 +2 可叠加、+4 可叠在任何罚牌上；吃罚者一次吃下全部 |
| 摸到能打 draw_until | 摸牌持续到摸到可打的牌为止（牌堆耗尽自动停，转 draw_result） |
| 严格+4 strict_wild4 | 手牌仍有台面颜色时禁止出 +4 |

## 规则实现

- 全部逻辑在 `rules/uno.json`（一张 JSON 通吃六变体 × 2–10 人），
  变体由 `variants` 节声明（`options` **dict**：键=变体名，值为
  `{"constants": {...}}` 布尔补丁），引擎纯数据解析；未知 variant →
  ValueError。
- 牌堆无实体数组：`deck = 108 − (所有手牌 + 弃牌)`，由 `undrawn_cards`
  查询（uniform 抽样）与 `deck_count` 别名表达；人数 2..10 由
  `player_ids` 公式生成（players 预置 p0..p9，默认 player_count=4）。
- 修改规则请改 `scripts/_gen_uno.py`（argparse：`--players` 默认 4、
  `--out` 默认 rules/uno.json）后运行 `python scripts/_gen_uno.py`
  重新生成。

## 已知说明

- 规则中的 `hand_of` 别名（10 分支 switch）内联在查询/合法条件里：
  编译器对 switch 生成无 `:=` 的首匹配 if/elif 链，因此
  switch-in-comprehension 形状可正常编译（`:=` 在 comprehension 迭代
  表达式里是 SyntaxError，曾是整套规则回退纯解释器的原因）。
  手牌部分可观测（`visibility` 隐藏他人牌面但保留张数）。