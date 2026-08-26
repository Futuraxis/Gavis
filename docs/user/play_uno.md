# UNO 使用说明

UNO（经典 108 张牌）已接入 Gavis 规则体系：一套 `rules/uno.json`
（v5.2 零 BUILTIN + variants 声明式）通吃 **六种变体 × 2–10 人**。
引擎装配、可见性（部分可观测）、终局/收益均由纯数据声明驱动，无
per-game 适配器。

## 启动

```bash
python _gen_uno.py                  # 重新生成 rules/uno.json（改了生成器后）
python -m pytest tests/test_layer2_engine/test_uno.py -v   # 引擎级测试
python train-cli/train.py --list    # 查看注册表（含 6 个 uno_* 条目）
```

训练 / 运行时装配走 `train-cli/games.py` 注册表：`uno` 与
`uno_seven_zero` / `uno_jump_in` / `uno_stacking` / `uno_draw_until` /
`uno_strict_wild4` 六个条目，默认 4 人，Hybrid 求解器以
`imperfect_information=True` 开部分可观测配置。

## 基础规则

- 每人 7 张手牌，翻一张作为弃牌（首张特殊效果按规则结算）。
- 出牌必须同色或同符号；`万能(wild)` 与 `万能+4(wild4)` 任意时刻可出并选色。
- `跳过`：下家被跳过；`反转`：方向反转（2 人局等价跳过）；`+2`/`+4`：
  下家吃对应罚牌并跳过（罚牌经 penalty_pick 逐张结算）。
- 摸牌后若可接可 `play_drawn`，否则 `pass` 过。
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
  变体由 `variants` 节声明（`options[n].constants` 布尔补丁），引擎纯
  数据解析；未知 variant → ValueError。
- 牌堆无实体数组：`deck = 108 − (所有手牌 + 弃牌)`，由 `undrawn_cards`
  查询（uniform 抽样）与 `deck_count` 别名表达。
- 修改规则请改 `_gen_uno.py` 后运行 `python _gen_uno.py` 重新生成。

## 已知说明

- 规则中的 `hand_of` 别名（10 分支 switch）被内联在查询/合法条件里，
  超出编译器的 switch-in-comprehension 支持形状 → 引擎自动回退纯解释器
  （设计内行为，功能与测试不受影响；如需提速可后续把 `hand_of` 查询
  改为直接数组引用）。