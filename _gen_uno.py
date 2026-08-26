#!/usr/bin/env python3
"""UNO rules generator — generates ``rules/uno.json`` (v5.2).

Generates the standard 108-card UNO rule set with six declarative variants:

    python _gen_uno.py [--players 4] [--out rules/uno.json]

Composition: 108 unique card ids (r0..r8a/b, skip/reverse/draw2 per color ×2,
wild ×4, wild4 ×4), 7 cards per hand, 2..10 players.

Variants (``variants.options`` constants patches — all off in ``classic``):

  - ``classic``       标准 UNO：数字 0-9 / skip / reverse / +2 / wild / wild4
  - ``seven_zero``    7-0 规则：打出 7 可与任一玩家换手（play7 带 target）；
                      打出 0 全场手牌按方向移交
  - ``jump_in``       抢牌：普通数字打出后，手牌有同色同数字牌的其他玩家
                      可按座位序抢先出牌（jumpQueue / jumpIndex 窗口）
  - ``stacking``      +2/+4 叠加：被罚玩家可叠打同色 +2（或任意 +4 选色），
                      罚牌累计（pendingDraw），否则吃下累计罚牌并跳过回合
  - ``draw_until``    摸到能打：打不出时持续摸牌直到摸到可打牌
  - ``strict_wild4``  严格 +4：手牌仍有台面颜色时禁止出任意 4

Game flow: deal(7×n) → flip(翻开首张，处理首张特殊效果) → play 循环：
出牌（play / play_wild / play7）或摸牌（draw → pick → draw_result 可打可过）。

Turn rotation is fully local/O(1) via ``env.turn`` / ``env.direction``;
special-card effects are settled inline by the effector of the played card
(skip=进2、reverse=翻方向（2 人局相当于 skip）、+2/+4=罚牌循环、7-0 换手/
移交、抢牌窗口)。

Win conditions:
  - 任一玩家手牌清空                      → 该玩家胜（手牌最少者 = 清空者）
  - play 阶段牌堆耗尽且当前玩家无可打牌    → 手牌最少者胜
  - penalty_pick 阶段牌堆耗尽（防空洞）    → 手牌最少者胜
  - 回合数上限 (2000)                     → 手牌最少者胜

Design notes
------------
- 部分可观测（v5.2 visibility）：``hand_view_pX`` 对非本人隐藏牌面；他人
  行仍可见（观察侧以字段表达手牌数）。``discard_view`` / ``card`` 公开。
- 发牌/摸牌/罚牌全部走 ``chance``/``uniform``（``undrawn_cards`` 查询纯数据
  排除 discard 与所有手牌中的牌——摸牌即真实牌堆均匀采样，无需物理牌堆数组）。
- 万能牌选色走第二个 action（``play_wild`` 带 color 参数）；7-0 的换手目标
  走 ``play7``（card + target），避免普通 play 的笛卡尔积爆炸。
- 7-0 换手/移交用 ``handsSnapshot`` 快照 + ``forEach`` 重绑（克隆安全：
  env 列表共享引用，op 全部走 rebind）；抢牌用 ``jumpQueue`` 窗口。
- 各特殊牌分支互斥（0/7 在 seven_zero 下不再落入普通数字分支，避免
  重复推进）；``turnCount`` 每次决策 +1 作为安全上限计数器。
- 终局为纯条件（terminal / utility.when / env.winner 共用 ``game_ended``），
  ``env.winner`` 在最后一个决策 effector 尾部写入（平台展示用）。
- 未知 variant → ValueError（引擎纯数据解析，无注入 API）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLORS = ["r", "b", "g", "y"]  # red / blue / green / yellow
NUMBER_SYMBOLS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# 抢牌仅限普通数字牌（0/7 有特殊连锁效果，不参与抢牌窗口）。
JUMP_SYMBOLS = ["1", "2", "3", "4", "5", "6", "8", "9"]

VARIANTS = [
    "classic",
    "seven_zero",
    "jump_in",
    "stacking",
    "draw_until",
    "strict_wild4",
]

HAND_SIZE = 7
MAX_TURNS = 2000


def _build_deck() -> tuple[list[str], dict[str, str], dict[str, str]]:
    """108 张唯一牌 id + (符号, 颜色) 映射；0 每色 1 张，其余每色 2 张。"""
    ids: list[str] = []
    symbol_of: dict[str, str] = {}
    color_of: dict[str, str] = {}
    for c in COLORS:
        cid = f"{c}0"
        ids.append(cid)
        symbol_of[cid] = "0"
        color_of[cid] = c
        for n in "123456789":
            for tag in ("a", "b"):
                cid = f"{c}{n}{tag}"
                ids.append(cid)
                symbol_of[cid] = n
                color_of[cid] = c
        for sym, tag in (("skip", "s"), ("reverse", "r"), ("draw2", "d")):
            for s in ("a", "b"):
                cid = f"{c}{tag}{s}"
                ids.append(cid)
                symbol_of[cid] = sym
                color_of[cid] = c
    for i in range(1, 5):
        cid = f"wild_{i}"
        ids.append(cid)
        symbol_of[cid] = "wild"
        color_of[cid] = "none"
        cid = f"wild4_{i}"
        ids.append(cid)
        symbol_of[cid] = "wild4"
        color_of[cid] = "none"
    assert len(ids) == 108, len(ids)
    assert len(set(ids)) == 108
    return ids, symbol_of, color_of


CARD_IDS, SYMBOL_OF, COLOR_OF = _build_deck()


# ── 表达式构建 helper ─────────────────────────────────────────────────


def V(path):
    return {"var": path}


def C(value):
    return {"const": value}


def GET(obj, field):
    return {"get": [obj, field]}


def AT(container, idx):
    return {"at": [container, idx]}


def EQ(a, b):
    return {"eq": [a, b]}


def NEQ(a, b):
    return {"neq": [a, b]}


def GT(a, b):
    return {"gt": [a, b]}


def GTE(a, b):
    return {"gte": [a, b]}


def AND(*args):
    return {"and": list(args)}


def OR(*args):
    return {"or": list(args)}


def NOT(a):
    return {"not": a}


def IF(cond, then_, else_=None):
    return {"if": {"cond": cond, "then": then_, "else": else_}}


def CALL(name, *args):
    return {"call": [name, *args]}


def COUNT(expr):
    return {"count": expr}


def FILTER(lst, where, as_var="$node"):
    return {"filter": {"list": lst, "as": as_var, "where": where}}


def ANY(lst, where, as_var="$node"):
    return {"any": {"list": lst, "as": as_var, "where": where}}


def ALL(lst, where, as_var="$node"):
    return {"all": {"list": lst, "as": as_var, "where": where}}


def MAP(lst, expr, as_var="$node"):
    return {"map": {"list": lst, "as": as_var, "expr": expr}}


def RANGE(frm, to):
    return {"range": {"from": frm, "to": to}}


def SINGLE(item):
    """[item] —— 防止 concat 摊平的单元素列表包装。"""
    return MAP(RANGE(C(0), C(1)), item)


def CONCAT(*items):
    return {"concat": list(items)}


def SORT(lst, by_expr, as_var="$node"):
    return {"sort": {"list": lst, "as": as_var, "by": by_expr}}


def _player_ids() -> list[str]:
    return [f"p{i}" for i in range(10)]


# ── 规则 idiom ────────────────────────────────────────────────────────


def _undrawn_filter():
    """card view 过滤器：不在 discard 且不在任何玩家手牌（count==0 判定）。

    语义：牌堆 = 108 张中尚未翻开持有/打出的牌；uniform 采样即真实摸牌。
    """
    card_id = GET(V("$node"), "id")
    return AND(
        NOT({"contains": [V("$discard"), card_id]}),
        ALL(
            V("$constants.player_ids"),
            EQ(COUNT(FILTER(CALL("hand_of", V("$pid")), EQ(V("$h"), card_id), as_var="$h")), C(0)),
            as_var="$pid",
        ),
    )


def _deck_count():
    """牌堆剩余张数（走 deck_count 别名——纯算术，不经过 query，编译器友好）。"""
    return CALL("deck_count")


def _hand_empty():
    """任一玩家手牌清空（获胜条件之一；发牌/翻牌阶段不触发——开局手牌全空）。"""
    return AND(
        NOT(OR(EQ(V("$env.phase"), C("deal")), EQ(V("$env.phase"), C("flip")))),
        ANY(V("$constants.player_ids"), EQ(COUNT(CALL("hand_of", V("$node"))), C(0))),
    )


def _stuck():
    """卡死：play 阶段 + 牌堆耗尽 + 当前玩家无可打牌（含严格 +4 约束）。"""
    return AND(
        EQ(V("$env.phase"), C("play")),
        EQ(_deck_count(), C(0)),
        EQ(
            COUNT(
                FILTER(
                    CALL("hand_of", V("$env.turn")),
                    CALL("can_play", V("$node"), CALL("hand_of", V("$env.turn"))),
                )
            ),
            C(0),
        ),
    )


def _stuck_penalty():
    """防空洞：penalty_pick 阶段牌堆耗尽（叠加链吃罚牌时牌堆恰好为空）。"""
    return AND(EQ(V("$env.phase"), C("penalty_pick")), EQ(_deck_count(), C(0)))


def _max_turns():
    return GTE(V("$env.turnCount"), C(MAX_TURNS))


def _end_conditions():
    """终局条件（terminal 与 utility.when / env.winner 共用同一布尔）。"""
    return [_hand_empty(), _stuck(), _stuck_penalty(), _max_turns()]


# ── 视图 / 查询 / 可见性 ──────────────────────────────────────────────


def _views():
    views = {
        "card": {"from": {"type": "literal", "list": {"var": "card_ids"}}, "fields": {"id": V("$self.value")}},
        "player": {"from": {"type": "literal", "list": {"var": "player_ids"}}, "fields": {"id": V("$self.value")}},
        "discard_view": {"from": {"type": "enum", "array": "discard"}, "fields": {"id": V("$self.value")}},
    }
    for pid in _player_ids():
        views[f"hand_view_{pid}"] = {
            "from": {"type": "enum", "array": f"hand_{pid}"},
            "fields": {"id": V("$self.value")},
        }
    return views


def _queries():
    return {"undrawn_cards": {"view": "card", "filter": _undrawn_filter()}}


def _visibility():
    """部分可观测：手牌视图对非 viewer 隐藏牌面（count 保留在字段）。"""
    return {
        "default": "partial",
        "rules": [
            {
                "view": f"hand_view_{pid}",
                "filter": {"not": {"eq": [V("$viewer"), C(pid)]}},
                "fields": {"id": "hidden"},
            }
            for pid in _player_ids()
        ],
    }


# ── actions ───────────────────────────────────────────────────────────


def _hand_param(filter_expr=None):
    """当前回合玩家手牌候选域（v5.1 动态数组模板）。"""
    pdef = {"domain": {"array": {"template": "hand_{$env.turn}"}}}
    if filter_expr is not None:
        pdef["filter"] = filter_expr
    return pdef


def _is_wild_expr(card_var="$card"):
    csym = CALL("symbol_of", V(card_var))
    return OR(EQ(csym, C("wild")), EQ(csym, C("wild4")))


def _actions():
    return [
        {
            "id": "play",
            "type": "move",
            "phases": ["play"],
            # 非万能牌；seven_zero 下 7 必须走 play7（带换手目标）。
            "params": {
                "card": _hand_param(
                    {
                        "and": [
                            {"not": _is_wild_expr()},
                            {
                                "not": {
                                    "and": [
                                        V("$constants.seven_zero"),
                                        EQ(CALL("symbol_of", V("$card")), C("7")),
                                    ]
                                }
                            },
                        ]
                    }
                )
            },
            "legal": CALL("can_play", V("$card"), CALL("hand_of", V("$env.turn"))),
            "effectRef": "do_play",
            "canonicalKey": {"template": "play:{card}"},
        },
        {
            "id": "play_wild",
            "type": "move",
            "phases": ["play"],
            "params": {
                "card": _hand_param(_is_wild_expr()),
                "color": {"domain": list(COLORS)},
            },
            "legal": CALL("can_play", V("$card"), CALL("hand_of", V("$env.turn"))),
            "effectRef": "do_play_wild",
            "canonicalKey": {"template": "play:{card}:{color}"},
        },
        {
            "id": "play7",
            "type": "move",
            "phases": ["play"],
            "params": {
                "card": _hand_param(EQ(CALL("symbol_of", V("$card")), C("7"))),
                "target": {
                    "domain": {"expr": V("$constants.player_ids")},
                    "filter": {"neq": [V("$target"), V("$env.turn")]},
                },
            },
            "legal": AND(
                EQ(V("$constants.seven_zero"), C(True)),
                CALL("can_play", V("$card"), CALL("hand_of", V("$env.turn"))),
            ),
            "effectRef": "do_play7",
            "canonicalKey": {"template": "play7:{card}:{target}"},
        },
        {
            "id": "draw",
            "type": "move",
            "phases": ["play"],
            "params": {},
            "legal": GT(_deck_count(), C(0)),
            "effectRef": "do_draw",
            "canonicalKey": {"const": "draw"},
        },
        # ── draw_result：可打出刚摸的牌（若可打）或过 ──
        {
            "id": "play_drawn",
            "type": "move",
            "phases": ["draw_result"],
            "params": {"card": _hand_param()},
            "legal": AND(
                EQ(V("$card"), V("$env.drawnCard")),
                CALL("can_play", V("$card"), CALL("hand_of", V("$env.turn"))),
            ),
            "effectRef": "do_play_drawn",
            "canonicalKey": {"template": "play:{card}"},
        },
        {
            "id": "pass",
            "type": "move",
            "phases": ["draw_result"],
            "params": {},
            "legal": C(True),
            "effectRef": "do_pass",
            "canonicalKey": {"const": "pass"},
        },
        # ── jump（抢牌变体专用：同色同数字普通数字牌）──
        {
            "id": "jump_play",
            "type": "move",
            "phases": ["jump"],
            "params": {"card": _hand_param()},
            "legal": AND(
                EQ(CALL("color_of", V("$card")), V("$env.topColor")),
                EQ(CALL("symbol_of", V("$card")), V("$env.topSymbol")),
                {"contains": [V("$constants.jump_symbols"), CALL("symbol_of", V("$card"))]},
            ),
            "effectRef": "do_jump",
            "canonicalKey": {"template": "jump:{card}"},
        },
        {
            "id": "jump_pass",
            "type": "move",
            "phases": ["jump"],
            "params": {},
            "legal": C(True),
            "effectRef": "do_jump_pass",
            "canonicalKey": {"const": "jump_pass"},
        },
        # ── respond（叠加变体专用：叠同色 +2 / 任意 +4（选色）/ 吃罚牌）──
        {
            "id": "stack2",
            "type": "move",
            "phases": ["respond"],
            "params": {
                "card": _hand_param(
                    {
                        "and": [
                            EQ(CALL("symbol_of", V("$card")), C("draw2")),
                            EQ(CALL("color_of", V("$card")), V("$env.topColor")),
                        ]
                    }
                )
            },
            "legal": C(True),
            "effectRef": "do_stack2",
            "canonicalKey": {"template": "stack:{card}"},
        },
        {
            "id": "stack4",
            "type": "move",
            "phases": ["respond"],
            "params": {
                "card": _hand_param(EQ(CALL("symbol_of", V("$card")), C("wild4"))),
                "color": {"domain": list(COLORS)},
            },
            "legal": C(True),
            "effectRef": "do_stack4",
            "canonicalKey": {"template": "stack:{card}:{color}"},
        },
        {
            "id": "take_penalty",
            "type": "move",
            "phases": ["respond"],
            "params": {},
            "legal": C(True),
            "effectRef": "do_take_penalty",
            "canonicalKey": {"const": "take"},
        },
    ]


# ── effectors ─────────────────────────────────────────────────────────


def _set_env(key, value):
    return {"op": "setEnv", "key": key, "value": value}


def _inc(key, by):
    return {"op": "inc", "key": key, "by": by}


def _append(arr_name, value):
    return {"op": "append", "array": arr_name, "value": value}


def _remove(arr_name, value, count=None):
    op = {"op": "remove", "array": arr_name, "value": value}
    if count is not None:
        op["count"] = count
    return op


def _set_array(arr_name, value):
    return {"op": "setArray", "array": arr_name, "value": value}


def _as_ops(ops):
    """确保分支体是 ops 列表：单个 op dict 会被包成单元素列表
    （否则解释器把 dict 当可迭代对象遍历其 key，子 op 变成字符串）。"""
    return [ops] if isinstance(ops, dict) else ops


def _branch(cond, then_ops, else_ops=None):
    op = {"op": "branch", "if": cond, "then": _as_ops(then_ops)}
    if else_ops is not None:
        op["else"] = _as_ops(else_ops)
    return op


def _call_effect(ref, args=None):
    return {"op": "callEffect", "effectRef": ref, "args": args or {}}


def _seat_of(player):
    return CALL("seat_of", player)


def _advance(seat, steps, direction=None):
    """seat_after(seat, steps, direction) 调用（方向缺省取 env.direction）。"""
    if direction is None:
        direction = V("$env.direction")
    return {"call": ["seat_after", seat, {"const": steps}, direction]}


def _advance_pid(seat, steps, direction=None):
    """座位推进并转成 player_id：AT(player_ids, seat_after(...))。

    env.turn / penaltyTarget 必须是 pid（hand_{$env.turn} 等数组模板按 pid
    取名）；seat_after 只返回座位索引，直接赋值会让回合/罚牌目标变成索引。
    """
    return AT(V("$constants.player_ids"), _advance(seat, steps, direction))


def _advance_ops(player_seat, steps, extra_phase="play"):
    """推进回合：turn = 下一家；phase = 指定阶段（默认 play）。"""
    return [
        _set_env("turn", _advance_pid(player_seat, steps)),
        _set_env("phase", C(extra_phase)),
    ]


def _penalty_start_ops(k):
    """+2/+4 罚牌：目标 = 当前出牌者方向下家，进入 penalty_pick 循环。"""
    return [
        _set_env("penaltyTarget", _advance_pid(_seat_of(V("$player")), 1)),
        _set_env("turn", V("$env.penaltyTarget")),
        _set_env("pendingDraw", C(k)),
        _set_env("phase", C("penalty_pick")),
    ]


def _rotate_ops():
    """0 牌：全场手牌按当前方向移交（快照 → forEach 重绑，克隆安全）。"""
    return [
        _set_env(
            "handsSnapshot",
            MAP(
                RANGE(C(0), V("$constants.player_count")),
                CALL("hand_of", AT(V("$constants.player_ids"), V("$node"))),
            ),
        ),
        {
            "op": "forEach",
            "list": V("$constants.player_ids"),
            "as": "$pid",
            "do": [
                _call_effect(
                    "do_set_one_hand",
                    {
                        "pid": V("$pid"),
                        "hand": {
                            "at": [
                                V("$env.handsSnapshot"),
                                {"call": ["rotate_src_seat", _seat_of(V("$pid")), V("$env.direction")]},
                            ]
                        },
                    },
                )
            ],
        },
    ]


def _swap_ops():
    """7 牌：$player 与 $swapTarget 手牌互换（快照先取两份再重绑）。"""
    return [
        _set_env(
            "handsSnapshot",
            MAP(
                CONCAT(SINGLE(V("$player")), SINGLE(V("$swapTarget"))),
                CALL("hand_of", V("$node")),
            ),
        ),
        _set_array({"template": "hand_{$player}"}, AT(V("$env.handsSnapshot"), C(1))),
        _set_array({"template": "hand_{$swapTarget}"}, AT(V("$env.handsSnapshot"), C(0))),
    ]


def _jump_candidates():
    """抢牌候选：除刚出牌者外，手牌存在「同色+同数字」普通数字牌的玩家。"""
    return FILTER(
        V("$constants.player_ids"),
        AND(
            NEQ(V("$pid"), V("$player")),
            ANY(
                CALL("hand_of", V("$pid")),
                AND(
                    EQ(CALL("color_of", V("$node")), V("$env.topColor")),
                    EQ(CALL("symbol_of", V("$node")), V("$env.topSymbol")),
                    {"contains": [V("$constants.jump_symbols"), CALL("symbol_of", V("$node"))]},
                ),
            ),
        ),
        as_var="$pid",
    )


def _jump_window_ops(seat):
    """普通数字推进前的抢牌窗口（jump_in 且有候选时进入 jump 阶段）。"""
    return [
        _set_env("nextTurn", _advance_pid(seat, 1)),
        _branch(
            AND(EQ(V("$constants.jump_in"), C(True)), GT(COUNT(_jump_candidates()), C(0))),
            [
                _set_env("jumpQueue", _jump_candidates()),
                _set_env("jumpIndex", C(0)),
                _set_env("turn", AT(_jump_candidates(), C(0))),
                _set_env("phase", C("jump")),
            ],
            [
                _set_env("turn", V("$env.nextTurn")),
                _set_env("phase", C("play")),
            ],
        ),
    ]


def _do_play_card():
    """核心出牌 effector（所有打出动作经 callEffect 复用）。

    参数：``player``（出牌者）、``card``、``color``（万能牌选色，None=非万能）、
    ``swapTarget``（7-0 换手对象，None=非 7-0 换手）、``advance``（是否推进回合）。

    特殊牌结算（增量局部判定，各分支互斥）：
      - skip    → 进 2（跳过下家；2 人局即自己再出）
      - reverse → 2 人局等价跳过；多人局翻方向后进 1
      - draw2   → stacking 开 respond 窗口（罚牌累计）；否则罚 2
      - wild4   → stacking 开 respond 窗口（+4）；否则罚 4
      - wild    → 只改台面颜色（选色），进 1
      - 0/7     → 7-0 变体：全场移交 / 与 swapTarget 换手，然后进 1
      - 普通数字 → 进 1；jump_in 变体先开抢牌窗口（仅数字且存在候选时）
    """
    seat = _seat_of(V("$player"))
    sym7 = AND(EQ(V("$constants.seven_zero"), C(True)), EQ(CALL("symbol_of", V("$card")), C("7")))
    sym0 = AND(EQ(V("$constants.seven_zero"), C(True)), EQ(CALL("symbol_of", V("$card")), C("0")))
    return [
        _remove({"template": "hand_{$player}"}, V("$card")),
        _append("discard", V("$card")),
        # 台面颜色：万能牌取玩家选色，否则取牌自身颜色。
        _branch(
            NEQ(V("$color"), C(None)),
            [_set_env("topColor", V("$color"))],
            [_set_env("topColor", CALL("color_of", V("$card")))],
        ),
        _set_env("topSymbol", CALL("symbol_of", V("$card"))),
        _set_env("lastActor", V("$player")),
        _inc("turnCount", 1),
        # ── skip：进 2 ──
        _branch(
            EQ(CALL("symbol_of", V("$card")), C("skip")),
            _branch(V("$advance"), _advance_ops(seat, 2)),
        ),
        # ── reverse ──
        _branch(
            EQ(CALL("symbol_of", V("$card")), C("reverse")),
            _branch(
                V("$advance"),
                _branch(
                    EQ(V("$constants.player_count"), C(2)),
                    _advance_ops(seat, 2),  # 2 人局：反转等价跳过
                    [
                        _set_env("direction", {"mul": [V("$env.direction"), C(-1)]}),
                        *_advance_ops(seat, 1),  # 翻方向后从新方向进 1
                    ],
                ),
            ),
        ),
        # ── draw2 ──
        _branch(
            EQ(CALL("symbol_of", V("$card")), C("draw2")),
            _branch(
                V("$constants.stacking"),
                [
                    _inc("pendingDraw", 2),
                    _set_env("turn", _advance_pid(seat, 1)),
                    _set_env("phase", C("respond")),
                ],
                _branch(V("$advance"), _penalty_start_ops(2)),
            ),
        ),
        # ── wild4 ──
        _branch(
            EQ(CALL("symbol_of", V("$card")), C("wild4")),
            _branch(
                V("$constants.stacking"),
                [
                    _inc("pendingDraw", 4),
                    _set_env("turn", _advance_pid(seat, 1)),
                    _set_env("phase", C("respond")),
                ],
                _branch(V("$advance"), _penalty_start_ops(4)),
            ),
        ),
        # ── wild：只改颜色，进 1 ──
        _branch(
            EQ(CALL("symbol_of", V("$card")), C("wild")),
            _branch(V("$advance"), _advance_ops(seat, 1)),
        ),
        # ── 0（7-0 变体：全场移交）──
        _branch(
            sym0,
            _branch(V("$constants.seven_zero"), [*_rotate_ops(), _branch(V("$advance"), _advance_ops(seat, 1))]),
        ),
        # ── 7（7-0 变体：与 swapTarget 换手）──
        _branch(
            sym7,
            _branch(NEQ(V("$swapTarget"), C(None)), [*_swap_ops(), _branch(V("$advance"), _advance_ops(seat, 1))]),
        ),
        # ── 普通数字：进 1 + 抢牌窗口（0/7 在 seven_zero 下已由上面分支消化）──
        _branch(
            AND(
                {"contains": [V("$constants.number_symbols"), CALL("symbol_of", V("$card"))]},
                NOT(sym0),
                NOT(sym7),
            ),
            _branch(V("$advance"), _jump_window_ops(seat)),
        ),
    ]


def _effectors():
    e: dict = {}
    e["do_play_card"] = {"description": "核心出牌结算（打出/抢牌/叠加/7-0 共用）", "ops": _do_play_card()}
    e["do_set_one_hand"] = {
        "description": "把 $hand 重绑到 hand_{$pid}（0 牌移交循环用，克隆安全）",
        "ops": [_set_array({"template": "hand_{$pid}"}, V("$hand"))],
    }
    # ── 玩家动作 ──
    e["do_play"] = {
        "description": "打出普通牌",
        "ops": [
            _call_effect(
                "do_play_card",
                {"player": V("$env.turn"), "card": V("$card"), "color": C(None), "swapTarget": C(None), "advance": C(True)},
            ),
            _call_effect("do_end_check"),
        ],
    }
    e["do_play_wild"] = {
        "description": "打出万能牌（选色）",
        "ops": [
            _call_effect(
                "do_play_card",
                {"player": V("$env.turn"), "card": V("$card"), "color": V("$color"), "swapTarget": C(None), "advance": C(True)},
            ),
            _call_effect("do_end_check"),
        ],
    }
    e["do_play7"] = {
        "description": "打出 7 并与目标换手",
        "ops": [
            _call_effect(
                "do_play_card",
                {"player": V("$env.turn"), "card": V("$card"), "color": C(None), "swapTarget": V("$target"), "advance": C(True)},
            ),
            _call_effect("do_end_check"),
        ],
    }
    e["do_jump"] = {
        "description": "抢牌：抢先打出同色同数字牌（随后从抢牌者方向推进）",
        "ops": [
            _call_effect(
                "do_play_card",
                {"player": V("$env.turn"), "card": V("$card"), "color": C(None), "swapTarget": C(None), "advance": C(False)},
            ),
            _set_env("turn", _advance_pid(_seat_of(V("$env.turn")), 1)),
            _set_env("phase", C("play")),
            _call_effect("do_end_check"),
        ],
    }
    e["do_jump_pass"] = {
        "description": "放弃抢牌：轮到下一位候选或回到正常回合",
        "ops": [
            _inc("turnCount", 1),
            _inc("jumpIndex", 1),
            _branch(
                GTE(V("$env.jumpIndex"), COUNT(V("$env.jumpQueue"))),
                [
                    _set_env("turn", V("$env.nextTurn")),
                    _set_env("phase", C("play")),
                ],
                [_set_env("turn", AT(V("$env.jumpQueue"), V("$env.jumpIndex")))],
            ),
            _call_effect("do_end_check"),
        ],
    }
    e["do_stack2"] = {
        "description": "叠打同色 +2（罚牌数 +2，respond 窗口继续）",
        "ops": [
            _call_effect(
                "do_play_card",
                {"player": V("$env.turn"), "card": V("$card"), "color": C(None), "swapTarget": C(None), "advance": C(False)},
            ),
        ],
    }
    e["do_stack4"] = {
        "description": "叠打 +4（选色，罚牌数 +4，respond 窗口继续）",
        "ops": [
            _call_effect(
                "do_play_card",
                {"player": V("$env.turn"), "card": V("$card"), "color": V("$color"), "swapTarget": C(None), "advance": C(False)},
            ),
        ],
    }
    e["do_take_penalty"] = {
        "description": "吃下累计罚牌（respond → penalty_pick 循环）",
        "ops": [
            _inc("turnCount", 1),
            _set_env("penaltyTarget", V("$env.turn")),
            _set_env("phase", C("penalty_pick")),
        ],
    }
    e["do_draw"] = {
        "description": "选择摸牌（进入 pick chance）",
        "ops": [_set_env("phase", C("pick"))],
    }
    e["do_pass"] = {
        "description": "摸牌后过（回合移交）",
        "ops": [
            _inc("turnCount", 1),
            _set_env("drawnCard", C(None)),
            _set_env("turn", _advance_pid(_seat_of(V("$env.turn")), 1)),
            _set_env("phase", C("play")),
            _call_effect("do_end_check"),
        ],
    }
    e["do_play_drawn"] = {
        "description": "打出刚摸的牌",
        "ops": [
            _call_effect(
                "do_play_card",
                {"player": V("$env.turn"), "card": V("$card"), "color": C(None), "swapTarget": C(None), "advance": C(True)},
            ),
            _set_env("drawnCard", C(None)),
            _call_effect("do_end_check"),
        ],
    }
    # ── chance ──
    e["do_deal"] = {
        "description": "发牌：摸 1 张给发牌座位（dealIdx / dealCount 驱动，7 张/人）",
        "ops": [
            _set_env("dealTarget", AT(V("$constants.player_ids"), V("$env.dealIdx"))),
            _append({"template": "hand_{$env.dealTarget}"}, V("outcome")),
            _inc("dealCount", 1),
            _branch(
                EQ(V("$env.dealCount"), C(HAND_SIZE)),
                [
                    _set_env("dealCount", C(0)),
                    _inc("dealIdx", 1),
                    _branch(
                        EQ(V("$env.dealIdx"), V("$constants.player_count")),
                        [_set_env("phase", C("flip"))],
                        [],
                    ),
                ],
                [],
            ),
        ],
    }
    e["do_flip"] = {
        "description": "翻开首张：设台面颜色/符号并处理首张特殊效果（p1 先手；反转=庄家先手）",
        "ops": [
            _append("discard", V("outcome")),
            _set_env("topColor", CALL("color_of", V("outcome"))),
            _set_env("topSymbol", CALL("symbol_of", V("outcome"))),
            _set_env("turn", AT(V("$constants.player_ids"), C(1))),
            _branch(
                EQ(V("$env.topSymbol"), C("reverse")),
                [
                    _set_env("direction", C(-1)),
                    _set_env("turn", AT(V("$constants.player_ids"), C(0))),  # 庄家先手
                    _set_env("phase", C("play")),
                ],
                _branch(
                    EQ(V("$env.topSymbol"), C("skip")),
                    [
                        _set_env("turn", _advance_pid(C(1), 1, C(1))),  # 跳过下家 p1 → 再下一家
                        _set_env("phase", C("play")),
                    ],
                    _branch(
                        EQ(V("$env.topSymbol"), C("wild")),
                        [
                            _set_env("topColor", C("r")),  # 首张万能默认选红
                            _set_env("phase", C("play")),
                        ],
                        _branch(
                            EQ(V("$env.topSymbol"), C("wild4")),
                            [
                                _set_env("topColor", C("r")),
                                _set_env("penaltyTarget", V("$env.turn")),  # p1 吃 4 并跳过
                                _set_env("pendingDraw", C(4)),
                                _set_env("phase", C("penalty_pick")),
                            ],
                            _branch(
                                EQ(V("$env.topSymbol"), C("draw2")),
                                [
                                    _set_env("penaltyTarget", V("$env.turn")),
                                    _set_env("pendingDraw", C(2)),
                                    _set_env("phase", C("penalty_pick")),
                                ],
                                [_set_env("phase", C("play"))],  # 数字/0/7 首张
                            ),
                        ),
                    ),
                ),
            ),
        ],
    }
    e["do_pick"] = {
        "description": "摸牌结算：进手牌；draw_until 摸到可打才停，否则 draw_result",
        "ops": [
            _append({"template": "hand_{$env.turn}"}, V("outcome")),
            _set_env("drawnCard", V("outcome")),
            _branch(
                AND(
                    EQ(V("$constants.draw_until"), C(True)),
                    NOT(CALL("can_play", V("outcome"), CALL("hand_of", V("$env.turn")))),
                    GT(_deck_count(), C(0)),
                ),
                [],  # 继续摸（phase 保持 pick，turn 不变 → 同一 chance 节点重采样）
                [_set_env("phase", C("draw_result"))],
            ),
        ],
    }
    e["do_penalty_pick"] = {
        "description": "罚牌循环：摸 1 张给罚牌目标；累计未清零继续，否则移交回合",
        "ops": [
            _append({"template": "hand_{$env.penaltyTarget}"}, V("outcome")),
            _inc("pendingDraw", -1),
            _branch(
                GT(V("$env.pendingDraw"), C(0)),
                [],
                [
                    _set_env("turn", _advance_pid(_seat_of(V("$env.penaltyTarget")), 1)),
                    _set_env("phase", C("play")),
                    _call_effect("do_end_check"),
                ],
            ),
        ],
    }
    # ── 终局 ──
    e["do_end_check"] = {
        "description": "终局检查：条件成立时把 env.winner 置为手牌最少者（展示/对局记录用）",
        "ops": [_branch({"or": _end_conditions()}, [_set_env("winner", CALL("least_player"))], [])],
    }
    return e


# ── chance / phases ───────────────────────────────────────────────────


def _chance():
    def tmpl(tid, phases, eid):
        return {
            "id": tid,
            "phases": phases,
            "params": {"card": {"view": "card", "domain": {"ref": "undrawn_cards"}}},
            "probability": {"uniform": {"over": "card"}},
            "effectRef": eid,
            "canonicalKey": {"template": f"{tid}:{{outcome}}"},
        }

    return [
        tmpl("deal", ["deal"], "do_deal"),
        tmpl("flip", ["flip"], "do_flip"),
        tmpl("pick", ["pick"], "do_pick"),
        tmpl("penalty_pick", ["penalty_pick"], "do_penalty_pick"),
    ]


def _phases():
    return [
        {"id": "deal", "actions": [], "description": "发牌：每名玩家 7 张（chance）"},
        {"id": "flip", "actions": [], "description": "翻开首张牌（chance）"},
        {"id": "play", "actions": ["play", "play_wild", "play7", "draw"], "description": "玩家：出牌或摸牌"},
        {"id": "pick", "actions": [], "description": "摸牌（chance）"},
        {"id": "draw_result", "actions": ["play_drawn", "pass"], "description": "玩家：可打出刚摸的牌或过"},
        {"id": "jump", "actions": ["jump_play", "jump_pass"], "description": "玩家：抢牌窗口"},
        {"id": "respond", "actions": ["stack2", "stack4", "take_penalty"], "description": "玩家：叠牌或吃罚牌"},
        {"id": "penalty_pick", "actions": [], "description": "罚牌摸取（chance）"},
    ]


# ── aliases（函数）───────────────────────────────────────────────────


def _aliases():
    return {
        "seat_of": {
            "description": "玩家 p 在 player_ids 中的座位索引",
            "params": ["p"],
            "expr": AT(
                FILTER(
                    RANGE(C(0), COUNT(V("$constants.player_ids"))),
                    EQ(AT(V("$constants.player_ids"), V("$i")), V("$p")),
                    as_var="$i",
                ),
                C(0),
            ),
        },
        "seat_after": {
            "description": "player_ids[(seat + steps * dir) % n]——方向化推进",
            "params": ["seat", "steps", "dir"],
            "expr": {"expr": "($seat + $steps * $dir) % $player_count"},
        },
        "rotate_src_seat": {
            "description": "0 牌移交的源座位：(seat - dir) % n",
            "params": ["seat", "dir"],
            "expr": {"expr": "($seat - $dir) % $player_count"},
        },
        "hand_of": {
            "description": "玩家 p 的手牌数组",
            "params": ["p"],
            "expr": {
                "switch": [{"case": pid, "then": V(f"$hand_{pid}")} for pid in _player_ids()],
                "input": V("$p"),
            },
        },
        "color_of": {
            "description": "牌的颜色（万能牌为 none）",
            "params": ["card"],
            "expr": AT(V("$constants.card_color"), V("$card")),
        },
        "symbol_of": {
            "description": "牌的符号（数字/skip/reverse/draw2/wild/wild4）",
            "params": ["card"],
            "expr": AT(V("$constants.card_symbol"), V("$card")),
        },
        "matches_top": {
            "description": "牌与台面匹配：同色或同符号",
            "params": ["card"],
            "expr": OR(
                EQ(CALL("color_of", V("$card")), V("$env.topColor")),
                EQ(CALL("symbol_of", V("$card")), V("$env.topSymbol")),
            ),
        },
        "hand_has_color": {
            "description": "手牌中是否存在颜色 color 的牌",
            "params": ["hand", "color"],
            "expr": ANY(V("$hand"), EQ(CALL("color_of", V("$node")), V("$color"))),
        },
        "can_play": {
            "description": "可打出：匹配台面，或任意万能，或（严格+4：手牌无台面颜色时）万能4",
            "params": ["card", "hand"],
            "expr": OR(
                CALL("matches_top", V("$card")),
                EQ(CALL("symbol_of", V("$card")), C("wild")),
                AND(
                    EQ(CALL("symbol_of", V("$card")), C("wild4")),
                    NOT(
                        AND(
                            EQ(V("$constants.strict_wild4"), C(True)),
                            CALL("hand_has_color", V("$hand"), V("$env.topColor")),
                        )
                    ),
                ),
            ),
        },
        "deck_count": {
            "description": "牌堆剩余张数 = 108 − (所有手牌总数 + 弃牌数)",
            "params": [],
            "expr": {
                "sub": [
                    {"const": 108},
                    {
                        "add": [
                            {"count": {"concat": [V(f"$hand_{pid}") for pid in _player_ids()]}},
                            {"count": V("$discard")},
                        ]
                    },
                ]
            },
        },
        "least_player": {
            "description": "手牌最少的玩家（按座位序取首个；清空者必然唯一最小）",
            "params": [],
            "expr": AT(
                SORT(V("$constants.player_ids"), COUNT(CALL("hand_of", V("$node")))),
                C(0),
            ),
        },
        "game_ended": {
            "description": "终局布尔（terminal / utility.when / env.winner 共用）",
            "params": [],
            "expr": {"or": _end_conditions()},
        },
    }


# ── 组装 ──────────────────────────────────────────────────────────────


def _ground_state():
    arrays: dict = {}
    for pid in _player_ids():
        arrays[f"hand_{pid}"] = {"type": "array", "mutable": True}
    arrays["discard"] = {"type": "array", "mutable": True}
    arrays["env"] = {
        "type": "env",
        "fields": {
            "phase": {"initial": "deal"},
            "turn": {"initial": "p0"},
            "direction": {"initial": 1},
            "topColor": {"initial": None},
            "topSymbol": {"initial": None},
            "lastActor": {"initial": None},
            "dealIdx": {"initial": 0},
            "dealCount": {"initial": 0},
            "dealTarget": {"initial": None},
            "drawnCard": {"initial": None},
            "penaltyTarget": {"initial": None},
            "pendingDraw": {"initial": 0},
            "nextTurn": {"initial": None},
            "jumpQueue": {"initial": []},
            "jumpIndex": {"initial": 0},
            "turnCount": {"initial": 0},
            "winner": {"initial": None},
            "handsSnapshot": {"initial": []},
        },
    }
    return arrays


def build() -> dict:
    return {
        "meta": {
            "name": "uno",
            "version": "5.2",
            "description": (
                "UNO 108 牌 2-10 人（默认 4 人、每手 7 张）；六个声明式变体：classic / "
                "seven_zero(7-0) / jump_in(抢牌) / stacking(+2叠加) / draw_until(摸到能打) / "
                "strict_wild4(严格+4)。手牌部分可观测；回合由 env.turn/direction 纯数据驱动。"
            ),
        },
        "players": [{"id": pid, "type": "player"} for pid in _player_ids()],
        "variants": {
            "variant": "classic",
            "player_count": 4,
            "options": {
                "classic": {},
                "seven_zero": {"constants": {"seven_zero": True}},
                "jump_in": {"constants": {"jump_in": True}},
                "stacking": {"constants": {"stacking": True}},
                "draw_until": {"constants": {"draw_until": True}},
                "strict_wild4": {"constants": {"strict_wild4": True}},
            },
            "player_ids": {
                "map": {
                    "list": {"range": {"from": {"const": 0}, "to": {"var": "$player_count"}}},
                    "as": "$node",
                    "expr": {"template": "p{$node}"},
                }
            },
            "trim_players": True,
            "trim_utility": True,
        },
        "constants": {
            "card_ids": CARD_IDS,
            "card_symbol": SYMBOL_OF,
            "card_color": COLOR_OF,
            "number_symbols": NUMBER_SYMBOLS,
            "jump_symbols": JUMP_SYMBOLS,
            "colors": COLORS,
            "seven_zero": False,
            "jump_in": False,
            "stacking": False,
            "draw_until": False,
            "strict_wild4": False,
        },
        "groundState": _ground_state(),
        "derivedViews": _views(),
        "queries": _queries(),
        "functions": _aliases(),
        "actions": _actions(),
        "effectors": _effectors(),
        "chance": _chance(),
        "phases": _phases(),
        "visibility": _visibility(),
        "terminal": [
            {"id": "hand_empty", "condition": _hand_empty()},
            {"id": "stuck", "condition": _stuck()},
            {"id": "stuck_penalty", "condition": _stuck_penalty()},
            {"id": "max_turns", "condition": _max_turns()},
        ],
        "utility": [
            {
                "player": pid,
                "value": {
                    "if": {"cond": {"eq": [C(pid), {"call": ["least_player"]}]}, "then": 1, "else": -1}
                },
                "when": {"call": ["game_ended"]},
            }
            for pid in _player_ids()
        ],
    }


def gen_rules() -> dict:
    """Generate the UNO rules dict (see module docstring)."""
    return build()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rules/uno.json")
    parser.add_argument("--players", type=int, default=4, help="默认（声明）人数，2..10")
    parser.add_argument("--out", type=str, default="rules/uno.json")
    args = parser.parse_args()
    if not 2 <= args.players <= 10:
        raise SystemExit(f"players={args.players} must satisfy 2 <= players <= 10")

    rules = gen_rules()
    rules["variants"]["player_count"] = args.players
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}  ({len(CARD_IDS)} cards, {rules['meta']['description']})")


if __name__ == "__main__":
    main()