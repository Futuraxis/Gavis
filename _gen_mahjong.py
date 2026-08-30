"""Generate rules/mahjong.json — one JSON serving all variants.

Variants (guangdong/hongzhong/blood/…) × player counts (2/4) are
declared in the JSON's ``variants`` section (v5.2, self-describing): the
engine only *selects* a declared option (variant / player_count) and
evaluates the declared formulas (``player_ids``, ``deal_target``).  The
default player count is **4** (mahjong is a four-seat game); 2 remains a
declared, explicitly selectable option.  Nothing is injected at
construction time and no adapter is required.  Everything else is
static and self-contained: zero builtins, pure expression aliases
(v5.1).
"""

from __future__ import annotations

import json
import sys

# ── Tile data ─────────────────────────────────────────────────────────

SUITS = ["m", "p", "s", "z"]
RANKS = {"m": 9, "p": 9, "s": 9, "z": 7}

TILE_IDS = []
for s in SUITS:
    for r in range(1, RANKS[s] + 1):
        TILE_IDS.extend([f"{s}{r}"] * 4)

# No-honor 108-tile deck (sichuan / changsha): m/p/s 1-9 × 4.
TILE_IDS_108 = []
for s in ("m", "p", "s"):
    for r in range(1, 10):
        TILE_IDS_108.extend([f"{s}{r}"] * 4)

# Changsha 258将: the winning pair must be 2/5/8 of a suit.
PAIR_258 = ["m2", "m5", "m8", "p2", "p5", "p8", "s2", "s5", "s8"]

SUIT_OF = {f"{s}{r}": s for s in SUITS for r in range(1, RANKS[s] + 1)}

CHI_RUNS = [[f"{s}{r}", f"{s}{r + 1}", f"{s}{r + 2}"] for s in ("m", "p", "s") for r in range(1, 8)]

THIRTEEN_ORPHANS = ["m1", "m9", "p1", "p9", "s1", "s9", "z1", "z2", "z3", "z4", "z5", "z6", "z7"]

FAN_PAY = [10, 20, 40, 80, 160, 320, 640, 1280]  # pay_base × 2^(n-1)
FAN_PAY_INTERNATIONAL = list(range(1, 89))

# 长沙番制：小胡 1 番 → 10；大胡 6 番 → 60；番上番（两个及以上大胡）
# 12 番 → 120（含海底捞月/杠上开花叠加）。索引 = fan_sum - 1（do_win
# 已夹取边界；超额番数落到 120）。[10]*5 覆盖 1-5 番，[60]*6 覆盖
# 6-11 番（大胡+小胡组合），[120]*9 覆盖 12 番起（番上番）。
FAN_PAY_CHANGSHA = [10] * 5 + [60] * 6 + [120] * 9

# 长沙杠上开花触发集合（last_action 属此集合即补牌后胡）。
GANG_ACTIONS = ["gang", "gang_concealed", "gang_added"]

PLAYERS4 = ["p0", "p1", "p2", "p3"]


# ── Expression helpers ────────────────────────────────────────────────


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


def GTE(a, b):
    return {"gte": [a, b]}


def LT(a, b):
    return {"lt": [a, b]}


def AND(*args):
    return {"and": list(args)}


def OR(*args):
    return {"or": list(args)}


def NOT(a):
    return {"not": a}


def ADD(a, b):
    return {"add": [a, b]}


def SUB(a, b):
    return {"sub": [a, b]}


def MUL(a, b):
    return {"mul": [a, b]}


def DIV(a, b):
    return {"div": [a, b]}


def IF(cond, then_, else_=None):
    return {"if": {"cond": cond, "then": then_, "else": else_}}


def CALL(name, *args):
    return {"call": [name, *args]}


def COUNT(expr):
    return {"count": expr}


def SUM(expr):
    return {"sum": expr}


def MIN2(a, b):
    """min(a, b) — list min; concat flattens singleton lists.

    Each singleton uses its own ``as`` var so ``$node`` (the outer
    group entity) is never shadowed by the range item.
    """

    def _sing(item, v):
        return MAP(RANGE(C(0), C(1)), item, as_var=v)

    return {"min": CONCAT(_sing(a, "$ma"), _sing(b, "$mb"))}


def MAX2(a, b):
    """max(a, b) — mirrors :func:`MIN2`."""

    def _sing(item, v):
        return MAP(RANGE(C(0), C(1)), item, as_var=v)

    return {"max": CONCAT(_sing(a, "$ma"), _sing(b, "$mb"))}


def FILTER(lst, where, as_var="$node"):
    return {"filter": {"list": lst, "as": as_var, "where": where}}


def MAP(lst, expr, as_var="$node"):
    return {"map": {"list": lst, "as": as_var, "expr": expr}}


def ANY(lst, where, as_var="$node"):
    return {"any": {"list": lst, "as": as_var, "where": where}}


def ALL(lst, where, as_var="$node"):
    return {"all": {"list": lst, "as": as_var, "where": where}}


def RANGE(frm, to):
    return {"range": {"from": frm, "to": to}}


def CONCAT(*items):
    return {"concat": list(items)}


def SINGLE(item):
    """[item] — a singleton list (concat str-joins when any item is scalar)."""
    return MAP(RANGE(C(0), C(1)), item)


def FLAT2(lst_of_lists):
    """Flatten a list of melds (lists of tiles) → tiles.

    ``concat`` flattens one level of its *items*; a single list argument
    would pass through unchanged, so each meld is addressed by ``at``
    (missing melds in partial combos become None and are dropped).
    Five slots (not four): standard form uses 4 melds and the 5th ``at``
    returns None (dropped by concat) — byte-identical for the default;
    the taiwan 16-tile variant (``meld_k = 5``) needs the 5th slot.
    """
    return CONCAT(
        AT(lst_of_lists, C(0)),
        AT(lst_of_lists, C(1)),
        AT(lst_of_lists, C(2)),
        AT(lst_of_lists, C(3)),
        AT(lst_of_lists, C(4)),
    )


def REPEAT(item, n):
    """[item] × n — n must be a LITERAL (wrapped via const)."""
    return MAP(RANGE(C(0), C(n)), item)


def SWITCH(cases, input_):
    return {"switch": [{"case": cv, "then": te} for cv, te in cases], "input": input_}


def GROUP(lst):
    return {"group": {"list": lst}}


def DISTINCT(lst):
    return {"distinct": lst}


# ── Rule idioms ───────────────────────────────────────────────────────


def _idx_of(p_expr):
    """0-based index of a player id within constants.player_ids
    (= count of ids strictly less than it)."""
    return COUNT(FILTER(V("$constants.player_ids"), LT(V("$node"), p_expr)))


def _next_turn_expr():
    """Cyclic next player: player_ids[(idx+1) % n]."""
    i = _idx_of(V("$p"))
    n = COUNT(V("$constants.player_ids"))
    j = ADD(i, C(1))
    return AT(V("$constants.player_ids"), SUB(j, MUL(DIV(j, n), n)))


def _hand_count(hand_expr, tile_expr):
    """Count of ``tile_expr`` in ``hand_expr``.

    The filter binds ``$h`` (not ``$node``) so a ``tile_expr`` that
    references the surrounding ``$node`` (e.g. a group key) still
    resolves to the outer entity.
    """
    return COUNT(FILTER(hand_expr, EQ(V("$h"), tile_expr), as_var="$h"))


def _wild_count(hand_expr):
    """Active wild-tile count for the hand.

    红中麻将 (hongzhong) declares z5 as a universal joker via the
    variants patch (``constants.wild_tile``); every other variant keeps
    the default empty string, and matching against "" never fires — so
    the wild count is structurally zero outside hongzhong.  (The old
    global ``z5`` default leaked a joker into guangdong/blood/taiwan
    win checks — a lone z5 could stand in for any missing tile.)
    """
    return _hand_count(hand_expr, V("$constants.wild_tile"))


def _min_sel_have(group_expr, hand_expr):
    """Σ_g min(sel_g, have_g) over the selected-tile groups."""
    return SUM(
        MAP(
            {"group": {"list": group_expr}},
            MIN2(GET(V("$node"), "count"), _hand_count(hand_expr, GET(V("$node"), "key"))),
        )
    )


def _cover_ok(selected_expr, hand_expr):
    """Selected tiles coverable by hand + active wild: min_sum + wild >= win_tiles."""
    return GTE(
        ADD(_min_sel_have(selected_expr, hand_expr), _wild_count(hand_expr)),
        V("$constants.win_tiles"),
    )


def _cover_prefix(hand_expr):
    """Monotone prefix prune for partial meld sets: the tiles selected so
    far (3k) must be coverable by hand + wild: min_sum + wild >= 3k.
    Adding a meld raises min_sum by ≤3 and the bound by exactly 3, so a
    failing prefix stays failing for every extension (sound prune)."""
    return GTE(
        ADD(_min_sel_have(FLAT2(V("$m")), hand_expr), _wild_count(hand_expr)),
        MUL(C(3), COUNT(V("$m"))),
    )


def MIN(lst):
    """min over a list expression (engine ``min`` primitive)."""
    return {"min": lst}


def _meld_pool(hand_expr):
    """Runs/pungs the hand can supply, once active wild tiles fill gaps.

    A run is a candidate when its three tiles are coverable:
    Σ min(1, have(t)) + wild ≥ 3; a pung needs count(t) + wild ≥ 3.
    Both pools hold *tile lists* (runs as-is; pungs as [k,k,k]) so the
    structure choose's ``FLAT2`` concat sees uniform lists — a key-only
    pung would string-join with the run lists and break coverage (the
    pre-v5.3 pung pool was unusable for pung-structured wins, e.g. 碰碰胡).

    Multiset supply (v5.4+): each run copy consumes one of each tile, and
    a tile can be sourced have(t)+wild times, so a run is suppliable
    min_t(have(t)+wild) times — bounded by the deck's 4 physical copies
    (plus wild).  The chi pool is the CONCAT of four filters: ``supply_k``
    admits every run whose per-tile supply reaches k (k = 1..4), so the
    multiset choose (``dedup: False``) can pick up to four identical
    runs — 一杯口/二杯口 (two identical runs), 三杯口 and above.  Hands
    with fewer copies simply fail the ``where`` coverage check — over-
    supply is harmless because the exact global coverage is always
    re-verified by the choose ``where``.  Pungs never repeat: a deck
    holds ≤4 copies of a kind, so two pungs of one kind (6 copies) are
    impossible.
    """
    wild = _wild_count(hand_expr)

    def supply_at_least(k: int):
        """Per-tile supply reaches k: min_t(have(t) + wild) ≥ k."""
        return GTE(
            MIN(MAP(V("$run"), ADD(_hand_count(hand_expr, V("$node2")), wild), as_var="$node2")),
            C(k),
        )

    chi = CONCAT(
        *[FILTER(V("$constants.chi_runs"), supply_at_least(k), as_var="$run") for k in range(1, 5)]
    )
    # [k,k,k] per pung: the inner range-map binds ``$i`` (its own as-var)
    # so the outer group ``$node`` stays visible — a plain REPEAT would
    # rebind ``$node`` to the range index and blank the key (the MIN2
    # singleton idiom, mirrored here for a triple).
    pung = MAP(
        FILTER(GROUP(hand_expr), GTE(ADD(GET(V("$node"), "count"), wild), C(3))),
        {
            "map": {
                "list": {"range": {"from": {"const": 0}, "to": {"const": 3}}},
                "as": "$i",
                "expr": GET(V("$node"), "key"),
            }
        },
    )
    return CONCAT(chi, pung)


def _pair_pool(hand_expr):
    """Pair candidates — unrestricted (any pairable key).

    Changsha's 258将 restriction is applied only to 小胡 by
    :func:`_pair_pool_258` from the variant branch of ``is_win_hand``:
    大胡 (碰碰胡/清一色/七对/将将胡) are 乱将 and must keep any pair.
    """
    wild = _wild_count(hand_expr)
    return MAP(FILTER(GROUP(hand_expr), GTE(ADD(GET(V("$node"), "count"), wild), C(2))), GET(V("$node"), "key"))


def _pair_pool_258(hand_expr):
    """Pair candidates restricted to 2/5/8 — the changsha 小胡 / sichuan
    将对 whitelist (``constants.pair_258``)."""
    wild = _wild_count(hand_expr)
    pool = FILTER(
        GROUP(hand_expr),
        AND(
            GTE(ADD(GET(V("$node"), "count"), wild), C(2)),
            {"contains": [V("$constants.pair_258"), GET(V("$node"), "key")]},
        ),
    )
    return MAP(pool, GET(V("$node"), "key"))


def _meld_tiles_expr(melds_var):
    """Flatten a player's open-meld list into the covered tile list.

    Mirror of :func:`FLAT2`: each meld's ``tiles`` is pulled by ``at``
    (missing melds → None → dropped by concat).  Five slots cover the
    taiwan 16-tile variant (up to 5 melds) and are a no-op elsewhere.
    """
    return CONCAT(
        GET(AT(melds_var, C(0)), "tiles"),
        GET(AT(melds_var, C(1)), "tiles"),
        GET(AT(melds_var, C(2)), "tiles"),
        GET(AT(melds_var, C(3)), "tiles"),
        GET(AT(melds_var, C(4)), "tiles"),
    )


def _suits_noz_expr(hand_expr):
    """Suits of a hand with honors filtered out (m/p/s only)."""
    return MAP(
        FILTER(hand_expr, NOT(EQ(AT(V("$node"), C(0)), C("z")))),
        AT(V("$node"), C(0)),
    )


def _qidui(hand_expr):
    """Seven pairs (14 tiles): every group of size 2, OR 龙七对 —
    exactly one group of size 4 (a quad read as two pairs) plus five
    pairs.  Pure quad-as-two-pairs is the standard Sichuan/Changsha
    reading; the fan layer distinguishes the two via ``fan_qidui``
    (pure) vs ``fan_longqidui`` (quad form).

    The size check is essential: without it a 2-tile pair hand (from a
    degenerate chi chain) vacuously passes and ``win_self`` becomes
    legal on a non-winning hand.
    """
    return AND(
        EQ(COUNT(hand_expr), C(14)),
        OR(
            ALL(GROUP(hand_expr), EQ(GET(V("$node"), "count"), C(2))),
            AND(
                EQ(COUNT(FILTER(GROUP(hand_expr), EQ(GET(V("$node"), "count"), C(4)))), C(1)),
                EQ(COUNT(FILTER(GROUP(hand_expr), EQ(GET(V("$node"), "count"), C(2)))), C(5)),
            ),
        ),
    )


def _standard_win(hand_expr, pair_pool_fn=_pair_pool):
    """exists ``meld_k`` melds (+ prefix prune) and 1 pair covering the hand.

    ``pair_pool_fn`` selects the pair candidate set (unrestricted, or the
    changsha 258 whitist via :func:`_pair_pool_258` for 小胡).

    ``dedup: False`` — multiset combinations: the meld pool legitimately
    offers duplicate copies of one run (see :func:`_meld_pool`), and a
    一杯口 hand must be able to pick both copies.
    """
    return {
        "choose": {
            "items": _meld_pool(hand_expr),
            "k": V("$constants.meld_k"),
            "as": "$m",
            "dedup": False,
            "prefix": _cover_prefix(hand_expr),
            "where": {
                "choose": {
                    "items": pair_pool_fn(hand_expr),
                    "k": 1,
                    "as": "$p",
                    "where": _cover_ok(CONCAT(FLAT2(V("$m")), REPEAT(AT(V("$p"), C(0)), 2)), hand_expr),
                }
            },
        }
    }


# ── Sections ──────────────────────────────────────────────────────────


def _ground_env_fields():
    return {
        "phase": {"type": "string", "initial": "deal"},
        "turn": {"type": "player_id", "initial": "p0"},
        "actor": {"type": "player_id?", "initial": None},
        "last_discard": {"type": "tile?", "initial": None},
        "last_discarder": {"type": "player_id?", "initial": None},
        "last_drawn": {"type": "tile?", "initial": None},
        "dealer_idx": {"type": "int", "initial": 0},
        "claim_queue": {"type": "list", "initial": []},
        "claim_index": {"type": "int", "initial": 0},
        # 响应优先级阶段（胡>碰/杠>吃）：``win`` → 只问胡；``meld`` → 只问
        # 碰/杠；``chi`` → 只问吃（下家）。整队过一阶段才进入下一阶段。
        "claim_mode": {"type": "string", "initial": "win"},
        "dealt_count": {"type": "int", "initial": 0},
        "wall_count": {"type": "int", "initial": 136},
        "winners": {"type": "list", "initial": []},
        "done": {"type": "list", "initial": []},
        # gang_added 过渡键：晋升的碰 → 杠的 tiles 拷贝（do_gang_added
        # 写入；曾在 groundState 中未声明）。
        "gang_tiles": {"type": "list", "initial": []},
        "fan_pay": {"type": "int", "initial": 0},
        "win_hand": {"type": "list", "initial": []},
        "payoffs": {"type": "list", "initial": []},
        "game_over": {"type": "bool", "initial": False},
        "winner": {"type": "player_id?", "initial": None},
        "last_action": {"type": "string?", "initial": None},
    }


def _ground_state():
    arrays = {}
    for p in PLAYERS4:
        arrays[f"hand_{p}"] = {"type": "array", "mutable": True, "element": "tile"}
        arrays[f"melds_{p}"] = {"type": "array", "mutable": True, "element": "meld"}
        arrays[f"discard_{p}"] = {"type": "array", "mutable": True, "element": "tile"}
    arrays["drawn"] = {"type": "array", "mutable": True, "element": "tile"}
    arrays["env"] = {"type": "env", "fields": _ground_env_fields()}
    return arrays


def _derived_views():
    views = {
        "tile": {"from": {"type": "literal", "list": {"var": "tile_ids"}}, "fields": {"id": V("$self.value")}},
        "player": {
            "from": {"type": "literal", "list": {"var": "$players"}},
            "fields": {"id": V("$self.value.id"), "idx": V("$i")},
        },
    }
    for p in PLAYERS4:
        views[f"hand_view_{p}"] = {"from": {"type": "enum", "array": f"hand_{p}"}, "fields": {"id": V("$self.value")}}
        views[f"meld_view_{p}"] = {"from": {"type": "enum", "array": f"melds_{p}"}, "fields": {"id": V("$self.value")}}
        views[f"discard_view_{p}"] = {
            "from": {"type": "enum", "array": f"discard_{p}"},
            "fields": {"id": V("$self.value")},
        }
    return views


def _queries():
    return {
        "undrawn_tiles": {
            "view": "tile",
            # The view holds 4 physical copies per tile kind but ``drawn``
            # records kinds — a kind stays available while fewer than 4
            # copies have been drawn (uniform over remaining kinds).
            "filter": {
                "lt": [{"count": FILTER(V("$drawn"), EQ(V("$t"), GET(V("$node"), "id")), as_var="$t")}, {"const": 4}]
            },
        }
    }


CLAIM_ACTOR = AT(V("$env.claim_queue"), V("$env.claim_index"))


def _actions():
    return [
        {
            "id": "discard",
            "type": "move",
            # ``discard`` phase — 吃/碰后的强制出牌（不摸牌），仅剩打牌可选。
            "phases": ["action", "discard"],
            "actor": V("$env.turn"),
            "params": {"tile": {"domain": {"array": {"template": "hand_{$env.turn}"}}}},
            "legal": C(True),
            "effectRef": "do_discard",
            "canonicalKey": {"template": "discard:{tile}"},
        },
        {
            "id": "win_self",
            "type": "move",
            "phases": ["action"],
            "actor": V("$env.turn"),
            "params": {},
            # The action phase already includes the drawn tile in the hand
            # (do_draw appends before setting phase=action), so the win check
            # runs on the real 14 tiles -- appending last_drawn again would
            # double-count it (15 tiles: seven pairs / thirteen orphans never
            # legal, and a 14-tile cover could be faked with the duplicate).
            # Meld-aware: the player's open melds count towards the structure.
            # Single-win guard: a player who already won may not win again
            # (blood winners keep playing but cannot re-win; elsewhere the
            # game ends on the first win so the guard is inert).
            "legal": AND(
                NOT({"contains": [V("$env.winners"), V("$env.turn")]}),
                CALL("is_win_hand", CALL("hand_of", V("$env.turn")), CALL("melds_of", V("$env.turn"))),
            ),
            "effectRef": "do_win_self",
            "canonicalKey": {"const": "win_self"},
        },
        {
            "id": "gang_concealed",
            "type": "move",
            "phases": ["action"],
            "actor": V("$env.turn"),
            "params": {"tile": {"domain": {"array": {"template": "hand_{$env.turn}"}}}},
            "legal": EQ(COUNT(FILTER(CALL("hand_of", V("$env.turn")), EQ(V("$node"), V("tile")))), C(4)),
            "effectRef": "do_gang_concealed",
            "canonicalKey": {"template": "gang_concealed:{tile}"},
        },
        {
            "id": "gang_added",
            "type": "move",
            "phases": ["action"],
            "actor": V("$env.turn"),
            "params": {"tile": {"domain": {"array": {"template": "hand_{$env.turn}"}}}},
            "legal": AND(
                GTE(COUNT(FILTER(CALL("hand_of", V("$env.turn")), EQ(V("$node"), V("tile")))), C(1)),
                ANY(
                    CALL("melds_of", V("$env.turn")),
                    AND(EQ(GET(V("$node"), "type"), C("peng")), EQ(AT(GET(V("$node"), "tiles"), C(0)), V("tile"))),
                ),
            ),
            "effectRef": "do_gang_added",
            "canonicalKey": {"template": "gang_added:{tile}"},
        },
        {
            "id": "claim_win",
            "type": "move",
            "phases": ["claim"],
            "actor": CLAIM_ACTOR,
            "params": {"tile": {"domain": {"expr": SINGLE(V("$env.last_discard"))}}},
            # Single-win guard mirrors win_self: a previous winner may not
            # ron again (blood winners still chi/peng/gang via the queue).
            # Stage gate: wins resolve in the ``win`` stage (胡>碰>吃 — a
            # later seat's win preempts any earlier 吃/碰).
            "legal": AND(
                EQ(V("$env.claim_mode"), C("win")),
                NOT({"contains": [V("$env.winners"), CLAIM_ACTOR]}),
                CALL(
                    "is_win_hand",
                    CONCAT(CALL("hand_of", CLAIM_ACTOR), SINGLE(V("tile"))),
                    CALL("melds_of", CLAIM_ACTOR),
                ),
            ),
            "effectRef": "do_claim_win",
            "canonicalKey": {"template": "claim_win:{tile}"},
        },
        {
            "id": "claim_peng",
            "type": "move",
            "phases": ["claim"],
            "actor": CLAIM_ACTOR,
            "params": {"tile": {"domain": {"expr": SINGLE(V("$env.last_discard"))}}},
            # Stage gate: 碰/杠 resolve in the ``meld`` stage — any player's
            # meld beats a later 吃 (碰>吃), and a win already had its chance.
            "legal": AND(
                EQ(V("$env.claim_mode"), C("meld")),
                GTE(COUNT(FILTER(CALL("hand_of", CLAIM_ACTOR), EQ(V("$node"), V("tile")))), C(2)),
            ),
            "effectRef": "do_claim_peng",
            "canonicalKey": {"template": "claim_peng:{tile}"},
        },
        {
            "id": "claim_gang",
            "type": "move",
            "phases": ["claim"],
            "actor": CLAIM_ACTOR,
            "params": {"tile": {"domain": {"expr": SINGLE(V("$env.last_discard"))}}},
            "legal": AND(
                EQ(V("$env.claim_mode"), C("meld")),
                GTE(COUNT(FILTER(CALL("hand_of", CLAIM_ACTOR), EQ(V("$node"), V("tile")))), C(3)),
            ),
            "effectRef": "do_claim_gang",
            "canonicalKey": {"template": "claim_gang:{tile}"},
        },
        {
            "id": "claim_chi",
            "type": "move",
            "phases": ["claim"],
            "actor": CLAIM_ACTOR,
            "params": {
                "tiles": {
                    "domain": {
                        "expr": FILTER(V("$constants.chi_runs"), {"contains": [V("$node"), V("$env.last_discard")]})
                    }
                }
            },
            # Chi is banned in the no-chi Sichuan family — sichuan (血战
            # 到底), changsha (长沙麻将只能碰杠), blood (血流成河) — and
            # elsewhere it is legal only for the first responder AND only
            # when the two non-discard tiles of the chosen run are
            # actually in the claimant's hand — without this gate, bogus
            # chi conjures melds from thin air and hands shrink
            # erratically (the observed "吃规则混乱").
            # 吃只在 chi 阶段（胡/碰/杠阶段过后）对首个响应者开放：任何后位
            # 的碰/杠（meld 阶段）与任意胡（win 阶段）都优先于下家的吃。
            "legal": AND(
                EQ(V("$env.claim_mode"), C("chi")),
                NOT(
                    OR(
                        EQ(V("$constants.variant"), C("sichuan")),
                        EQ(V("$constants.variant"), C("changsha")),
                        EQ(V("$constants.variant"), C("blood")),
                    )
                ),
                EQ(V("$env.claim_index"), C(0)),
                GTE(COUNT(CALL("hand_of", CLAIM_ACTOR)), C(2)),
                ALL(
                    FILTER(
                        V("tiles"),
                        NOT(EQ(V("$node"), V("$env.last_discard"))),
                    ),
                    GTE(
                        COUNT(FILTER(CALL("hand_of", CLAIM_ACTOR), EQ(V("$h"), V("$ct")), as_var="$h")),
                        C(1),
                    ),
                    as_var="$ct",
                ),
            ),
            "effectRef": "do_claim_chi",
            "canonicalKey": {"template": "claim_chi:{tiles}"},
        },
        {
            "id": "claim_pass",
            "type": "move",
            "phases": ["claim"],
            "actor": CLAIM_ACTOR,
            "params": {},
            "legal": C(True),
            "effectRef": "do_claim_pass",
            "canonicalKey": {"const": "claim_pass"},
        },
    ]


def _set_env(key, value):
    return {"op": "setEnv", "key": key, "value": value}


def _append(arr_name, value):
    return {"op": "append", "array": arr_name, "value": value}


def _remove(arr_name, value, count=1):
    return {"op": "remove", "array": arr_name, "value": value, "count": {"const": count}}


def _branch(cond, then_ops, else_ops=None):
    return {"op": "branch", "if": cond, "then": then_ops, "else": else_ops if else_ops is not None else []}


def _inc(key, by):
    return {"op": "inc", "key": key, "by": by}


def _call_effect(ref, args=None):
    return {"op": "callEffect", "effectRef": ref, "args": args or {}}


def _end_game(action_label=None):
    """Ops that end the round: set game_over + phase + last_action."""
    ops = [_set_env("game_over", C(True)), _set_env("phase", C("game_over"))]
    if action_label is not None:
        ops.append(_set_env("last_action", C(action_label)))
    return ops


def _effectors():
    # Wall empty: the classic counter at zero OR the deck exhausted
    # (|drawn| == |tile_ids| — each tile kind is drawn at most 4 times, so
    # this holds exactly when every tile is taken).  Deck-reduced variants
    # (sichuan/changsha 108 no-honor) end here instead of running dry;
    # for the default 136-tile deck the counter reaches 0 exactly then.
    # ``drawn`` is a ground array → bound as ``$drawn``; ``$env.drawn``
    # is always None (dead arm in the draft baseline — count(None) == 0,
    # so the deck-exhausted arm never fired and 108-tile variants could
    # never end on an empty wall).
    wall_empty = OR(
        EQ(V("$env.wall_count"), C(0)),
        GTE(COUNT(V("$drawn")), COUNT(V("$constants.tile_ids"))),
    )
    turn_hand = {"template": "hand_{$env.turn}"}
    actor = CLAIM_ACTOR
    actor_hand_tmpl = {"template": "hand_{$env.actor}"}
    actor_melds_tmpl = {"template": "melds_{$env.actor}"}
    to_gang_draw = [
        _branch(wall_empty, _end_game("wall_empty"), [_set_env("phase", C("gang_draw"))]),
    ]

    return {
        "to_draw": {
            "description": "Advance to the draw phase (or end on an empty wall)",
            "ops": [
                _branch(wall_empty, _end_game("wall_empty"), [_set_env("phase", C("draw"))]),
            ],
        },
        "do_draw": {
            "description": "Deal / draw / gang-draw: one tile from the wall",
            "ops": [
                _branch(
                    EQ(V("$env.phase"), C("deal")),
                    [
                        _append(turn_hand, V("outcome")),
                        _append("drawn", V("outcome")),
                        _inc("wall_count", -1),
                        _inc("dealt_count", 1),
                        _set_env("last_drawn", V("outcome")),
                        _set_env("turn", CALL("next_turn", V("$env.turn"))),
                        _branch(
                            GTE(V("$env.dealt_count"), V("$constants.deal_target")),
                            [
                                _set_env("phase", C("action")),
                                _set_env("turn", C("p0")),
                                _set_env("dealt_count", C(0)),
                                _set_env("last_action", C("deal_done")),
                                # zero-score baseline — a wall-empty end settles 0
                                _set_env("payoffs", MAP(V("$players"), C(0))),
                            ],
                            [],
                        ),
                    ],
                    [
                        _append(turn_hand, V("outcome")),
                        _append("drawn", V("outcome")),
                        _inc("wall_count", -1),
                        _set_env("last_drawn", V("outcome")),
                        _set_env("phase", C("action")),
                    ],
                ),
            ],
        },
        "do_discard": {
            "description": "Discard a tile, open the claim queue",
            "ops": [
                _remove(turn_hand, V("tile")),
                _append({"template": "discard_{$env.turn}"}, V("tile")),
                _set_env("last_discard", V("tile")),
                _set_env("last_discarder", V("$env.turn")),
                _set_env(
                    "dealer_idx",
                    GET(
                        AT(FILTER({"query": {"view": "player"}}, EQ(GET(V("$node"), "id"), V("$env.turn"))), C(0)),
                        "idx",
                    ),
                ),
                _set_env(
                    "claim_queue",
                    MAP(
                        {
                            "sort": {
                                "list": FILTER(
                                    {"query": {"view": "player"}},
                                    AND(
                                        NOT({"contains": [V("$env.done"), GET(V("$node"), "id")]}),
                                        {"not": {"eq": [GET(V("$node"), "id"), V("$env.last_discarder")]}},
                                    ),
                                ),
                                "by": {"expr": "(node.idx - $env.dealer_idx + 4) % 4"},
                            }
                        },
                        GET(V("$node"), "id"),
                    ),
                ),
                _set_env("claim_index", C(0)),
                # 响应从 win 阶段开始（胡>碰/杠>吃）。
                _set_env("claim_mode", C("win")),
                # Claim-phase actor is the queue head — ``env.turn`` is set
                # to CLAIM_ACTOR so the base engine's ``get_current_player``
                # needs no adapter override (v5.2 declarative rotation).
                _set_env("turn", CLAIM_ACTOR),
                _set_env("last_action", C("discard")),
                _set_env("phase", C("claim")),
            ],
        },
        "do_claim_pass": {
            "description": "Pass the claim; advance the priority stage (胡>碰/杠>吃), open the draw once all pass",
            "ops": [
                _inc("claim_index", 1),
                _branch(
                    GTE(V("$env.claim_index"), COUNT(V("$env.claim_queue"))),
                    [
                        # 整队放弃当前阶段 → 进入下一优先级阶段（胡 → 碰/杠 →
                        # 吃）；chi 阶段也无人认领时弃牌作废，回到下家摸牌。
                        _set_env("claim_index", C(0)),
                        _branch(
                            EQ(V("$env.claim_mode"), C("win")),
                            [
                                _set_env("claim_mode", C("meld")),
                                _set_env("last_action", C("pass_win_stage")),
                                _set_env("turn", CLAIM_ACTOR),
                            ],
                            [
                                _branch(
                                    EQ(V("$env.claim_mode"), C("meld")),
                                    [
                                        _set_env("claim_mode", C("chi")),
                                        _set_env("last_action", C("pass_meld_stage")),
                                        _set_env("turn", CLAIM_ACTOR),
                                    ],
                                    [
                                        _set_env("last_action", C("pass_all")),
                                        _set_env("turn", AT(V("$env.claim_queue"), C(0))),
                                        _set_env("claim_queue", C([])),
                                        _set_env("claim_index", C(0)),
                                        _call_effect("to_draw"),
                                    ],
                                )
                            ],
                        ),
                    ],
                    [
                        _set_env("last_action", C("pass")),
                        # Next responder becomes the acting player (the queue
                        # head), keeping ``env.turn`` aligned with CLAIM_ACTOR.
                        _set_env("turn", CLAIM_ACTOR),
                    ],
                ),
            ],
        },
        "do_claim_peng": {
            "description": "Pung: take the discard, meld a triplet, then discard immediately (no draw)",
            "ops": [
                _set_env("actor", actor),
                _remove(actor_hand_tmpl, V("tile"), 2),
                _append(
                    actor_melds_tmpl, {"type": "peng", "tiles": REPEAT(V("tile"), 3), "from": V("$env.last_discarder")}
                ),
                _set_env("turn", V("$env.actor")),
                _set_env("claim_queue", C([])),
                _set_env("claim_index", C(0)),
                _set_env("last_action", C("peng")),
                # 标准麻将：吃/碰后直接打出一张牌（不摸牌）。旧实现进入 draw，
                # 使吃/碰牌者每副露永久多一张牌 → 手牌张数错乱（14/15 张）且
                # 自摸/荣和结构永远凑不齐。
                _set_env("phase", C("discard")),
            ],
        },
        "do_claim_gang": {
            "description": "Exposed gang: take the discard, meld quads",
            "ops": [
                _set_env("actor", actor),
                _remove(actor_hand_tmpl, V("tile"), 3),
                _append(
                    actor_melds_tmpl, {"type": "gang", "tiles": REPEAT(V("tile"), 4), "from": V("$env.last_discarder")}
                ),
                _set_env("turn", V("$env.actor")),
                _set_env("claim_queue", C([])),
                _set_env("claim_index", C(0)),
                _set_env("last_action", C("gang")),
                *to_gang_draw,
            ],
        },
        "do_claim_chi": {
            "description": "Chi (first responder only): meld a run, then discard immediately (no draw)",
            "ops": [
                _set_env("actor", actor),
                # 只从手牌移除顺子中除弃牌外的两张：弃牌已由打出方打出、不属于
                # 本家手牌，本家自己持有的同名牌必须保留（旧实现把整副顺子
                # 3 张都从手牌 remove，导致手牌 13→10 且副露凭空多一张）。
                {
                    "op": "forEach",
                    "list": FILTER(V("tiles"), NOT(EQ(V("$node"), V("$env.last_discard")))),
                    "do": [_remove(actor_hand_tmpl, V("$item"))],
                },
                _append(actor_melds_tmpl, {"type": "chi", "tiles": V("tiles")}),
                _set_env("turn", V("$env.actor")),
                _set_env("claim_queue", C([])),
                _set_env("claim_index", C(0)),
                _set_env("last_action", C("chi")),
                # 吃后直接出牌（不摸牌），保持 13 张在手的不变量。
                _set_env("phase", C("discard")),
            ],
        },
        "do_claim_win": {
            "description": "Win off a discard (ron)",
            "ops": [
                _call_effect("do_win", {"pid": CLAIM_ACTOR, "tile": V("$env.last_discard"), "self_win": C(False)}),
            ],
        },
        "do_win_self": {
            "description": "Win on the self-drawn tile (tsumo)",
            "ops": [
                _call_effect("do_win", {"pid": V("$env.turn"), "tile": V("$env.last_drawn"), "self_win": C(True)}),
            ],
        },
        "do_gang_concealed": {
            "description": "Concealed gang from four held tiles",
            "ops": [
                {"op": "forEach", "list": RANGE(C(0), C(4)), "do": [_remove(turn_hand, V("tile"))]},
                _append({"template": "melds_{$env.turn}"}, {"type": "concealed_gang", "tiles": REPEAT(V("tile"), 4)}),
                _set_env("last_action", C("gang_concealed")),
                *to_gang_draw,
            ],
        },
        "do_gang_added": {
            "description": "Added gang: promote a pung to a quad",
            "ops": [
                _remove(turn_hand, V("tile")),
                _set_env(
                    "gang_tiles",
                    GET(
                        AT(
                            FILTER(
                                CALL("melds_of", V("$env.turn")),
                                AND(
                                    EQ(GET(V("$node"), "type"), C("peng")),
                                    EQ(AT(GET(V("$node"), "tiles"), C(0)), V("tile")),
                                ),
                            ),
                            C(0),
                        ),
                        "tiles",
                    ),
                ),
                {
                    "op": "setArray",
                    "array": {"template": "melds_{$env.turn}"},
                    "value": FILTER(
                        CALL("melds_of", V("$env.turn")),
                        NOT(
                            AND(
                                EQ(GET(V("$node"), "type"), C("peng")),
                                EQ(AT(GET(V("$node"), "tiles"), C(0)), V("tile")),
                            )
                        ),
                    ),
                },
                _append(
                    {"template": "melds_{$env.turn}"},
                    {"type": "added_gang", "tiles": CONCAT(V("$env.gang_tiles"), SINGLE(V("tile")))},
                ),
                _set_env("last_action", C("gang_added")),
                *to_gang_draw,
            ],
        },
        "do_win": {
            "description": "Common win settlement: fans, winners, continue or end",
            "ops": [
                # win_hand = 手牌（自摸已含摸牌；荣和补弃牌）∪ 副露牌 ——
                # 计番与胡牌结构同口径（副露碰碰胡此前只按暗手计 1 番；
                # 副露混入的牌也参与清一色/混一色判定）。门清时副露为空，
                # 结果与纯手牌一致。
                _set_env(
                    "win_hand",
                    CONCAT(
                        IF(
                            V("self_win"),
                            CALL("hand_of", V("pid")),
                            CONCAT(CALL("hand_of", V("pid")), SINGLE(V("tile"))),
                        ),
                        IF(
                            EQ(COUNT(CALL("melds_of", V("pid"))), C(0)),
                            C([]),
                            _meld_tiles_expr(CALL("melds_of", V("pid"))),
                        ),
                    ),
                ),
                # Clamp the fan table index: `at` returns None out of
                # range, and a None fan_pay crashes payoff arithmetic.
                _set_env(
                    "fan_pay",
                    AT(
                        V("$constants.fan_pay"),
                        MAX2(
                            C(0),
                            MIN2(
                                SUB(
                                    CALL("fan_sum", V("$env.win_hand"), V("pid"), V("self_win")),
                                    C(1),
                                ),
                                SUB(COUNT(V("$constants.fan_pay")), C(1)),
                            ),
                        ),
                    ),
                ),
                _branch(NOT({"contains": [V("$env.winners"), V("pid")]}), [_append("winners", V("pid"))], []),
                # done = 退场名单，仅 sichuan（血战到底胡家退场）追加；
                # blood（血流成河）胡家继续摸打，靠 winners 守卫禁止再胡
                # （win_self/claim_win 的 legality 均含 NOT contains(winners)）。
                _branch(
                    AND(
                        EQ(V("$constants.variant"), C("sichuan")),
                        NOT({"contains": [V("$env.done"), V("pid")]}),
                    ),
                    [_append("done", V("pid"))],
                    [],
                ),
                _set_env("last_action", IF(V("self_win"), C("win_self"), C("win_discard"))),
                _branch(
                    OR(EQ(V("$constants.variant"), C("blood")), EQ(V("$constants.variant"), C("sichuan"))),
                    [
                        _branch(
                            OR(
                                # 血战到底/血流成河统一终局：player_count-1 家
                                # 胡过（4p → 三家胡；2p → 首胡即终局，只剩
                                # 一家无以为继）或牌墙抽干。
                                GTE(COUNT(V("$env.winners")), SUB(V("$constants.player_count"), C(1))),
                                wall_empty,
                            ),
                            _end_game("blood_over"),
                            [
                                _set_env(
                                    "turn",
                                    IF(
                                        {"contains": [V("$env.done"), CALL("next_turn", V("pid"))]},
                                        CALL("next_turn", CALL("next_turn", V("pid"))),
                                        CALL("next_turn", V("pid")),
                                    ),
                                ),
                                _call_effect("to_draw"),
                            ],
                        ),
                    ],
                    [
                        *_end_game(),
                        _set_env("winner", V("pid")),
                    ],
                ),
                _set_env("last_winner", V("pid")),
                _set_env(
                    "payoffs",
                    MAP(V("$players"), CALL("payoff_after", GET(V("$node"), "id"))),
                ),
            ],
        },
    }


def _chance():
    return [
        {
            "id": "draw",
            "phases": ["deal", "draw", "gang_draw"],
            "params": {"tile": {"view": "tile", "domain": {"ref": "undrawn_tiles"}}},
            "probability": {"uniform": {"over": "tile"}},
            "effectRef": "do_draw",
            "canonicalKey": {"template": "draw:{outcome}"},
        }
    ]


def _phases():
    return [
        {"id": "deal", "actions": [], "description": "Chance: deal 13N+1 tiles"},
        {
            "id": "action",
            "actions": ["discard", "win_self", "gang_concealed", "gang_added"],
            "description": "Player: discard / win / gang",
        },
        {
            "id": "discard",
            "actions": ["discard"],
            "description": "Player: forced discard right after a chi/peng claim (no draw)",
        },
        {
            "id": "claim",
            "actions": ["claim_win", "claim_peng", "claim_gang", "claim_chi", "claim_pass"],
            "description": "Player: respond to a discard",
        },
        {"id": "draw", "actions": [], "description": "Chance: draw one tile"},
        {"id": "gang_draw", "actions": [], "description": "Chance: gang replacement"},
        {"id": "game_over", "actions": [], "description": "Round finished"},
    ]


def _aliases():
    hand = V("$hand")
    suits = MAP(hand, AT(V("$node"), C(0)))
    suits_noz = MAP(FILTER(hand, NOT(EQ(AT(V("$node"), C(0)), C("z")))), AT(V("$node"), C(0)))
    # 副露感知：胡牌检查用 手牌 ∪ 副露牌（吃/碰/杠已亮出的牌计入 14/17 张
    # 结构）。七对/十三幺/呖咕呖咕 仍只用门清的 ``hand``（见 is_win_hand）。
    # 无副露时直接用空列表而非展开（全 None 的 concat 会退化成字符串 ""）。
    full = CONCAT(hand, IF(EQ(COUNT(V("$melds")), C(0)), C([]), _meld_tiles_expr(V("$melds"))))
    fans = {
        "fan_jihu": {"description": "鸡胡", "params": ["hand"], "expr": C(1)},
        "fan_pinghu": {
            "description": "平胡 (approx: no triplets)",
            "params": ["hand"],
            "expr": EQ(COUNT(FILTER(GROUP(hand), GTE(GET(V("$node"), "count"), C(3)))), C(0)),
        },
        "fan_pengpenghu": {
            "description": "碰碰胡 (approx: meld_k triplets — 4 for 14-tile, 5 for taiwan 17-tile)",
            "params": ["hand"],
            "expr": EQ(COUNT(FILTER(GROUP(hand), GTE(GET(V("$node"), "count"), C(3)))), V("$constants.meld_k")),
        },
        # 清一色 = m/p/s 单花色且无字牌——此前按首字符匹配，纯字牌手
        # （字一色）会被误判为清一色。
        "fan_qingyise": {
            "description": "清一色 (one of m/p/s, NO honors)",
            "params": ["hand"],
            "expr": AND(
                EQ(COUNT(DISTINCT(suits_noz)), C(1)),
                EQ(COUNT(suits_noz), COUNT(hand)),
            ),
        },
        "fan_hunyise": {
            "description": "混一色 (approx: honors + one suit)",
            "params": ["hand"],
            "expr": AND(ANY(hand, EQ(AT(V("$node"), C(0)), C("z"))), EQ(COUNT(DISTINCT(suits_noz)), C(1))),
        },
        # 七对番只认纯七对（全 size-2 组）；龙七对（一组四张+五对）由
        # fan_longqidui 独立计番（四川 8 番），不再 4+8 重复计。
        "fan_qidui": {
            "description": "七对 (pure pairs — quad form scores via fan_longqidui)",
            "params": ["hand"],
            "expr": AND(EQ(COUNT(hand), C(14)), ALL(GROUP(hand), EQ(GET(V("$node"), "count"), C(2)))),
        },
        # 十三幺双向判定：幺九全集 ⊆ 手牌 且 手牌 ⊆ 幺九——此前只有正向，
        # "13 幺九 + 1 普通牌"也能骗过（fan 层同样收紧）。
        "fan_shisanyao": {
            "description": "十三幺 (bidirectional: orphans ⊆ hand AND hand ⊆ orphans)",
            "params": ["hand"],
            "expr": AND(
                ALL(V("$constants.thirteen_orphans"), {"contains": [hand, V("$node")]}),
                ALL(hand, {"contains": [V("$constants.thirteen_orphans"), V("$node")]}),
            ),
        },
        "fan_hongzhongke": {
            "description": "红中刻 (hongzhong only)",
            "params": ["hand"],
            "expr": IF(
                EQ(V("$constants.variant"), C("hongzhong")),
                GTE(COUNT(FILTER(hand, EQ(V("$node"), V("$constants.wild_tile")))), C(3)),
                C(0),
            ),
        },
        "fan_jueshang": {
            "description": "缺一门 (blood only, approx)",
            "params": ["hand"],
            "expr": IF(EQ(V("$constants.variant"), C("blood")), EQ(COUNT(DISTINCT(suits)), C(2)), C(0)),
        },
        "fan_longqidui": {
            "description": "龙七对: one quad + five pairs (14 tiles)",
            "params": ["hand"],
            "expr": AND(
                EQ(COUNT(hand), C(14)),
                EQ(COUNT(FILTER(GROUP(hand), EQ(GET(V("$node"), "count"), C(4)))), C(1)),
                EQ(COUNT(FILTER(GROUP(hand), EQ(GET(V("$node"), "count"), C(2)))), C(5)),
            ),
        },
        "fan_jiangdui": {
            "description": "将对: four triplets + pair, every tile 2/5/8",
            "params": ["hand"],
            "expr": AND(
                EQ(COUNT(FILTER(GROUP(hand), GTE(GET(V("$node"), "count"), C(3)))), C(4)),
                ALL(hand, {"contains": [V("$constants.pair_258"), V("$node")]}),
            ),
        },
        "fan_jiangjianghu": {
            "description": "将将胡: 14 tiles, every tile 2/5/8, structure-exempt (changsha)",
            "params": ["hand"],
            "expr": AND(
                EQ(COUNT(hand), C(14)),
                ALL(hand, {"contains": [V("$constants.pair_258"), V("$node")]}),
            ),
        },
        # 海底捞月：牌墙抽干时胡牌。wall_count 初值恒为 136，108 张牌型
        # （sichuan/blood/changsha）抽干时 wall_count=28≠0 永不触发——改用
        # 与 effectors wall_empty 同款判定（wall_count==0 或 drawn ≥ 牌池）。
        "fan_haidilaoyue": {
            "description": "海底捞月 (changsha 6 / sichuan·blood 8: win with the wall exhausted)",
            "params": ["hand"],
            "expr": IF(
                OR(
                    EQ(V("$constants.variant"), C("changsha")),
                    EQ(V("$constants.variant"), C("sichuan")),
                    EQ(V("$constants.variant"), C("blood")),
                ),
                OR(
                    EQ(V("$env.wall_count"), C(0)),
                    GTE(COUNT(V("$drawn")), COUNT(V("$constants.tile_ids"))),
                ),
                C(0),
            ),
        },
        "fan_gangshangkaihua": {
            "description": "杠上开花 (changsha 6 / sichuan 8: win right after a gang action)",
            "params": ["hand"],
            "expr": IF(
                OR(EQ(V("$constants.variant"), C("changsha")), EQ(V("$constants.variant"), C("sichuan"))),
                {"contains": [V("$constants.gang_actions"), V("$env.last_action")]},
                C(0),
            ),
        },
        "fan_menqing": {
            "description": "门清 (taiwan/international: no open melds)",
            "params": ["pid"],
            "expr": IF(
                OR(EQ(V("$constants.variant"), C("taiwan")), EQ(V("$constants.variant"), C("international"))),
                EQ(COUNT(CALL("melds_of", V("$pid"))), C(0)),
                C(0),
            ),
        },
        "fan_zimo": {
            "description": "自摸 (taiwan/international: self-drawn win)",
            "params": ["self_win"],
            "expr": IF(
                OR(EQ(V("$constants.variant"), C("taiwan")), EQ(V("$constants.variant"), C("international"))),
                V("$self_win"),
                C(0),
            ),
        },
    }
    fan_order = [
        "fan_jihu",
        "fan_pinghu",
        "fan_pengpenghu",
        "fan_qingyise",
        "fan_hunyise",
        "fan_qidui",
        "fan_shisanyao",
        "fan_hongzhongke",
        "fan_jueshang",
        "fan_longqidui",
        "fan_jiangdui",
        "fan_jiangjianghu",
        "fan_haidilaoyue",
        "fan_gangshangkaihua",
        "fan_menqing",
        "fan_zimo",
    ]
    # Alias parameter lists: most fans read the hand; 门清 needs the pid
    # (melds_of) and 自摸 needs the self_win flag (taiwan 台数).
    fan_params = {"fan_menqing": ["pid"], "fan_zimo": ["self_win"]}
    # Per-variant fan values (番/台).  Conditions stay in the fan aliases;
    # the VALUE is mapped here per variant (v5.3, documented approximation
    # where platform tables differ).  ``default`` covers guangdong /
    # hongzhong / blood.  jiangjianghu never *adds* to itself (changsha
    # 大胡 6 番) and qidui/jiangjianghu overlap → 番上番 12 番.
    fan_values_variant = {
        "fan_jihu": {"sichuan": 0, "changsha": 0, "taiwan": 0, "default": 1},
        "fan_pinghu": {"sichuan": 1, "changsha": 1, "taiwan": 2, "international": 2, "default": 2},
        "fan_pengpenghu": {"sichuan": 2, "changsha": 6, "taiwan": 4, "international": 6, "default": 3},
        "fan_qingyise": {"sichuan": 4, "changsha": 6, "taiwan": 8, "international": 24, "default": 5},
        "fan_hunyise": {"sichuan": 0, "changsha": 0, "taiwan": 4, "international": 6, "default": 2},
        "fan_qidui": {"sichuan": 4, "changsha": 6, "taiwan": 0, "international": 24, "default": 4},
        "fan_shisanyao": {"sichuan": 0, "changsha": 0, "taiwan": 0, "international": 88, "default": 8},
        "fan_hongzhongke": {"default": 1},  # expr already hongzhong-gated
        "fan_jueshang": {"sichuan": 0, "changsha": 0, "taiwan": 0, "default": 1},
        "fan_longqidui": {"sichuan": 8, "default": 0},
        "fan_jiangdui": {"sichuan": 8, "default": 0},
        "fan_jiangjianghu": {"changsha": 6, "default": 0},
        "fan_haidilaoyue": {"changsha": 6, "sichuan": 8, "blood": 8, "default": 0},
        "fan_gangshangkaihua": {"changsha": 6, "sichuan": 8, "default": 0},
        "fan_menqing": {"taiwan": 1, "international": 2, "default": 0},
        "fan_zimo": {"taiwan": 1, "international": 1, "default": 0},
    }
    fan_sum_expr = C(0)
    for name in fan_order:
        vmap = fan_values_variant[name]
        variant_val_expr = C(vmap["default"])
        for vname in ("taiwan", "changsha", "sichuan", "blood", "international"):
            if vname in vmap:
                variant_val_expr = IF(EQ(V("$constants.variant"), C(vname)), C(vmap[vname]), variant_val_expr)
        params_ = fan_params.get(name, ["hand"])
        args = [{"hand": hand, "pid": V("$pid"), "self_win": V("$self_win")}[p] for p in params_]
        fan_sum_expr = ADD(fan_sum_expr, MUL(CALL(name, *args), variant_val_expr))

    return {
        "next_turn": {
            "description": "Cyclic next player within constants.player_ids",
            "params": ["p"],
            "expr": _next_turn_expr(),
        },
        "hand_of": {
            "description": "A player's hand array",
            "params": ["p"],
            "expr": SWITCH([(p, V(f"$hand_{p}")) for p in PLAYERS4], V("$p")),
        },
        "melds_of": {
            "description": "A player's meld list",
            "params": ["p"],
            "expr": SWITCH([(p, V(f"$melds_{p}")) for p in PLAYERS4], V("$p")),
        },
        "seat_of": {
            "description": "A player's seat index (0-based) within PLAYERS4",
            "params": ["p"],
            "expr": SWITCH([(p, C(i)) for i, p in enumerate(PLAYERS4)], V("$p")),
        },
        "is_qidui": {
            "description": "Seven pairs (every group of size 2)",
            "params": ["hand"],
            "expr": _qidui(hand),
        },
        "is_licu": {
            "description": "呖咕呖咕 / 八对半: 7 pairs + 1 triplet = 17 tiles (taiwan, concealed)",
            "params": ["hand"],
            "expr": AND(
                EQ(COUNT(hand), C(17)),
                EQ(COUNT(FILTER(GROUP(hand), EQ(GET(V("$node"), "count"), C(3)))), C(1)),
                EQ(COUNT(FILTER(GROUP(hand), EQ(GET(V("$node"), "count"), C(2)))), C(7)),
            ),
        },
        "is_win_hand": {
            "description": (
                "Meld-aware winning hand. Second arg ``melds`` is the player's "
                "open melds: the standard form checks the CONCEALED hand UNION "
                "meld tiles (副露计入胡牌结构), while 七对 / 十三幺 / 呖咕呖咕 "
                "require melds empty (门清). guangdong/hongzhong = 7 pairs, "
                "thirteen orphans, or standard 4 melds + pair (14); taiwan = "
                "呖咕呖咕 or standard 5 melds + pair (17); changsha = 7 pairs / "
                "将将胡 / 258将小胡 / 大胡(碰碰胡·清一色)乱将; sichuan & blood "
                "(血战到底/血流成河) = base wins AND 缺一门 gate (m/p/s distinct < 3)."
            ),
            "params": ["hand", "melds"],
            "expr": IF(
                EQ(V("$constants.variant"), C("taiwan")),
                OR(
                    AND(EQ(COUNT(V("$melds")), C(0)), CALL("is_licu", hand)),
                    _standard_win(full),
                ),
                IF(
                    EQ(V("$constants.variant"), C("changsha")),
                    OR(
                        AND(EQ(COUNT(V("$melds")), C(0)), CALL("is_qidui", hand)),
                        # 将将胡: 14 tiles all 2/5/8, structure-exempt (incl. melds).
                        AND(EQ(COUNT(full), C(14)), ALL(full, {"contains": [V("$constants.pair_258"), V("$node")]})),
                        # 小胡: standard 4 melds + pair, pair must be 2/5/8.
                        _standard_win(full, _pair_pool_258),
                        # 大胡 (碰碰胡/清一色): 乱将 — any pair + standard structure.
                        AND(_standard_win(full), OR(CALL("fan_pengpenghu", full), CALL("fan_qingyise", full))),
                    ),
                    IF(
                        # 血流成河 (blood) 与血战到底 (sichuan) 同款胡牌判定：
                        # 七对/十三幺/标准形 + 缺一门（108 张无字牌，定缺门
                        # 是两种川麻的共同前置）。
                        OR(
                            EQ(V("$constants.variant"), C("sichuan")),
                            EQ(V("$constants.variant"), C("blood")),
                        ),
                        AND(
                            OR(
                                AND(EQ(COUNT(V("$melds")), C(0)), CALL("is_qidui", hand)),
                                AND(
                                    EQ(COUNT(V("$melds")), C(0)),
                                    EQ(COUNT(hand), C(14)),
                                    ALL(V("$constants.thirteen_orphans"), {"contains": [hand, V("$node")]}),
                                    # 双向：手牌也必须全是幺九（否则 13 幺九+1 普通牌骗和）。
                                    ALL(hand, {"contains": [V("$constants.thirteen_orphans"), V("$node")]}),
                                ),
                                _standard_win(full),
                            ),
                            # 缺一门 gate: fewer than 3 suits incl. meld tiles.
                            NOT(EQ(COUNT(DISTINCT(_suits_noz_expr(full))), C(3))),
                        ),
                        IF(
                            EQ(V("$constants.variant"), C("international")),
                            AND(
                                OR(
                                    AND(EQ(COUNT(V("$melds")), C(0)), CALL("is_qidui", hand)),
                                    AND(
                                        EQ(COUNT(V("$melds")), C(0)),
                                        EQ(COUNT(hand), C(14)),
                                        ALL(V("$constants.thirteen_orphans"), {"contains": [hand, V("$node")]}),
                                        # 双向：手牌也必须全是幺九（否则 13 幺九+1 普通牌骗和）。
                                        ALL(hand, {"contains": [V("$constants.thirteen_orphans"), V("$node")]}),
                                    ),
                                    _standard_win(full),
                                ),
                                GTE(CALL("fan_sum", hand, C("p0"), C(False)), C(8)),
                            ),
                            OR(
                                AND(EQ(COUNT(V("$melds")), C(0)), CALL("is_qidui", hand)),
                                AND(
                                    EQ(COUNT(V("$melds")), C(0)),
                                    EQ(COUNT(hand), C(14)),
                                    ALL(V("$constants.thirteen_orphans"), {"contains": [hand, V("$node")]}),
                                    # 双向：手牌也必须全是幺九（否则 13 幺九+1 普通牌骗和）。
                                    ALL(hand, {"contains": [V("$constants.thirteen_orphans"), V("$node")]}),
                                ),
                                _standard_win(full),
                            ),
                        ),
                    ),
                ),
            ),
        },
        "fan_sum": {
            "description": "Sum of all fan flags for a winning hand (taiwan: 台数 with 门清/自摸)",
            "params": ["hand", "pid", "self_win"],
            "expr": fan_sum_expr,
        },
        # 付分人数：player_count − done（退场）−（本次 winner 已入 done ? 0 : 1）。
        # 结算发生在 do_win 把 winner 计入 winners 之后：sichuan 的 winner 同时
        # 入 done（血战到底胡家退场），其余变体 done 不含本次 winner（仍在局中
        # ——blood 血流成河胡家继续、普通变体游戏直接结束）。
        "payoff_for": {
            "description": (
                "本次胡 pid 的得分增量。自摸：赢家收 fan_pay×付分人数，其余"
                "在局玩家各付 fan_pay；荣和：点炮者（last_discarder）包铳付全部"
                "份额，其余玩家 0——此前荣和与自摸同构，无辜玩家被扣分。"
            ),
            "params": ["pid", "winner", "fan_pay", "self_win"],
            "expr": IF(
                EQ(V("pid"), V("winner")),
                MUL(
                    V("fan_pay"),
                    SUB(
                        SUB(V("$constants.player_count"), COUNT(V("$env.done"))),
                        IF({"contains": [V("$env.done"), V("winner")]}, C(0), C(1)),
                    ),
                ),
                IF(
                    V("self_win"),
                    IF({"contains": [V("$env.done"), V("pid")]}, C(0), SUB(C(0), V("fan_pay"))),
                    IF(
                        EQ(V("pid"), V("$env.last_discarder")),
                        SUB(
                            C(0),
                            MUL(
                                V("fan_pay"),
                                SUB(
                                    SUB(V("$constants.player_count"), COUNT(V("$env.done"))),
                                    IF({"contains": [V("$env.done"), V("winner")]}, C(0), C(1)),
                                ),
                            ),
                        ),
                        C(0),
                    ),
                ),
            ),
        },
        # 血战累计结算：每次胡牌把 ``payoff_for`` 的增量**加**到既有
        # payoffs 上（而非覆写）——先前胡家（已进 done，delta=0）的分数
        # 得以保留；普通变种单胡 = 初始 0 + delta，与旧覆写行为等价。
        "payoff_after": {
            "description": "Accumulated payoff for ``pid`` after the current win ($env.last_winner)",
            "params": ["pid"],
            "expr": ADD(
                AT(V("$env.payoffs"), CALL("seat_of", V("pid"))),
                CALL(
                    "payoff_for",
                    V("pid"),
                    V("$env.last_winner"),
                    V("$env.fan_pay"),
                    V("self_win"),
                ),
            ),
        },
        **fans,
    }


def _visibility():
    rules = []
    for p in PLAYERS4:
        rules.append(
            {
                "view": f"hand_view_{p}",
                "filter": NOT(EQ(V("$viewer"), p)),
                "fields": {"id": "hidden"},
            }
        )
    return {
        "default": "partial",
        "rules": rules,
        # env 标量级投影过滤（v5.2）：``win_hand`` 是胡牌者整手牌的拷贝
        # （do_win 结算写 $env.win_hand 供 fan_sum 计算），对任何观察者都
        # 属隐藏信息 —— 投影观测一律隐藏（hidden_guard 黑名单字段）。
        # 信息不丢失：胡牌者自己的手牌经 ``hand_view_pN`` 已可见。
        "env": {
            "win_hand": {
                "filter": {"const": False},
            },
        },
    }


def _terminal():
    return [{"id": "round_done", "condition": EQ(V("$env.game_over"), C(True))}]


def _utility():
    return [
        {"player": p, "value": AT(V("$env.payoffs"), C(i)), "when": EQ(V("$env.game_over"), C(True))}
        for i, p in enumerate(PLAYERS4)
    ]


def build() -> dict:
    return {
        "meta": {
            "gameId": "mahjong",
            "version": "5.2.0",
            "description": (
                "Mahjong — guangdong / hongzhong (wild z5) / blood (血流成河) / "
                "sichuan (血战到底) / changsha (258将) / taiwan (16张) / "
                "international (国标) × 2-4 players (default 4). The JSON's "
                "``variants`` section declares every option (v5.2); the engine "
                "selects a variant and player count without any adapter injection. "
                "Pure-expression aliases (zero builtins)."
            ),
        },
        "players": PLAYERS4,
        "variants": {
            "variant": "guangdong",
            #: 麻将标准人数为 4 人：默认按 4 人开局（引擎仍接受显式 2 人参数，
            #: 2/4 人都是声明过的合法取值；平台与训练注册表均按 4 人装配）。
            "player_count": 4,
            "options": {
                "guangdong": {},
                "hongzhong": {"constants": {"wild_tile": "z5"}},
                # 血流成河 (blood): 108 no-honor deck（与血战到底同款牌池
                # 与缺一门胡牌判定）；区别在结算——胡家不退场继续摸打
                # （done 仅 sichuan 追加），可多点重复胡牌累计分数，终局
                # 条件 player_count-1 家胡过或牌墙抽干（do_win）。
                "blood": {"constants": {"tile_ids": TILE_IDS_108}},
                # Sichuan 血战到底: 108 no-honor deck (win gate 缺一门 and
                # no-chi are expressed via ``$constants.variant`` in rules).
                "sichuan": {"constants": {"tile_ids": TILE_IDS_108}},
                # Changsha 麻将: 108 no-honor deck; 258将 for 小胡 (大胡乱将);
                # 将将胡 exempt; 1/6/12 番制 via patched fan_pay table.
                "changsha": {
                    "constants": {
                        "tile_ids": TILE_IDS_108,
                        "pair_258": PAIR_258,
                        "fan_pay": FAN_PAY_CHANGSHA,
                        "gang_actions": GANG_ACTIONS,
                    }
                },
                # Taiwan 16张 (no-flower simplification): 5 melds + pair = 17.
                "taiwan": {"constants": {"win_tiles": 17, "meld_k": 5}},
                # International / 国标（Botzone 复式国标接入）：136 张无花，
                # 保留标准 4 副+将/七对/十三幺结构，按近似国标番表 8 番起胡。
                "international": {"constants": {"fan_pay": FAN_PAY_INTERNATIONAL}},
            },
            "player_ids": {
                "map": {
                    "list": {"range": {"from": {"const": 0}, "to": {"var": "$player_count"}}},
                    "as": "$node",
                    "expr": {"template": "p{$node}"},
                }
            },
            "deal_target": {
                "if": {
                    "cond": {"eq": [{"var": "$variant"}, {"const": "taiwan"}]},
                    "then": {"add": [{"mul": [{"const": 16}, {"var": "$player_count"}]}, {"const": 1}]},
                    "else": {"add": [{"mul": [{"const": 13}, {"var": "$player_count"}]}, {"const": 1}]},
                }
            },
            "trim_players": True,
            "trim_utility": True,
        },
        "constants": {
            "tile_ids": TILE_IDS,
            "suit_of": SUIT_OF,
            "chi_runs": CHI_RUNS,
            "thirteen_orphans": THIRTEEN_ORPHANS,
            # 癞子（万能牌）仅红中麻将变体声明（variants.options.hongzhong
            # 补丁 "z5"）；默认空串 = 无癞子——广东/血战/四川/长沙/台湾/国标的
            # 红中都是普通字牌（wild 计数路径对空串恒为 0，无需分支）。
            "wild_tile": "",
            "fan_pay": FAN_PAY,
            # Win-structure parametrization (v5.3): standard form is
            # ``meld_k`` melds + 1 pair = ``win_tiles``; taiwan 16-tile
            # patches both (5 melds / 17 tiles).  ``pair_258`` is the
            # changsha pair whitelist ([] elsewhere = unrestricted, and the
            # pair-pool rule only consults it for variant == changsha).
            "win_tiles": 14,
            "meld_k": 4,
            "pair_258": [],
            "gang_actions": [],
        },
        "groundState": _ground_state(),
        "derivedViews": _derived_views(),
        "queries": _queries(),
        "actions": _actions(),
        "effectors": _effectors(),
        "chance": _chance(),
        "phases": _phases(),
        "visibility": _visibility(),
        "terminal": _terminal(),
        "utility": _utility(),
        "functions": _aliases(),
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "rules/mahjong.json"
    rules = build()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    print(f"written {path} ({len(json.dumps(rules))} bytes)")
