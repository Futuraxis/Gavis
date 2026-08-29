"""Engine tests for UNO rules — ``rules/uno.json`` (v5.2).

Covers:
- variants：六个变种 (classic / seven_zero / jump_in / stacking / draw_until /
  strict_wild4) 与人数 (2..10) 纯数据选择；未知 variant → ValueError
- 发牌→翻牌：每手 ≥7、弃牌 1、牌堆守恒（108 = 手牌+弃牌+牌堆）；首张特殊
  效果（reverse→庄家先手、skip→跳过下家并翻到 p2、draw2/wild4→p1 吃罚牌、
  wild→默认选红）
- 出牌合法性：同色/同符号；万能永远可出；严格+4 在仍有台面颜色时禁止
- 特殊牌效果：skip 进 2、reverse 翻方向（2 人局等价跳过）、draw2/wild4 罚牌
  循环、摸牌后 play_drawn/pass
- 回合推进全部使用 player_id（env.turn / penaltyTarget 必为 pid，
  hand_{$env.turn} 数组模板按 pid 取名）
- 7-0：打出 7 与目标换手、打出 0 全场按方向移交
- 抢牌：同色同数字候选窗口、jump_play / jump_pass 轮转、无候选直接跳过
- 叠加：stack2/stack4 累计罚牌、take_penalty 吃下全部并跳过（classic 无叠加）
- 摸到能打：pick 循环直到摸到可打牌；牌堆耗尽自动停（不会死循环）
- 终局与收益：手牌清空获胜；卡死（牌堆空+无可打）；回合上限；胜者 +1 其余 -1
- 部分可观测：本人手牌可见、他人手牌隐藏但张数可见

Note: 规则内 ``hand_of`` 别名被内联在查询/合法条件里时超出编译器
switch-in-comprehension 形状，引擎自动回退纯解释器（设计内行为）——
本文件用 ``allow_codegen=False`` 直接走解释器路径以保证速度与确定性；
编译路径由 ``tests/test_train_cli.py::test_every_game_builds_engine`` 覆盖。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from _gen_uno import gen_rules
from layer1_translator.schema_validator import SchemaValidator
from layer2_engine.core.engine import GameEngine

RULES_PATH = Path(__file__).resolve().parent.parent.parent / "rules" / "uno.json"
VARIANTS = ["classic", "seven_zero", "jump_in", "stacking", "draw_until", "strict_wild4"]
FILLER = "b1a"  # 蓝 1：对 (r,5) 台面不可接，且非特殊牌，用于给非当前玩家填手牌


def _engine(seed: int = 7, player_count: int = 4, variant: str = "classic") -> GameEngine:
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=seed, player_count=player_count, variant=variant, allow_codegen=False)


def _hand(state: dict, pid: str) -> list:
    return state["_arrays"].get(f"hand_{pid}", [])


def _advance_to_play(engine: GameEngine, rng: random.Random, max_steps: int = 200) -> dict | None:
    """驱动 chance 直到 phase == play；返回状态（或 None）。"""
    state = engine.create_initial_state()
    for _ in range(max_steps):
        if state["env"].get("phase") == "play":
            return state
        nt = engine.get_node_type(state)
        if nt == "chance":
            outs = engine.get_chance_outcomes(state)
            if not outs:
                return None
            state = engine.apply_chance(state, rng.choices(outs, weights=[o.probability for o in outs], k=1)[0])
        else:
            return None
    return None


def _legal(engine: GameEngine, state: dict, tid: str, **params):
    """在 state 中寻找 template_id == tid 且参数匹配的合法动作。"""
    for a in engine.get_legal_actions(state):
        if a.template_id == tid and all(a.params.get(k) == v for k, v in params.items()):
            return a
    return None


def _deck_count(engine: GameEngine, state: dict) -> int:
    """牌堆剩余（规则别名 deck_count，等价于 108 − 手牌 − 弃牌）。"""
    ctx = engine._build_context(state)
    return engine.expr.eval({"call": ["deck_count"]}, ctx)


def _craft(engine: GameEngine, *, hands: dict[str, list], top: tuple | None = ("r", "5"),
           phase: str = "play", turn: str = "p0", direction: int = 1,
           discard: list | None = None, fill_others: bool = True,
           extra_env: dict | None = None) -> dict:
    """手工构造状态：非当前玩家缺省填 1 张 FILLER，避免 hand_empty 提前触发。"""
    state = engine.create_initial_state()
    pids = engine._constants["player_ids"]
    for pid in pids:
        if pid in hands:
            state["_arrays"][f"hand_{pid}"] = list(hands[pid])
        elif fill_others:
            state["_arrays"][f"hand_{pid}"] = [FILLER]
    state["_arrays"]["discard"] = list(discard if discard is not None else ["wild_1"])
    env = state["env"]
    env["phase"] = phase
    env["turn"] = turn
    env["direction"] = direction
    if top is not None:
        env["topColor"], env["topSymbol"] = top
    if extra_env:
        env.update(extra_env)
    return state


def _fill_rest_of_deck(engine: GameEngine, state: dict, keep: set[str], deck_target: int = 0) -> int:
    """把除 keep（+现有手牌）之外的所有牌排进弃牌，使最终牌堆张数 == deck_target。

    牌堆（伪） = 108 − 手牌总数 − 弃牌张数；keep 中的牌留在牌堆。返回最终牌堆张数。
    """
    pids = engine._constants["player_ids"]
    keep = set(keep)
    hand_ids = {c for pid in pids for c in _hand(state, pid)}
    rest = [c for c in engine._constants["card_ids"] if c not in hand_ids and c not in keep]
    hands_total = sum(len(_hand(state, pid)) for pid in pids)
    discard_needed = 108 - hands_total - deck_target
    state["_arrays"]["discard"] = rest[: max(0, discard_needed)]
    return deck_target


def _rand_play(engine: GameEngine, rng: random.Random, state: dict, max_steps: int = 20000) -> tuple[dict, int]:
    for step in range(max_steps):
        nt = engine.get_node_type(state)
        if nt == "terminal":
            return state, step
        if nt == "chance":
            _, state = engine.sample_chance(state)
            continue
        legal = engine.get_legal_actions(state)
        assert legal, f"玩家阶段无合法动作: phase={state['env'].get('phase')} turn={state['env'].get('turn')}"
        state = engine.apply_action(state, rng.choice(legal))
    raise AssertionError(f"{max_steps} 步未到终局 phase={state['env'].get('phase')}")


# ── schema / variants ────────────────────────────────────────────────


def test_schema_valid() -> None:
    rules = gen_rules()
    result = SchemaValidator.validate(rules)
    assert result.valid, result.errors


def test_schema_valid_shipped_json() -> None:
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        rules = json.load(f)
    result = SchemaValidator.validate(rules)
    assert result.valid, result.errors


def test_variants_select_flags() -> None:
    for variant in VARIANTS:
        eng = _engine(variant=variant)
        assert eng.variant == variant
        for name in ("seven_zero", "jump_in", "stacking", "draw_until", "strict_wild4"):
            assert bool(eng._constants[name]) is (variant == name), (variant, name)


def test_player_count_resolution() -> None:
    for n in (2, 3, 6, 10):
        eng = _engine(player_count=n)
        assert eng.player_count == n
        assert len(eng._constants["player_ids"]) == n
        assert len(eng._players) == n


def test_unknown_variant_raises() -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        _engine(variant="no_such_uno")


# ── 发牌 / 翻牌 ─────────────────────────────────────────────────────


def test_deal_and_flip_invariants() -> None:
    rng = random.Random(1)
    for variant in VARIANTS:
        for n in (2, 4, 10):
            eng = _engine(seed=11, player_count=n, variant=variant)
            state = _advance_to_play(eng, rng)
            assert state is not None, variant
            pids = eng._constants["player_ids"]
            for pid in pids:
                assert len(_hand(state, pid)) >= 7, (variant, pid)
            assert len(state["_arrays"]["discard"]) == 1
            assert state["env"]["topColor"] is not None
            assert state["env"]["topSymbol"] is not None
            hands = sum(len(_hand(state, pid)) for pid in pids)
            assert hands + len(state["_arrays"]["discard"]) + _deck_count(eng, state) == 108


def test_first_card_special_effects() -> None:
    """首张特殊效果按规则结算（统计验证：多次取样，每种符号出现时断言其效果）。"""
    rng = random.Random(5)
    seen: dict[str, int] = {}
    for _ in range(200):
        eng = _engine(seed=1000 + _, player_count=4, variant="classic")
        state = _advance_to_play(eng, rng)
        assert state is not None
        sym = state["env"]["topSymbol"]
        seen[sym] = seen.get(sym, 0) + 1
        if sym == "reverse":
            assert state["env"]["direction"] == -1 and state["env"]["turn"] == "p0"
        elif sym == "skip":
            assert state["env"]["turn"] == "p2"  # 跳过 p1 → p2 先手
        elif sym == "draw2":
            assert len(_hand(state, "p1")) == 9 and state["env"]["turn"] == "p2"  # p1 吃 2 并跳过
        elif sym == "wild4":
            assert len(_hand(state, "p1")) == 11 and state["env"]["turn"] == "p2"  # p1 吃 4
        elif sym == "wild":
            assert state["env"]["topColor"] == "r"  # 首张万能默认选红
    assert set(seen) >= {"reverse", "skip", "draw2", "wild", "wild4", "0", "1", "2"}


def test_first_card_reverse_two_players() -> None:
    rng = random.Random(9)
    for s in range(200):
        eng = _engine(seed=2000 + s, player_count=2, variant="classic")
        state = _advance_to_play(eng, rng)
        assert state is not None
        if state["env"]["topSymbol"] == "reverse":
            assert state["env"]["direction"] == -1
            assert state["env"]["turn"] == "p0"  # 2 人局反转首张 = 庄家先手
            return
    pytest.fail("200 次取样未见首张 reverse（概率极低，视为失败）")


# ── 出牌合法性 / 特殊牌效果 ─────────────────────────────────────────


def test_play_legality_color_and_symbol() -> None:
    eng = _engine()
    state = _craft(eng, hands={"p0": ["r5a", "b6a", "wild_1", "wild4_1"]}, top=("r", "5"), turn="p0")
    assert _legal(eng, state, "play", card="r5a") is not None  # 同色
    assert _legal(eng, state, "play", card="b6a") is None  # 不同色不同符号
    assert _legal(eng, state, "play_wild", card="wild_1", color="b") is not None
    assert _legal(eng, state, "play_wild", card="wild4_1", color="y") is not None
    assert _legal(eng, state, "draw") is not None


def test_play_legality_top_symbol_only() -> None:
    eng = _engine()
    state = _craft(eng, hands={"p0": ["b5a", "g5b", "g8a"]}, top=("r", "5"), turn="p0")
    assert _legal(eng, state, "play", card="b5a") is not None  # 同符号
    assert _legal(eng, state, "play", card="g5b") is not None
    assert _legal(eng, state, "play", card="g8a") is None


def test_strict_wild4_restriction() -> None:
    eng = _engine(variant="strict_wild4")
    state = _craft(eng, hands={"p0": ["r5a", "wild4_1", "b6a"]}, top=("r", "5"), turn="p0")
    assert _legal(eng, state, "play_wild", card="wild4_1", color="g") is None  # 手上还有红
    state2 = _craft(eng, hands={"p0": ["b6a", "wild4_1"]}, top=("r", "5"), turn="p0")
    assert _legal(eng, state2, "play_wild", card="wild4_1", color="g") is not None  # 无红 → 可出
    eng2 = _engine(variant="classic")
    state3 = _craft(eng2, hands={"p0": ["r5a", "wild4_1"]}, top=("r", "5"), turn="p0")
    assert _legal(eng2, state3, "play_wild", card="wild4_1", color="g") is not None  # classic 恒可出


def test_skip_advances_two() -> None:
    eng = _engine()
    state = _craft(eng, hands={"p0": ["rsa", "b1a"]}, top=("r", "skip"), turn="p0")
    act = _legal(eng, state, "play", card="rsa")
    assert act is not None
    nxt = eng.apply_action(state, act)
    assert nxt["env"]["turn"] == "p2"  # p0 + 2
    assert nxt["env"]["phase"] == "play"


def test_reverse_flips_direction() -> None:
    eng = _engine()
    state = _craft(eng, hands={"p0": ["rra", "b1a"]}, top=("r", "reverse"), turn="p0", direction=1)
    nxt = eng.apply_action(state, _legal(eng, state, "play", card="rra"))
    assert nxt["env"]["direction"] == -1
    assert nxt["env"]["turn"] == "p3"  # 新方向下的下家（p0 的 -1 方向）
    assert nxt["env"]["phase"] == "play"


def test_reverse_two_player_acts_like_skip() -> None:
    eng = _engine(player_count=2)
    state = _craft(eng, hands={"p0": ["rra", "b1a"]}, top=("r", "reverse"), turn="p0")
    nxt = eng.apply_action(state, _legal(eng, state, "play", card="rra"))
    assert nxt["env"]["turn"] == "p0"  # 反转 = 跳过 → 自己再出
    assert nxt["env"]["phase"] == "play"


def test_draw2_penalty_loop() -> None:
    eng = _engine()
    state = _craft(eng, hands={"p0": ["rda", "b1a"]}, top=("r", "draw2"), turn="p0")
    nxt = eng.apply_action(state, _legal(eng, state, "play", card="rda"))
    assert nxt["env"]["phase"] == "penalty_pick"
    assert nxt["env"]["pendingDraw"] == 2
    assert nxt["env"]["turn"] == "p1"  # 罚牌目标 = 下家（pid）
    for _ in range(2):
        assert nxt["env"]["phase"] == "penalty_pick"
        nxt = eng.sample_chance(nxt)[1]
    assert nxt["env"]["phase"] == "play"
    assert len(_hand(nxt, "p1")) == 3  # 默认填充 b1a + 2 罚牌


def test_wild4_penalty_loop() -> None:
    eng = _engine()
    state = _craft(eng, hands={"p0": ["wild4_1", "b1a"]}, top=("r", "7"), turn="p0")
    nxt = eng.apply_action(state, _legal(eng, state, "play_wild", card="wild4_1", color="g"))
    assert nxt["env"]["phase"] == "penalty_pick"
    assert nxt["env"]["pendingDraw"] == 4
    assert nxt["env"]["topColor"] == "g"  # 选色生效
    for _ in range(4):
        nxt = eng.sample_chance(nxt)[1]
    assert nxt["env"]["phase"] == "play"
    assert len(_hand(nxt, "p1")) == 5  # b1a + 4 罚牌
    assert nxt["env"]["turn"] == "p2"


def test_draw_then_play_drawn_or_pass() -> None:
    eng = _engine()
    state = _craft(eng, hands={"p0": ["b6a"]}, top=("r", "5"), turn="p0")
    st = eng.apply_action(state, _legal(eng, state, "draw"))
    assert st["env"]["phase"] == "pick"
    st = eng.sample_chance(st)[1]
    drawn = st["env"]["drawnCard"]
    assert st["env"]["phase"] == "draw_result"
    assert len(_hand(st, "p0")) == 2
    ctx = eng._build_context(st)
    col = eng.expr.eval({"call": ["color_of", {"var": "$env.drawnCard"}]}, ctx)
    sym = eng.expr.eval({"call": ["symbol_of", {"var": "$env.drawnCard"}]}, ctx)
    playable = col == "r" or sym == "5" or sym in ("wild", "wild4")
    pd = _legal(eng, st, "play_drawn", card=drawn)
    assert (pd is not None) == playable, (drawn, col, sym)
    if pd is not None:
        st2 = eng.apply_action(st, pd)
        assert drawn not in _hand(st2, "p0")
        assert st2["env"]["drawnCard"] is None
        assert st2["env"]["phase"] in ("play", "penalty_pick", "respond")
    else:
        ps = _legal(eng, st, "pass")
        assert ps is not None
        st2 = eng.apply_action(st, ps)
        assert st2["env"]["phase"] == "play"
        assert st2["env"]["drawnCard"] is None
        assert st2["env"]["turn"] == "p1"


# ── 7-0 变种 ─────────────────────────────────────────────────────────


def test_seven_zero_swap_hands() -> None:
    eng = _engine(variant="seven_zero")
    state = _craft(eng, hands={"p0": ["r7a", "b6a"], "p2": ["g3a", "y8a"]}, top=("r", "7"), turn="p0")
    act = _legal(eng, state, "play7", card="r7a", target="p2")
    assert act is not None
    nxt = eng.apply_action(state, act)
    assert _hand(nxt, "p0") == ["g3a", "y8a"]
    assert _hand(nxt, "p2") == ["b6a"]
    assert nxt["env"]["turn"] == "p1"  # 换手后正常进 1
    # classic 下没有 play7（7 走普通出牌）
    eng2 = _engine(variant="classic")
    st2 = _craft(eng2, hands={"p0": ["r7a"]}, top=("r", "7"), turn="p0")
    assert _legal(eng2, st2, "play7") is None
    assert _legal(eng2, st2, "play", card="r7a") is not None


def test_seven_zero_rotate_hands_on_zero() -> None:
    eng = _engine(variant="seven_zero")
    state = _craft(
        eng,
        hands={"p0": ["g1a", "g2a"], "p1": ["r0", "b0"], "p2": ["b1a"], "p3": ["y1a", "y2a", "y3a"]},
        top=("r", "0"),
        turn="p1",  # p1 打出 0 → 全场按方向（+1）移交
    )
    act = _legal(eng, state, "play", card="r0")
    assert act is not None
    nxt = eng.apply_action(state, act)
    assert _hand(nxt, "p1") == _hand(state, "p0")  # p0 的手原封移给 p1（r0 已入弃牌）
    assert _hand(nxt, "p2") == ["b0"]  # p1 出 r0 后的余手 [b0] 移给 p2
    assert _hand(nxt, "p3") == _hand(state, "p2")  # p2 原手移给 p3
    assert _hand(nxt, "p0") == _hand(state, "p3")  # p3 原手移给 p0
    assert nxt["env"]["turn"] == "p2"


def test_seven_zero_hands_snapshot_not_leaked() -> None:
    """P1-3 回归：7-0 换手快照（``env.handsSnapshot``）含他人手牌，
    不得泄露给任何观察者。

    修复前 ``visibility`` 无 ``env`` 子段 → ``handsSnapshot`` 按契约对任意
    viewer 公开；且快照永不清理（0 牌含全场手牌、7 牌含两名换牌者手牌），
    跨回合残留。修复后 ``visibility.env.handsSnapshot`` 对所有 viewer 隐藏
    （filter 恒假）+ 轮转末尾 ``setEnv handsSnapshot=[]`` 清空，双重保险。
    """
    eng = _engine(variant="seven_zero")
    # ── 7 牌换手：快照曾含 [p0 手, p2 手] ──
    state = _craft(eng, hands={"p0": ["r7a", "b6a"], "p2": ["g3a", "y8a"]}, top=("r", "7"), turn="p0")
    nxt = eng.apply_action(state, _legal(eng, state, "play7", card="r7a", target="p2"))
    # (a) 原始 state 的快照已清空（不再残留 p0/p2 的手牌）
    assert nxt["env"].get("handsSnapshot") == []
    # (b) 任何观察者的投影都不含该字段（visibility.env 过滤兜底）
    for viewer in ("p0", "p1", "p2", "p3"):
        env_obs = eng.project_observation(nxt, viewer)["env"]
        assert "handsSnapshot" not in env_obs, f"handsSnapshot leaked to {viewer} on 7-swap"

    # ── 0 牌全场移交：快照曾含全场手牌 ──
    state0 = _craft(
        eng,
        hands={"p0": ["g1a", "g2a"], "p1": ["r0", "b0"], "p2": ["b1a"], "p3": ["y1a", "y2a", "y3a"]},
        top=("r", "0"),
        turn="p1",
    )
    nxt0 = eng.apply_action(state0, _legal(eng, state0, "play", card="r0"))
    assert nxt0["env"].get("handsSnapshot") == []
    for viewer in ("p0", "p1", "p2", "p3"):
        env_obs = eng.project_observation(nxt0, viewer)["env"]
        assert "handsSnapshot" not in env_obs, f"handsSnapshot leaked to {viewer} on 0-rotate"


# ── 抢牌变种 ─────────────────────────────────────────────────────────


def test_jump_in_window_and_play() -> None:
    eng = _engine(variant="jump_in")
    state = _craft(
        eng,
        hands={"p0": ["r5a", "b1a"], "p1": ["r5b", "g8a"], "p2": ["g8a"], "p3": ["y1a"]},
        top=("r", "4"),  # r5a 同色可打；打出后台面 (r,5)
        turn="p0",
    )
    nxt = eng.apply_action(state, _legal(eng, state, "play", card="r5a"))
    assert nxt["env"]["phase"] == "jump"
    assert nxt["env"]["turn"] == "p1"  # 唯一候选：p1 有 r5b（同色同数字）
    assert _legal(eng, nxt, "jump_play", card="r5b") is not None
    assert _legal(eng, nxt, "jump_play", card="g8a") is None  # 非同色同数字
    nxt2 = eng.apply_action(nxt, _legal(eng, nxt, "jump_play", card="r5b"))
    assert nxt2["env"]["phase"] == "play"
    assert nxt2["env"]["turn"] == "p2"  # 抢牌者 p1 之后推进
    assert _hand(nxt2, "p1") == ["g8a"]


def test_jump_in_pass_closes_window() -> None:
    eng = _engine(variant="jump_in")
    state = _craft(
        eng,
        hands={"p0": ["r5a", "b1a"], "p1": ["r5b"], "p2": ["g8a"], "p3": ["y1a"]},
        top=("r", "4"),
        turn="p0",
    )
    nxt = eng.apply_action(state, _legal(eng, state, "play", card="r5a"))
    assert nxt["env"]["phase"] == "jump"
    assert nxt["env"]["turn"] == "p1"
    nxt = eng.apply_action(nxt, _legal(eng, nxt, "jump_pass"))
    assert nxt["env"]["phase"] == "play"  # 唯一候选放弃 → 窗口关闭
    assert nxt["env"]["turn"] == "p1"  # 回到正常下家（p0 的下家 = p1）


def test_jump_in_no_candidates_skips_window() -> None:
    eng = _engine(variant="jump_in")
    state = _craft(
        eng,
        hands={"p0": ["r5a", "b1a"], "p1": ["b6a"], "p2": ["g8a"], "p3": ["y2a"]},
        top=("r", "4"),
        turn="p0",
    )
    nxt = eng.apply_action(state, _legal(eng, state, "play", card="r5a"))
    assert nxt["env"]["phase"] == "play"  # 无候选 → 直接进正常回合
    assert nxt["env"]["turn"] == "p1"


def test_jump_in_off_in_other_variants() -> None:
    eng = _engine(variant="classic")
    state = _craft(eng, hands={"p0": ["r5a", "b1a"], "p1": ["r5b"]}, top=("r", "4"), turn="p0")
    nxt = eng.apply_action(state, _legal(eng, state, "play", card="r5a"))
    assert nxt["env"]["phase"] == "play"  # classic 无抢牌窗口
    assert nxt["env"]["turn"] == "p1"


# ── 叠加变种 ─────────────────────────────────────────────────────────


def test_stacking_respond_chain() -> None:
    eng = _engine(variant="stacking")
    state = _craft(
        eng,
        hands={"p0": ["rda", "b1a"], "p1": ["rdb", FILLER], "p2": ["g1a"]},
        top=("r", "5"),
        turn="p0",
    )
    nxt = eng.apply_action(state, _legal(eng, state, "play", card="rda"))
    assert nxt["env"]["phase"] == "respond"
    assert nxt["env"]["pendingDraw"] == 2
    assert nxt["env"]["turn"] == "p1"
    nxt = eng.apply_action(nxt, _legal(eng, nxt, "stack2", card="rdb"))
    assert nxt["env"]["pendingDraw"] == 4
    assert nxt["env"]["turn"] == "p2"
    nxt = eng.apply_action(nxt, _legal(eng, nxt, "take_penalty"))
    assert nxt["env"]["phase"] == "penalty_pick"
    assert nxt["env"]["pendingDraw"] == 4
    for _ in range(4):
        nxt = eng.sample_chance(nxt)[1]
    assert nxt["env"]["phase"] == "play"
    assert len(_hand(nxt, "p2")) == 5  # g1a + 4 罚牌
    assert nxt["env"]["turn"] == "p3"  # p2 被跳过


def test_stacking_wild4_chain() -> None:
    eng = _engine(variant="stacking")
    state = _craft(
        eng,
        hands={"p0": ["rdb", "b1a"], "p1": ["wild4_1", "b1a"], "p2": ["g1a"]},
        top=("r", "5"),
        turn="p0",
    )
    nxt = eng.apply_action(state, _legal(eng, state, "play", card="rdb"))
    assert nxt["env"]["turn"] == "p1"
    nxt = eng.apply_action(nxt, _legal(eng, nxt, "stack4", card="wild4_1", color="b"))
    assert nxt["env"]["pendingDraw"] == 6
    assert nxt["env"]["topColor"] == "b"
    nxt = eng.apply_action(nxt, _legal(eng, nxt, "take_penalty"))
    for _ in range(6):
        nxt = eng.sample_chance(nxt)[1]
    assert nxt["env"]["phase"] == "play"
    assert len(_hand(nxt, "p2")) == 7  # g1a + 6


def test_draw2_no_stacking_in_classic() -> None:
    eng = _engine(variant="classic")
    state = _craft(eng, hands={"p0": ["rda", "b1a"], "p1": ["rdb"]}, top=("r", "5"), turn="p0")
    nxt = eng.apply_action(state, _legal(eng, state, "play", card="rda"))
    assert nxt["env"]["phase"] == "penalty_pick"  # 直接进罚牌，无 respond
    assert _legal(eng, nxt, "stack2") is None


# ── 摸到能打变种 ─────────────────────────────────────────────────────


def test_draw_until_keeps_picking() -> None:
    eng = _engine(variant="draw_until")
    state = _craft(eng, hands={"p0": ["b6a", "b7a"]}, top=("r", "5"), turn="p0")
    st = eng.apply_action(state, _legal(eng, state, "draw"))
    assert st["env"]["phase"] == "pick"
    assert len(_hand(st, "p0")) == 2  # 摸牌动作本身不进手；pick chance 才 append
    for _ in range(300):
        st = eng.sample_chance(st)[1]
        if st["env"]["phase"] == "draw_result":
            break
    assert st["env"]["phase"] == "draw_result"  # 牌堆充足 → 最终摸到可打牌
    ctx = eng._build_context(st)
    col = eng.expr.eval({"call": ["color_of", {"var": "$env.drawnCard"}]}, ctx)
    sym = eng.expr.eval({"call": ["symbol_of", {"var": "$env.drawnCard"}]}, ctx)
    assert col == "r" or sym == "5" or sym in ("wild", "wild4")  # 停下的那张一定可打


def test_draw_until_stops_when_deck_empty() -> None:
    """最后一张不可打 → 摸完停在 draw_result（牌堆空不再续摸，不会死循环）。"""
    eng = _engine(variant="draw_until")
    deck_keep = "b8a"  # 不可接 r5
    assert deck_keep in eng._constants["card_ids"]
    state = _craft(eng, hands={"p0": ["b6a", "b7a"]}, top=("r", "5"), turn="p0", discard=[])
    leftover = _fill_rest_of_deck(eng, state, keep={deck_keep}, deck_target=1)
    assert leftover == 1  # 牌堆只剩 b8a
    assert _deck_count(eng, state) == 1
    st = eng.apply_action(state, _legal(eng, state, "draw"))
    assert st["env"]["phase"] == "pick"
    st = eng.sample_chance(st)[1]
    assert st["env"]["drawnCard"] == deck_keep
    assert st["env"]["phase"] == "draw_result"  # 牌堆空 → 循环停
    assert _legal(eng, st, "play_drawn") is None
    assert _legal(eng, st, "pass") is not None
    assert len(_hand(st, "p0")) == 3


def test_draw_until_off_in_classic() -> None:
    eng = _engine(variant="classic")
    state = _craft(eng, hands={"p0": ["b6a", "b7a", "b8a"]}, top=("r", "5"), turn="p0")
    st = eng.apply_action(state, _legal(eng, state, "draw"))
    st = eng.sample_chance(st)[1]
    assert st["env"]["phase"] == "draw_result"  # classic：摸一张即停


# ── 终局 / 收益 ──────────────────────────────────────────────────────


def test_win_by_empty_hand() -> None:
    eng = _engine()
    state = _craft(eng, hands={"p0": ["rra"]}, top=("r", "reverse"), turn="p0")
    nxt = eng.apply_action(state, _legal(eng, state, "play", card="rra"))
    assert eng.is_terminal(nxt)
    assert nxt["env"]["winner"] == "p0"  # p0 清空手牌 → 胜
    assert eng.get_utility(nxt, "p0") == 1.0
    for pid in ("p1", "p2", "p3"):
        assert eng.get_utility(nxt, pid) == -1.0


def test_stuck_terminal() -> None:
    eng = _engine()
    state = _craft(eng, hands={"p0": ["b6a", "b7a"], "p1": ["g1a"], "p2": ["y1a"], "p3": ["b8a"]},
                   top=("r", "5"), turn="p0", discard=[])
    leftover = _fill_rest_of_deck(eng, state, keep=set())
    assert leftover == 0  # 牌堆空
    assert _deck_count(eng, state) == 0
    assert eng.is_terminal(state)  # p0 无可打 + 牌堆空 → 卡死
    # 手工构造的卡死状态未经任何 action，env.winner 尚未由 do_end_check 写入；
    # utility 不需要 env.winner（least_player 直接按 game_ended 结算）。
    assert state["env"]["winner"] is None
    assert eng.get_utility(state, "p1") == 1.0
    assert eng.get_utility(state, "p0") == -1.0


def test_max_turns_terminal() -> None:
    eng = _engine()
    state = _craft(eng, hands={"p0": ["r7a"]}, top=("r", "7"), turn="p0", extra_env={"turnCount": 1999})
    nxt = eng.apply_action(state, _legal(eng, state, "play", card="r7a"))
    assert nxt["env"]["turnCount"] == 2000
    assert eng.is_terminal(nxt)
    assert nxt["env"]["winner"] == "p0"  # 最少手牌（清空者）
    assert eng.get_utility(nxt, "p0") == 1.0


def test_utility_sum_is_two_minus_n() -> None:
    eng = _engine(player_count=4)
    state = _craft(eng, hands={"p0": ["rsa"]}, top=("r", "skip"), turn="p0")
    nxt = eng.apply_action(state, _legal(eng, state, "play", card="rsa"))
    assert eng.is_terminal(nxt)  # p0 打出唯一牌 → 全空手牌
    pids = eng._constants["player_ids"]
    total = sum(eng.get_utility(nxt, pid) for pid in pids)
    assert total == 2 - 4  # +1（胜者）+ 3×(−1) = −2


# ── 部分可观测 ───────────────────────────────────────────────────────


def test_visibility_own_hand_visible_others_hidden() -> None:
    eng = _engine()
    state = _craft(eng, hands={"p0": ["r5a", "b6a"], "p1": ["g1a", "g2a"]}, top=("r", "5"), turn="p0")
    obs = eng.get_observation(state, "p0")
    own = obs.get("hand_view_p0", [])
    other = obs.get("hand_view_p1", [])
    assert [e.get("id") for e in own] == ["r5a", "b6a"]  # 本人可见牌面
    assert len(other) == 2  # 他人行仍在 → 张数可见
    assert all(e.get("id") is None for e in other)  # 牌面隐藏


# ── 随机自对弈（终止性）──────────────────────────────────────────────


def test_random_selfplay_terminates() -> None:
    rng = random.Random(3)
    for variant in VARIANTS:
        for n in (2, 4, 6):
            eng = _engine(seed=42, player_count=n, variant=variant)
            for _ in range(2):
                st, _ = _rand_play(eng, rng, eng.create_initial_state())
                assert eng.is_terminal(st)
                winner = st["env"].get("winner")
                pids = eng._constants["player_ids"]
                assert winner is None or winner in pids
                total = sum(eng.get_utility(st, pid) for pid in pids)
                assert total == 2 - n, (variant, n, total)


def test_random_selfplay_conserves_cards() -> None:
    rng = random.Random(8)
    eng = _engine(seed=1, player_count=4, variant="classic")
    st, _ = _rand_play(eng, rng, eng.create_initial_state())
    pids = eng._constants["player_ids"]
    hands = sum(len(_hand(st, pid)) for pid in pids)
    assert hands + len(st["_arrays"]["discard"]) + _deck_count(eng, st) == 108
