"""Tests for Mahjong (Layer 2, v5.1 — one JSON, all variants).

Covers dealing, draw/discard loops, chi/peng/gang claims, concealed and
added gangs, self-win (tsumo) and ron wins, 7 pairs / thirteen orphans,
hongzhong wild-tile coverage, blood-variant done-skip, wall-empty draws,
and claim-queue rotation.
"""

from __future__ import annotations

import json
from pathlib import Path

from layer2_engine.core.engine import GameEngine

RULES_PATH = Path(__file__).resolve().parent.parent.parent / "rules" / "mahjong.json"


def _engine(variant: str | None = None, player_count: int = 2, seed: int = 1) -> GameEngine:
    """Bare engine — variant/player count are declared data (v5.2)."""
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=seed, variant=variant, player_count=player_count)


def _resolve(adapter, state: dict) -> dict:
    while adapter.get_node_type(state) == "chance":
        _, state = adapter.sample_chance(state)
    return state


def _act(adapter, state: dict, template: str, **params) -> dict:
    for a in adapter.get_legal_actions(state):
        if a.template_id == template and all(a.params.get(k) == v for k, v in params.items()):
            return adapter.apply_action(state, a)
    raise AssertionError(f"not legal: {template} {params} at {state['env']['phase']}")


def _resolve_after(adapter, state: dict) -> dict:
    state = _resolve(adapter, state)
    return state


def _win_legal(adapter, state: dict) -> bool:
    """Is a win legal for the current turn (self-win or ron)?"""
    for a in adapter.get_legal_actions(state):
        if a.template_id in ("win_self", "claim_win"):
            return True
    return False


def _eval_win(adapter, hand: list, melds: list | None = None) -> bool:
    ctx = adapter._build_context(adapter.create_initial_state())  # noqa: SLF001
    return bool(adapter.expr.eval({"call": ["is_win_hand", {"const": hand}, {"const": melds or []}]}, ctx))


# ── Dealing ───────────────────────────────────────────────────────────


class TestDeal:
    def test_deal_2p(self):
        a = _engine(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        assert len(s["_arrays"]["hand_p0"]) == 14
        assert len(s["_arrays"]["hand_p1"]) == 13
        assert s["env"]["wall_count"] == 136 - 27
        assert s["env"]["phase"] == "action"
        assert s["env"]["turn"] == "p0"

    def test_deal_4p(self):
        a = _engine(player_count=4, seed=1)
        s = _resolve(a, a.create_initial_state())
        assert [len(s["_arrays"][f"hand_p{i}"]) for i in range(4)] == [14, 13, 13, 13]
        assert s["env"]["wall_count"] == 136 - 53

    def test_no_duplicate_tiles(self):
        a = _engine(player_count=4, seed=2)
        s = _resolve(a, a.create_initial_state())
        drawn = s["_arrays"]["drawn"]
        assert len(drawn) == 53
        assert max(drawn.count(t) for t in set(drawn)) <= 4  # kinds, 4 copies


# ── Draw / discard loop ───────────────────────────────────────────────


class TestFlow:
    def test_discard_then_draw(self):
        a = _engine(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s = _act(a, s, "discard", tile=s["_arrays"]["hand_p0"][0])
        assert s["env"]["phase"] == "claim"
        assert s["env"]["claim_queue"] == ["p1"]
        # 胡>碰/杠>吃 三阶段全过 → draw（2p 队列每人每阶段一次）。
        for _ in range(3):
            assert s["env"]["phase"] == "claim"
            s = _act(a, s, "claim_pass")
        assert s["env"]["phase"] == "draw"
        s = _resolve_after(a, s)
        assert s["env"]["phase"] == "action"
        assert s["env"]["turn"] == "p1"
        assert len(s["_arrays"]["hand_p1"]) == 14

    def test_claim_queue_rotation_4p(self):
        a = _engine(player_count=4, seed=1)
        s = _resolve(a, a.create_initial_state())
        s = _act(a, s, "discard", tile=s["_arrays"]["hand_p0"][0])
        assert s["env"]["claim_queue"] == ["p1", "p2", "p3"]
        # 三阶段 × 三人 = 9 次 pass 后弃牌作废回摸牌。
        for _ in range(9):
            assert s["env"]["phase"] == "claim"
            s = _act(a, s, "claim_pass")
        assert s["env"]["phase"] == "draw"
        s = _resolve_after(a, s)
        assert s["env"]["turn"] == "p1"

    def test_peng_claims_and_melds(self):
        a = _engine(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        tile = s["_arrays"]["hand_p0"][0]
        s = _act(a, s, "discard", tile=tile)
        # Force a peng by p1: plant two copies into p1's hand
        s["_arrays"]["hand_p1"] = [tile, tile] + s["_arrays"]["hand_p1"][:-2]
        # 碰在 meld 阶段：win 阶段（p1 不能胡）先 pass 一次。
        assert s["env"]["claim_mode"] == "win"
        s = _act(a, s, "claim_pass")
        assert s["env"]["claim_mode"] == "meld"
        s = _act(a, s, "claim_peng", tile=tile)
        # 标准麻将：碰后直接打牌（不摸牌），手牌 13 → 11。
        assert s["env"]["phase"] == "discard"
        assert s["env"]["turn"] == "p1"
        melds = s["_arrays"]["melds_p1"]
        assert melds == [{"type": "peng", "tiles": [tile, tile, tile], "from": "p0"}]
        assert s["_arrays"]["hand_p1"].count(tile) == 0

    def test_chi_only_first_responder(self):
        a = _engine(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        # Plant a run around a tile p0 discards
        hand = s["_arrays"]["hand_p1"]
        tile = "m5"
        s["_arrays"]["hand_p0"] = [tile] + s["_arrays"]["hand_p0"][1:]
        rest = [t for t in hand if t not in ("m4", "m5", "m6")][:10]
        s["_arrays"]["hand_p1"] = ["m4", "m6", "m5"] + rest
        s = _act(a, s, "discard", tile=tile)
        # 吃只在 chi 阶段（win/meld 两阶段过后）对下家开放。
        for _ in range(2):
            assert s["env"]["phase"] == "claim"
            s = _act(a, s, "claim_pass")
        assert s["env"]["claim_mode"] == "chi"
        # p1 can chi m4+m6+m5
        s = _act(a, s, "claim_chi", tiles=["m4", "m5", "m6"])
        assert s["env"]["turn"] == "p1"
        # 吃后直接打牌（不摸牌），保持 13 张在手的不变量。
        assert s["env"]["phase"] == "discard"
        assert s["_arrays"]["melds_p1"][0]["type"] == "chi"
        assert s["_arrays"]["melds_p1"][0]["tiles"] == ["m4", "m5", "m6"]
        assert "m4" not in s["_arrays"]["hand_p1"]
        assert "m6" not in s["_arrays"]["hand_p1"]
        # 只移除顺子里另外两张（m4/m6）；本家自己持有的 m5 必须保留
        # （副露里的 m5 是打出方的弃牌，不是本家手牌）。
        assert s["_arrays"]["hand_p1"].count("m5") == 1
        assert len(s["_arrays"]["hand_p1"]) == 11  # 13 − 2（而非 −3）

    def test_chi_requires_run_tiles_in_hand(self):
        """Chi must not be legal when the two non-discard run tiles are
        absent — the observed chaos (bogus chi conjured melds from thin
        air and shrank hands erratically)."""
        a = _engine(player_count=4, seed=1)
        s = _resolve(a, a.create_initial_state())
        s["_arrays"]["hand_p0"] = ["m5"] + s["_arrays"]["hand_p0"][1:]
        s["_arrays"]["hand_p1"] = [t for t in s["_arrays"]["hand_p1"] if t not in ("m4", "m6")][:13]
        s = _act(a, s, "discard", tile="m5")
        # 推进到 chi 阶段（win/meld 全队过）再检查 —— chi 只在 chi 阶段出现。
        while s["env"]["claim_mode"] != "chi":
            assert s["env"]["phase"] == "claim"
            s = _act(a, s, "claim_pass")
        assert s["env"]["claim_index"] == 0  # 吃仅下家（队列头）
        chi = [x for x in a.get_legal_actions(s) if x.template_id == "claim_chi"]
        assert chi == []

    def test_chi_only_valid_run_offered(self):
        """Only runs whose two non-discard tiles are in hand are offered."""
        a = _engine(player_count=4, seed=1)
        s = _resolve(a, a.create_initial_state())
        s["_arrays"]["hand_p0"] = ["m5"] + s["_arrays"]["hand_p0"][1:]
        hand = s["_arrays"]["hand_p1"]
        rest = [t for t in hand if t not in ("m4", "m6")][:11]
        s["_arrays"]["hand_p1"] = ["m4", "m6"] + rest  # has m4, m6; no m3/m7
        s = _act(a, s, "discard", tile="m5")
        while s["env"]["claim_mode"] != "chi":
            assert s["env"]["phase"] == "claim"
            s = _act(a, s, "claim_pass")
        assert s["env"]["claim_index"] == 0
        chi = [x.params.get("tiles") for x in a.get_legal_actions(s) if x.template_id == "claim_chi"]
        assert chi == [["m4", "m5", "m6"]]

    def test_win_with_open_meld_tsumo_and_ron(self):
        """Meld-aware wins: open melds count toward the 14-tile structure
        (a chi holder can tsumo and ron — previously claims made winning
        impossible)."""
        # Tsumo: chi meld + 11-tile ready hand (drawing already included).
        a = _engine(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s["_arrays"]["melds_p0"] = [{"type": "chi", "tiles": ["m2", "m3", "m4"]}]
        s["_arrays"]["hand_p0"] = ["m5", "m6", "m7", "p1", "p2", "p3", "s1", "s2", "s3", "z1", "z1"]
        s["env"]["last_drawn"] = "z1"
        assert _win_legal(a, s)
        s = _act(a, s, "win_self")
        assert a.is_terminal(s)
        assert s["env"]["winner"] == "p0"
        # Ron: chi meld + 10-tile hand, the discard completes the 14th.
        a = _engine(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s["_arrays"]["hand_p1"] = ["m6", "m7", "p1", "p2", "p3", "s1", "s2", "s3", "z1", "z1"]
        s["_arrays"]["melds_p1"] = [{"type": "chi", "tiles": ["m2", "m3", "m4"]}]
        s["_arrays"]["hand_p0"] = ["m5"] + s["_arrays"]["hand_p0"][1:]
        s = _act(a, s, "discard", tile="m5")
        s = _act(a, s, "claim_win", tile="m5")
        assert a.is_terminal(s)
        assert s["env"]["winner"] == "p1"

    def test_qidui_requires_concealed(self):
        """七对 stays concealed-only: open melds must not enable it."""
        a = _engine(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s["_arrays"]["melds_p0"] = [{"type": "chi", "tiles": ["m1", "m2", "m3"]}]
        s["_arrays"]["hand_p0"] = ["z1", "z1", "z2", "z2", "z3", "z3", "z4", "z4", "z5", "z5", "z6", "z6"]
        s["env"]["last_drawn"] = "z6"
        assert not _win_legal(a, s)

    def test_eval_win_meld_aware(self):
        a = _engine(player_count=2, seed=1)
        # 11-tile ready hand + chi meld → complete 14-tile structure.
        assert _eval_win(
            a,
            ["m5", "m6", "m7", "p1", "p2", "p3", "s1", "s2", "s3", "z1", "z1"],
            [{"type": "chi", "tiles": ["m2", "m3", "m4"]}],
        )
        # Same tiles but no melds → not a win (only 11 tiles).
        assert not _eval_win(a, ["m5", "m6", "m7", "p1", "p2", "p3", "s1", "s2", "s3", "z1", "z1"])

    def test_concealed_gang_draws_replacement(self):
        a = _engine(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        hand = s["_arrays"]["hand_p0"]
        tile = hand[0]
        hand[1:5] = [tile, tile, tile]  # four of a kind
        s = _act(a, s, "gang_concealed", tile=tile)
        assert s["env"]["phase"] == "gang_draw"
        s = _resolve_after(a, s)
        assert s["env"]["phase"] == "action"
        assert s["env"]["turn"] == "p0"
        assert s["_arrays"]["melds_p0"][0]["type"] == "concealed_gang"

    def test_added_gang_promotes_peng(self):
        a = _engine(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        tile = "m7"
        s["_arrays"]["melds_p0"] = [{"type": "peng", "tiles": [tile, tile, tile], "from": "p1"}]
        hand = s["_arrays"]["hand_p0"]
        if tile not in hand:
            hand[0] = tile
        s = _act(a, s, "gang_added", tile=tile)
        assert s["env"]["phase"] == "gang_draw"
        s = _resolve_after(a, s)
        assert s["_arrays"]["melds_p0"] == [{"type": "added_gang", "tiles": [tile, tile, tile, tile]}]
        assert tile not in s["_arrays"]["hand_p0"]


# ── 响应优先级（胡>碰/杠>吃）──────────────────────────────────────────


class TestClaimPriority:
    """标准麻将响应优先级：任意胡 > 任意碰/杠 > 下家吃。

    旧模型按响应队列先到先得（首个响应者先表态，可能提前于后位碰/杠）。
    v5.5 改为三阶段（``claim_mode``：win → meld → chi）——每一阶段整队
    问询，全过才进入下一阶段。
    """

    def test_peng_of_later_seat_preempts_chi_of_next(self):
        """p1 可吃 m5（下家）、p2 可碰 m5：meld 阶段先于 chi 阶段 —— p2 的
        碰必须压过 p1 的吃（旧模型 p1 先表态会直接吃掉）。"""
        a = _engine(player_count=4, seed=3)
        s = _resolve(a, a.create_initial_state())
        s["_arrays"]["hand_p0"] = ["m5"] + s["_arrays"]["hand_p0"][1:]
        s["_arrays"]["hand_p1"] = ["m4", "m6", "m5"] + [
            t for t in s["_arrays"]["hand_p1"] if t not in ("m4", "m5", "m6")
        ][:10]
        s["_arrays"]["hand_p2"] = ["m5", "m5"] + s["_arrays"]["hand_p2"][:-2]
        s = _act(a, s, "discard", tile="m5")
        assert s["env"]["claim_mode"] == "win"
        # win 阶段全队只有 pass（无人能胡）。
        for _ in range(3):
            assert {x.template_id for x in a.get_legal_actions(s)} == {"claim_pass"}
            s = _act(a, s, "claim_pass")
        assert s["env"]["claim_mode"] == "meld"
        # meld 阶段 p1 不能碰（只能吃，但吃在更后的 chi 阶段）。
        assert s["env"]["turn"] == "p1"
        assert {x.template_id for x in a.get_legal_actions(s)} == {"claim_pass"}
        s = _act(a, s, "claim_pass")
        assert s["env"]["turn"] == "p2"
        legal_ids = {x.template_id for x in a.get_legal_actions(s)}
        assert "claim_peng" in legal_ids and "claim_chi" not in legal_ids
        s = _act(a, s, "claim_peng", tile="m5")
        assert s["env"]["phase"] == "discard" and s["env"]["turn"] == "p2"

    def test_chi_offered_only_after_meld_stage_passes(self):
        """吃只在 chi 阶段出现；meld 阶段即使下家有吃牌组也不提供 claim_chi。"""
        a = _engine(player_count=2, seed=4)
        s = _resolve(a, a.create_initial_state())
        s["_arrays"]["hand_p0"] = ["m5"] + s["_arrays"]["hand_p0"][1:]
        s["_arrays"]["hand_p1"] = ["m4", "m6", "m5"] + [
            t for t in s["_arrays"]["hand_p1"] if t not in ("m4", "m5", "m6")
        ][:10]
        s = _act(a, s, "discard", tile="m5")
        assert {x.template_id for x in a.get_legal_actions(s)} == {"claim_pass"}  # win
        s = _act(a, s, "claim_pass")
        assert s["env"]["claim_mode"] == "meld"
        assert {x.template_id for x in a.get_legal_actions(s)} == {"claim_pass"}  # 吃不在 meld 阶段
        s = _act(a, s, "claim_pass")
        assert s["env"]["claim_mode"] == "chi" and s["env"]["turn"] == "p1"
        assert "claim_chi" in {x.template_id for x in a.get_legal_actions(s)}

    def test_win_preempts_peng_of_earlier_seat(self):
        """p1 可碰 z1、p2 可荣和 z1：win 阶段（p2 轮到）胡优先于任何碰。"""
        a = _engine(player_count=4, seed=5)
        s = _resolve(a, a.create_initial_state())
        s["_arrays"]["hand_p0"] = ["z1"] + s["_arrays"]["hand_p0"][1:]
        s["_arrays"]["hand_p1"] = ["z1", "z1"] + s["_arrays"]["hand_p1"][:-2]
        s["_arrays"]["hand_p2"] = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "p1", "p2", "p3", "z1"]
        s = _act(a, s, "discard", tile="z1")
        # win 阶段：p1 不能胡 → 只有 pass；p2 可荣和。
        assert {x.template_id for x in a.get_legal_actions(s)} == {"claim_pass"}
        s = _act(a, s, "claim_pass")
        assert s["env"]["turn"] == "p2"
        legal_ids = {x.template_id for x in a.get_legal_actions(s)}
        assert "claim_win" in legal_ids and "claim_peng" not in legal_ids
        s = _act(a, s, "claim_win", tile="z1")
        assert a.is_terminal(s) and s["env"]["winner"] == "p2"

    def test_win_scan_precedes_chi_of_next_seat(self):
        """后位 p3 能荣和：win 阶段全队先扫胡 —— 胡优先于任何下家吃/碰。"""
        a = _engine(player_count=4, seed=7)
        s = _resolve(a, a.create_initial_state())
        s["_arrays"]["hand_p0"] = ["z1"] + s["_arrays"]["hand_p0"][1:]
        s["_arrays"]["hand_p1"] = ["m4", "m6", "m5"] + [
            t for t in s["_arrays"]["hand_p1"] if t not in ("m4", "m5", "m6")
        ][:10]
        p3_win = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "p1", "p2", "p3", "z1"]
        s["_arrays"]["hand_p3"] = p3_win
        s = _act(a, s, "discard", tile="z1")
        assert {x.template_id for x in a.get_legal_actions(s)} == {"claim_pass"}  # p1 win 阶段
        s = _act(a, s, "claim_pass")
        assert {x.template_id for x in a.get_legal_actions(s)} == {"claim_pass"}  # p2
        s = _act(a, s, "claim_pass")
        assert s["env"]["turn"] == "p3"
        assert "claim_win" in {x.template_id for x in a.get_legal_actions(s)}
        s = _act(a, s, "claim_win", tile="z1")
        assert s["env"]["winner"] == "p3"


# ── Win detection (expression aliases) ────────────────────────────────


class TestWinHand:
    def test_standard_win_123_456_789_pairs(self):
        a = _engine(player_count=2, seed=1)
        hand = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "p1", "p2", "p3", "p3", "p3"]
        assert _eval_win(a, hand)

    def test_not_a_win(self):
        a = _engine(player_count=2, seed=1)
        hand = ["m1", "m2", "m4", "m5", "m6", "m7", "m8", "m9", "p1", "p1", "p2", "p3", "p3", "p4"]
        assert not _eval_win(a, hand)

    def test_seven_pairs(self):
        a = _engine(player_count=2, seed=1)
        hand = ["m1", "m1", "m2", "m2", "p3", "p3", "s4", "s4", "z1", "z1", "z2", "z2", "z3", "z3"]
        assert _eval_win(a, hand)

    def test_thirteen_orphans(self):
        a = _engine(player_count=2, seed=1)
        hand = ["m1", "m9", "p1", "p9", "s1", "s9", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "m1"]
        assert _eval_win(a, hand)

    def test_international_mahjong_requires_eight_fan(self):
        a = _engine(variant="international", player_count=4, seed=1)
        low_fan = ["m1", "m2", "m3", "m4", "m5", "m6", "p1", "p2", "p3", "s1", "s2", "s3", "z1", "z1"]
        qingyise = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m2", "m3", "m4", "m5", "m5"]
        thirteen = ["m1", "m9", "p1", "p9", "s1", "s9", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "m1"]
        assert not _eval_win(a, low_fan)
        assert _eval_win(a, qingyise)
        assert _eval_win(a, thirteen)

    def test_hongzhong_wild_fills_gap(self):
        a = _engine(variant="hongzhong", player_count=2, seed=1)
        # Two red dragons stand in for the missing m3 and m6
        hand = ["m1", "m2", "m4", "m5", "p1", "p1", "p1", "p2", "p3", "s1", "s2", "s3", "z5", "z5"]
        assert _eval_win(a, hand)

    def test_hongzhong_wild_not_enough(self):
        a = _engine(variant="hongzhong", player_count=2, seed=1)
        hand = ["m1", "m2", "m4", "m5", "p1", "p1", "p1", "p2", "p3", "s1", "s2", "s3", "z5"]
        assert not _eval_win(a, hand)

    def test_wild_not_leaked_to_guangdong(self):
        """癞子只属于红中麻将：广东局同样两手 z5 不能补 m3/m6 缺口。"""
        a = _engine(variant="guangdong", player_count=2, seed=1)
        hand = ["m1", "m2", "m4", "m5", "p1", "p1", "p1", "p2", "p3", "s1", "s2", "s3", "z5", "z5"]
        assert not _eval_win(a, hand)

    def test_wild_not_leaked_to_blood(self):
        a = _engine(variant="blood", player_count=2, seed=1)
        hand = ["m1", "m2", "m4", "m5", "p1", "p1", "p1", "p2", "p3", "s1", "s2", "s3", "z5", "z5"]
        assert not _eval_win(a, hand)

    def test_wild_not_leaked_to_taiwan(self):
        """台湾 17 张：两张 z5 补 m3/m6 在癞子规则下是胡，无癞子时不是。"""
        a = _engine(variant="taiwan", player_count=2, seed=1)
        hand = [
            "m1",
            "m2",
            "m4",
            "m5",
            "z5",
            "z5",
            "p1",
            "p2",
            "p3",
            "s1",
            "s2",
            "s3",
            "z1",
            "z1",
            "z1",
            "p9",
            "p9",
        ]
        assert not _eval_win(a, hand)


# ── Full wins through the engine ──────────────────────────────────────


class TestWins:
    def test_tsumo_win_ends_game(self):
        a = _engine(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        # Plant a ready hand for p0 (14 tiles: three runs + pair; the last
        # drawn tile is the second p3, already in the hand).
        s["_arrays"]["hand_p0"] = ["m1", "m2", "m3", "m4", "m5", "m6", "p1", "p2", "p3", "s1", "s2", "s3", "p3", "p3"]
        s["env"]["last_drawn"] = "p3"
        assert _win_legal(a, s)
        s = _act(a, s, "win_self")
        assert a.is_terminal(s)
        assert s["env"]["winner"] == "p0"
        p0, p1 = (float(x) for x in s["env"]["payoffs"])
        assert p0 == -p1 and p0 > 0

    def test_ron_win_via_claim(self):
        a = _engine(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        # p0 discards m3; p1 (13 tiles) rons with m3 completing the pair.
        s["_arrays"]["hand_p0"] = ["m3"] + s["_arrays"]["hand_p0"][1:]
        s["_arrays"]["hand_p1"] = ["m1", "m2", "m4", "m5", "m6", "p1", "p2", "p3", "s1", "s2", "s3", "z1", "z1"]
        s = _act(a, s, "discard", tile="m3")
        assert s["env"]["phase"] == "claim"
        s = _act(a, s, "claim_win", tile="m3")
        assert a.is_terminal(s)
        assert s["env"]["winner"] == "p1"

    def test_fan_pay_scale(self):
        a = _engine(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        # 14-tile ready hand: three runs + p123 + z1 pair; the drawn tile
        # (z1) is already part of the hand.
        s["_arrays"]["hand_p0"] = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "p1", "p2", "p3", "z1", "z1"]
        s["env"]["last_drawn"] = "z1"
        s = _act(a, s, "win_self")
        # 鸡胡(1) + 平胡(2, 无刻子) = 3 番 → 10 × 2^2 = 40
        assert s["env"]["fan_pay"] == 40

    def test_observation_hides_win_hand(self):
        """env.win_hand (the winner's full hand, written by do_win) must never
        appear in any player's projected observation — hidden_guard's
        blacklist rejects it (``visibility.env`` filter, v5.2)."""
        a = _engine(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s["_arrays"]["hand_p0"] = ["m1", "m2", "m3", "m4", "m5", "m6", "p1", "p2", "p3", "s1", "s2", "s3", "p3", "p3"]
        s["env"]["last_drawn"] = "p3"
        s = _act(a, s, "win_self")
        assert a.is_terminal(s)
        # do_win still records the winning hand on the raw state (fan_sum
        # needs it) …
        assert s["env"]["win_hand"] == [
            "m1",
            "m2",
            "m3",
            "m4",
            "m5",
            "m6",
            "p1",
            "p2",
            "p3",
            "s1",
            "s2",
            "s3",
            "p3",
            "p3",
        ]
        # …but no viewer's observation carries it.
        for pid in ("p0", "p1"):
            obs = a.project_observation(s, pid)
            assert "win_hand" not in obs["env"]


# ── Variants ──────────────────────────────────────────────────────────


class TestVariants:
    def test_blood_continues_after_first_win(self):
        """血流成河 (blood): 胡家不退场（done 不追加），winners 守卫禁止
        再胡；player_count-1 家胡过才终局；payoffs 跨胡**累加**而非覆写。

        手牌 m123 m456 m789 p123 p33 — 缺 s 门（blood 的缺一门 gate 要求
        少于三门），番型 鸡胡1+缺一门1 = 2 番 → fan_pay = FAN_PAY[1] = 20；
        自摸 4 人局付分人数 3 → 赢家 +60，其余各 -20。
        """
        a = _engine(variant="blood", player_count=4, seed=1)
        s = _resolve(a, a.create_initial_state())
        win_hand = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "p1", "p2", "p3", "p3", "p3"]
        # p0 tsumos (drawn p3 already in hand)
        s["_arrays"]["hand_p0"] = win_hand
        s["env"]["last_drawn"] = "p3"
        s = _act(a, s, "win_self")
        assert not a.is_terminal(s), "blood continues after one win (1 < player_count-1 = 3)"
        assert s["env"]["winners"] == ["p0"]
        assert s["env"]["done"] == [], "blood 胡家不退场（对比 sichuan 血战到底）"
        assert s["env"]["turn"] == "p1"
        assert s["env"]["phase"] == "draw"
        # 自摸 4p: p0 +20×3，其余各 -20
        assert [float(x) for x in s["env"]["payoffs"]] == [60.0, -20.0, -20.0, -20.0]
        p0_first = 60.0
        # p1 wins (2nd) — still continues (2 < 3)
        s = _resolve(a, s)
        s["_arrays"]["hand_p1"] = win_hand
        s["env"]["last_drawn"] = "p3"
        s = _act(a, s, "win_self")
        assert not a.is_terminal(s)
        assert s["env"]["winners"] == ["p0", "p1"]
        # 血战累计结算：先前胡家的分数在后续胡牌后保留（旧实现覆写归零）。
        assert float(s["env"]["payoffs"][0]) == p0_first - 20.0
        assert float(s["env"]["payoffs"][1]) == -20.0 + 60.0  # 先前 -20 + 本次自摸 +60
        # p2 wins (3rd) → player_count-1 winners reached → game over
        s = _resolve(a, s)
        s["_arrays"]["hand_p2"] = win_hand
        s["env"]["last_drawn"] = "p3"
        s = _act(a, s, "win_self")
        assert a.is_terminal(s)
        assert s["env"]["winners"] == ["p0", "p1", "p2"]
        assert s["env"]["winner"] is None, "blood 多胡局无单一 winner，由 winners 承载"
        # 跨三次胡牌的累计分（零和）：p0/p1/p2 各 +60-20-20，p3 付满 -60
        assert [float(x) for x in s["env"]["payoffs"]] == [20.0, 20.0, 20.0, -60.0]

    def test_guangdong_ends_after_one_win(self):
        a = _engine(variant="guangdong", player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s["_arrays"]["hand_p0"] = ["m1", "m2", "m3", "m4", "m5", "m6", "p1", "p2", "p3", "s1", "s2", "s3", "p3", "p3"]
        s["env"]["last_drawn"] = "p3"
        s = _act(a, s, "win_self")
        assert a.is_terminal(s)

    def test_seven_pairs_tsumo_legal(self):
        """P1-4: seven pairs tsumo must be legal and scored on the real
        14-tile hand (the old check appended last_drawn to the hand that
        already contained it, making COUNT==15 and qidui never legal)."""
        a = _engine(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        hand = ["z1", "z1", "z2", "z2", "z3", "z3", "z4", "z4", "z5", "z5", "z6", "z6", "m1", "m1"]
        s["_arrays"]["hand_p0"] = hand
        s["env"]["last_drawn"] = "m1"
        assert _win_legal(a, s)
        s = _act(a, s, "win_self")
        assert a.is_terminal(s)
        assert s["env"]["winner"] == "p0"
        # win_hand must be the 14-tile hand itself, not 15 with a duplicate.
        assert s["env"]["win_hand"] == hand
        # Engine-computed fan for this qidui hand (鸡胡+平胡+七对+绝张
        # → 8 番): FAN_PAY[7] = 1280.  The point is qidui now contributes;
        # the old 15-tile path could never reach this.
        assert s["env"]["fan_pay"] == 1280

    def test_non_winning_hand_cannot_tsumo(self):
        """P1-4: a 14-tile non-winning hand (with the drawn tile in hand)
        must not expose a legal win_self — the old 15-tile check could
        cover 14 tiles with a duplicate draw."""
        a = _engine(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        # m123 m456 p123 + p333 pung + z1 z2 singletons → no pair, not a win.
        s["_arrays"]["hand_p0"] = ["m1", "m2", "m3", "m4", "m5", "m6", "p1", "p2", "p3", "p3", "p3", "p3", "z1", "z2"]
        s["env"]["last_drawn"] = "z1"
        assert not _win_legal(a, s)

    def test_claim_phase_rotation(self):
        """审查 J1 → 规则化 (v5.2): claim 阶段行动者由环境队列决定 —
        ``claim_queue`` + ``claim_index``/``claim_mode`` 推进，效应器轮转，
        而非适配器 get_current_player 特判。"""
        a = _engine(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s["_arrays"]["hand_p0"] = ["m3"] + s["_arrays"]["hand_p0"][1:]
        s["_arrays"]["hand_p1"] = ["m3", "m3", "p1", "p2", "p3", "s1", "s2", "s3", "z1", "z2", "z3", "m1", "m2"]
        s = _act(a, s, "discard", tile="m3")
        assert s["env"]["phase"] == "claim"
        # 响应者 = 队列头（p1）；win 阶段只有 pass（p1 手牌不能胡）。
        actor = (s["env"].get("claim_queue") or [None])[int(s["env"].get("claim_index", 0))]
        assert actor == "p1"
        legal_ids = {x.template_id for x in a.get_legal_actions(s)}
        assert legal_ids == {"claim_pass"}
        # p1 pass → meld 阶段：p1 手牌有两张 m3 → 碰合法（吃仍在更后的 chi 阶段）。
        s = _act(a, s, "claim_pass")
        assert s["env"]["claim_mode"] == "meld"
        assert s["env"]["claim_index"] == 0 and s["env"]["turn"] == "p1"
        legal_ids = {x.template_id for x in a.get_legal_actions(s)}
        assert "claim_peng" in legal_ids
        # meld、chi 两阶段继续 pass → 队列耗尽 → 进入抽牌。
        s = _act(a, s, "claim_pass")
        assert s["env"]["claim_mode"] == "chi"
        s = _act(a, s, "claim_pass")
        actor = (s["env"].get("claim_queue") or [None])[int(s["env"].get("claim_index", 0))]
        assert actor is None  # 队列耗尽 → 进入抽牌
        assert s["env"]["phase"] == "draw"

    def test_wall_empty_draw(self):
        a = _engine(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s = _act(a, s, "discard", tile=s["_arrays"]["hand_p0"][0])
        # Drain the wall before the draw resolves
        s["env"]["wall_count"] = 0
        for _ in range(3):  # 胡>碰/杠>吃 三阶段全过
            assert s["env"]["phase"] == "claim"
            s = _act(a, s, "claim_pass")
        assert a.is_terminal(s)
        assert s["env"]["last_action"] == "wall_empty"
        assert a.get_utility(s, "p0") == 0.0

    def test_visibility_hides_opponent_hand(self):
        a = _engine(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        obs = a.project_observation(s, "p0")
        assert all("id" in c for c in obs["hand_view_p0"])
        assert all("id" not in c for c in obs["hand_view_p1"])
        obs1 = a.project_observation(s, "p1")
        assert all("id" in c for c in obs1["hand_view_p1"])

    def test_view_observation(self):
        """v5.2 obs is view-shaped (hand views + env), no adapter obs dict."""
        a = _engine(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        obs = a.get_observation(s, "p0")
        assert len(obs["hand_view_p0"]) == 14
        assert obs["env"]["phase"] == "action"
        assert obs["env"]["turn"] == "p0"  # my turn (env is public here)
        assert any(x.template_id == "discard" for x in a.get_legal_actions(s))


# ── v5.4 audit fixes ─────────────────────────────────────────────────


class TestV54AuditFixes:
    """回归测试：rules/mahjong.json 审计（P0/P1/P2 级缺陷）修复。"""

    def test_yibeikou_win(self):
        """P0-1: 一杯口（两个相同顺子）必须可胡 —— choose ``dedup: False`` +
        chi 池多重供应（旧版每顺子只入池一次，两个相同顺子选不出来）。"""
        a = _engine(variant="guangdong", player_count=2, seed=1)
        hand = ["m1", "m2", "m3", "m1", "m2", "m3", "p4", "p5", "p6", "s7", "s8", "s9", "z1", "z1"]
        assert _eval_win(a, hand)

    def test_erbeikou_win(self):
        """P0-1 连带: 二杯口（两对相同顺子）—— 每个顺子各需两份多重供应。"""
        a = _engine(variant="guangdong", player_count=2, seed=1)
        hand = ["m1", "m2", "m3", "m1", "m2", "m3", "p7", "p8", "p9", "p7", "p8", "p9", "z1", "z1"]
        assert _eval_win(a, hand)

    def test_sanbeikou_win(self):
        """三杯口（三个相同顺子）—— 顺子多重供应需要第三份（旧模型每顺子
        至多 2 份，三杯口及以上不可胡；v5.5 按牌池上限供应到 4 份）。"""
        a = _engine(variant="guangdong", player_count=2, seed=1)
        hand = ["m1", "m1", "m1", "m2", "m2", "m2", "m3", "m3", "m3", "p4", "p4", "p4", "z1", "z1"]
        assert _eval_win(a, hand)

    def test_sibeikou_win(self):
        """四杯口（四个相同顺子 = 牌池上限 4 份/张）—— 12 张 + 将。"""
        a = _engine(variant="guangdong", player_count=2, seed=1)
        hand = ["m1", "m1", "m1", "m1", "m2", "m2", "m2", "m2", "m3", "m3", "m3", "m3", "z1", "z1"]
        assert _eval_win(a, hand)

    def test_same_run_capped_by_supply(self):
        """供应边界：多重供应不会让不可胡的手牌误胡 —— 三份 m123 只占
        9 张，余下 5 张孤字牌凑不出副露+将。"""
        a = _engine(variant="guangdong", player_count=2, seed=1)
        three_runs_plus_orphans = [
            "m1",
            "m1",
            "m1",
            "m2",
            "m2",
            "m2",
            "m3",
            "m3",
            "m3",
            "z1",
            "z2",
            "z3",
            "z4",
            "z5",
        ]
        assert not _eval_win(a, three_runs_plus_orphans)

    def test_ron_bao_gong_discarder_pays_all(self):
        """P0-2: 荣和包铳 —— 点炮者独付全部份额，其余玩家 0 分
        （旧版荣和与自摸同构，无辜玩家一起扣分 [-40,120,-40,-40]）。"""
        a = _engine(variant="guangdong", player_count=4, seed=1)
        s = _resolve(a, a.create_initial_state())
        # p0 discards z1; p1 rons (m123 m456 m789 p123 + z1z1).
        s["_arrays"]["hand_p0"] = ["z1"] + s["_arrays"]["hand_p0"][1:]
        s["_arrays"]["hand_p1"] = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "p1", "p2", "p3", "z1"]
        s = _act(a, s, "discard", tile="z1")
        assert s["env"]["phase"] == "claim"
        s = _act(a, s, "claim_win", tile="z1")
        assert a.is_terminal(s)
        assert s["env"]["winner"] == "p1"
        # 鸡胡1 + 平胡2 = 3 番 → fan_pay 40；包铳 4 人局：
        # p1 +40×3、点炮者 p0 −120、无关的 p2/p3 为 0。
        assert [float(x) for x in s["env"]["payoffs"]] == [-120.0, 120.0, 0.0, 0.0]

    def test_melded_pengpenghu_scores_full_hand(self):
        """P0-4: 副露碰碰胡按整手（暗手 ∪ 副露）计番 —— 旧版 win_hand 只含
        暗手，11 张暗手 + 碰出来的 z222 只按鸡胡 1 番结算。"""
        a = _engine(variant="guangdong", player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s["_arrays"]["hand_p0"] = ["m1", "m1", "m1", "p4", "p4", "p4", "s7", "s7", "s7", "z1", "z1"]
        s["_arrays"]["melds_p0"] = [{"type": "peng", "tiles": ["z2", "z2", "z2"], "from": "p1"}]
        s["env"]["last_drawn"] = "z1"
        s = _act(a, s, "win_self")
        # 整手 4 刻子 + 将 → 碰碰胡 3 番 + 鸡胡 1 = 4 番 → FAN_PAY[3] = 80。
        assert s["env"]["fan_pay"] == 80

    def test_thirteen_orphans_bidirectional(self):
        """P1-6: 十三幺双向 —— 13 张幺九 + 1 张普通牌不能骗和（旧版只查
        幺九 ⊆ 手牌，漏了手牌 ⊆ 幺九）。"""
        a = _engine(variant="guangdong", player_count=2, seed=1)
        bogus = ["m1", "m9", "p1", "p9", "s1", "s9", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "m5"]
        assert not _eval_win(a, bogus)
        real = ["m1", "m9", "p1", "p9", "s1", "s9", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "m1"]
        assert _eval_win(a, real)

    def test_longqidui_wins(self):
        """P1-7: 龙七对（一组四张 + 五对）可胡 —— 旧版 is_qidui 只认纯七对。"""
        a = _engine(variant="guangdong", player_count=2, seed=1)
        hand = ["m1", "m1", "m1", "m1", "m2", "m2", "m3", "m3", "p4", "p4", "s7", "s7", "z2", "z2"]
        assert _eval_win(a, hand)

    def test_fan_qidui_pure_pairs_only(self):
        """v5.4: 七对番只认纯七对；龙七对独走 fan_longqidui（四川 8 番），
        不再 4+8 双计。"""
        a = _engine(variant="sichuan", player_count=2, seed=1)
        ctx = a._build_context(a.create_initial_state())  # noqa: SLF001
        longq = ["m1", "m1", "m1", "m1", "m2", "m2", "m3", "m3", "m4", "m4", "p4", "p4", "p8", "p8"]
        fan_sum = a.expr.eval({"call": ["fan_sum", {"const": longq}, {"const": "p0"}, {"const": True}]}, ctx)
        assert fan_sum == 8, "龙七对只计 fan_longqidui(8)，fan_qidui 不叠加"
        # 缺一门 (m+p 两门) → 该手牌在四川也确实可胡。
        assert _eval_win(a, longq)

    def test_pure_honors_not_qingyise(self):
        """P2-14: 纯字牌手不计清一色（旧版按首字符匹配，z 开头的字牌
        全部互相"同花色"而误判）。"""
        a = _engine(variant="guangdong", player_count=2, seed=1)
        ctx = a._build_context(a.create_initial_state())  # noqa: SLF001
        honors = ["z1", "z1", "z1", "z2", "z2", "z2", "z3", "z3", "z3", "z4", "z4", "z4", "z5", "z5"]
        qing = a.expr.eval({"call": ["fan_qingyise", {"const": honors}]}, ctx)
        assert not qing
        # 对照：真清一色（纯 m 门）为真。
        pure_m = ["m1", "m1", "m2", "m2", "m3", "m3", "m4", "m4", "m5", "m5", "m6", "m6", "m7", "m7"]
        assert a.expr.eval({"call": ["fan_qingyise", {"const": pure_m}]}, ctx)

    def test_gang_tiles_env_field_declared(self):
        """P2-12: gang_tiles 在 groundState 声明（do_gang_added 的过渡键，
        旧版未声明导致 env 结构漂移）。"""
        with open(RULES_PATH, encoding="utf-8") as f:
            rules = json.load(f)
        fields = rules["groundState"]["env"]["fields"]
        assert fields["gang_tiles"] == {"type": "list", "initial": []}
        a = _engine(variant="guangdong", player_count=2, seed=1)
        s = a.create_initial_state()
        assert s["env"]["gang_tiles"] == []

    def test_blood_winner_cannot_win_twice(self):
        """血流成河 winners 守卫：胡过的玩家 win_self 不再合法（但仍正常
        摸打）—— blood 的防二胡机制（对比 sichuan 的 done 退场）。"""
        a = _engine(variant="blood", player_count=4, seed=1)
        s = _resolve(a, a.create_initial_state())
        win_hand = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "p1", "p2", "p3", "p3", "p3"]
        s["_arrays"]["hand_p0"] = win_hand
        s["env"]["last_drawn"] = "p3"
        s = _act(a, s, "win_self")  # winners = [p0]，局继续
        assert s["env"]["winners"] == ["p0"]
        # p1/p2/p3 各摸一张弃一张（其余三家手牌替换成碰不了 m1 的孤立牌），
        # 轮转一圈回到 p0 再摸一张。
        for pid in ("p1", "p2", "p3"):
            s["_arrays"][f"hand_{pid}"] = [
                "p9",
                "p9",
                "s1",
                "s1",
                "s2",
                "s2",
                "s3",
                "s3",
                "s4",
                "s4",
                "s5",
                "s5",
                "s6",
            ]
        while True:
            s = _resolve(a, s)  # draw for the current turn
            turn = s["env"]["turn"]
            s = _act(a, s, "discard", tile=s["_arrays"][f"hand_{turn}"][0])
            while s["env"]["phase"] == "claim":
                s = _act(a, s, "claim_pass")
            if turn == "p3":
                break
        s = _resolve(a, s)  # p0 draws again
        # p0 再拿到一副可胡手牌 —— win_self 必须被 winners 守卫拦下。
        s["_arrays"]["hand_p0"] = win_hand
        s["env"]["last_drawn"] = "p3"
        assert not _win_legal(a, s), "blood 胡家不得二次胡牌"
        ids = {x.template_id for x in a.get_legal_actions(s)}
        assert "discard" in ids, "胡家仍正常摸打"
