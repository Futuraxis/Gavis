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
        # both pass → draw
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
        s = _act(a, s, "claim_pass")
        s = _act(a, s, "claim_pass")
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

    def test_hongzhong_wild_fills_gap(self):
        a = _engine(variant="hongzhong", player_count=2, seed=1)
        # Two red dragons stand in for the missing m3 and m6
        hand = ["m1", "m2", "m4", "m5", "p1", "p1", "p1", "p2", "p3", "s1", "s2", "s3", "z5", "z5"]
        assert _eval_win(a, hand)

    def test_hongzhong_wild_not_enough(self):
        a = _engine(variant="hongzhong", player_count=2, seed=1)
        hand = ["m1", "m2", "m4", "m5", "p1", "p1", "p1", "p2", "p3", "s1", "s2", "s3", "z5"]
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
        a = _engine(variant="blood", player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        # p0 tsumos (drawn tile p3 already in hand)
        s["_arrays"]["hand_p0"] = ["m1", "m2", "m3", "m4", "m5", "m6", "p1", "p2", "p3", "s1", "s2", "s3", "p3", "p3"]
        s["env"]["last_drawn"] = "p3"
        s = _act(a, s, "win_self")
        assert not a.is_terminal(s), "blood continues after one win"
        assert s["env"]["done"] == ["p0"]
        assert s["env"]["turn"] == "p1"
        assert s["env"]["phase"] == "draw"
        # p1 draws, then tsumos → two done → over
        s = _resolve_after(a, s)
        s["_arrays"]["hand_p1"] = ["m1", "m2", "m3", "m4", "m5", "m6", "p1", "p2", "p3", "s1", "s2", "s3", "z2", "z2"]
        s["env"]["last_drawn"] = "z2"
        s = _act(a, s, "win_self")
        assert a.is_terminal(s)
        assert s["env"]["done"] == ["p0", "p1"]

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
        ``claim_queue`` + ``claim_index`` 推进，效应器轮转，而非适配器
        get_current_player 特判。"""
        a = _engine(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s["_arrays"]["hand_p0"] = ["m3"] + s["_arrays"]["hand_p0"][1:]
        s["_arrays"]["hand_p1"] = ["m3", "m3", "p1", "p2", "p3", "s1", "s2", "s3", "z1", "z2", "z3", "m1", "m2"]
        s = _act(a, s, "discard", tile="m3")
        assert s["env"]["phase"] == "claim"
        # 响应者 = 队列头（p1），claim 选项对响应者合法
        actor = (s["env"].get("claim_queue") or [None])[int(s["env"].get("claim_index", 0))]
        assert actor == "p1"
        legal_ids = {x.template_id for x in a.get_legal_actions(s)}
        assert "claim_peng" in legal_ids
        # 弃牌者继续推进队列 → 轮到它时不再有 claim 选项（等下一响应者）
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
