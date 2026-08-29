"""Tests for the v5.3 mahjong variants: sichuan (血战到底), changsha
(258将), taiwan (16张, no-flower simplification).

All variants live in the same declarative ``rules/mahjong.json``
(``variants.options`` patching constants + ``$constants.variant``
conditional rules); the engine only selects declared data — nothing here
depends on per-game adapters.

Covered:
  - sichuan: 108 no-honor deck, 缺一门 win gate, no chi, continuation
    ends at player_count-1 done, wall-empty on a 108-tile deck.
  - changsha: 108 no-honor deck, 小胡 258将, 大胡 (碰碰胡/清一色/七对/
    将将胡) 乱将, 将将胡 structure-exempt, 1/6/12 番 pay mapping.
  - taiwan: 17-tile deal (2p), 5 melds + pair, 呖咕呖咕 (7 pairs +
    1 triplet), no 14-tile seven pairs.
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


def _resolve(adapter: GameEngine, state: dict) -> dict:
    while adapter.get_node_type(state) == "chance":
        _, state = adapter.sample_chance(state)
    return state


def _act(adapter: GameEngine, state: dict, template: str, **params) -> dict:
    for a in adapter.get_legal_actions(state):
        if a.template_id == template and all(a.params.get(k) == v for k, v in params.items()):
            return adapter.apply_action(state, a)
    raise AssertionError(f"not legal: {template} {params} at {state['env']['phase']}")


def _eval_win(adapter: GameEngine, hand: list, melds: list | None = None) -> bool:
    ctx = adapter._build_context(adapter.create_initial_state())  # noqa: SLF001
    return bool(adapter.expr.eval({"call": ["is_win_hand", {"const": hand}, {"const": melds or []}]}, ctx))


def _tsumo_win(adapter: GameEngine, state: dict, hand: list, drawn: str, pid: str = "p0") -> dict:
    """Plant a ready hand (player ``pid``) whose drawn tile is ``drawn``, then tsumo."""
    state["_arrays"][f"hand_{pid}"] = hand
    state["env"]["last_drawn"] = drawn
    return _act(adapter, state, "win_self")


class TestSichuan:
    def test_108_deck_no_honors(self):
        a = _engine(variant="sichuan", player_count=2, seed=1)
        assert len(a._constants["tile_ids"]) == 108  # noqa: SLF001
        s = _resolve(a, a.create_initial_state())
        tiles = s["_arrays"]["hand_p0"] + s["_arrays"]["hand_p1"]
        assert not any(t.startswith("z") for t in tiles)
        assert len(s["_arrays"]["hand_p0"]) == 14

    def test_missing_suit_gate(self):
        a = _engine(variant="sichuan", player_count=2, seed=1)
        # Three suits → not 缺一门 → no win.
        three_suits = ["m1", "m2", "m3", "m4", "m5", "m6", "p1", "p2", "p3", "s1", "s2", "s3", "m7", "m7"]
        assert not _eval_win(a, three_suits)
        # Two suits → 缺一门 → win.
        two_suits = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "p1", "p2", "p3", "m5", "m5"]
        assert _eval_win(a, two_suits)
        # 清一色 (one suit) → trivially 缺一门 → win (pung structure).
        one_suit = ["m1", "m1", "m1", "m2", "m2", "m2", "m3", "m3", "m3", "m4", "m4", "m5", "m5", "m5"]
        assert _eval_win(a, one_suit)
        # 七对 spread over three suits → not 缺一门 → no win.
        seven_pairs_3s = ["m1", "m1", "m2", "m2", "m3", "m3", "p1", "p1", "p2", "p2", "s1", "s1", "s2", "s2"]
        assert not _eval_win(a, seven_pairs_3s)
        # 七对 over two suits → 缺一门 → win.
        seven_pairs_2s = ["m1", "m1", "m2", "m2", "m3", "m3", "p1", "p1", "p2", "p2", "p3", "p3", "m4", "m4"]
        assert _eval_win(a, seven_pairs_2s)

    def test_haidilaoyue_fan_pay_e2e(self):
        # 海底捞月 8 番（+ 平胡 1 = 9 番 → clamp 1280），墙空时自摸。
        a = _engine(variant="sichuan", player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s["env"]["wall_count"] = 0
        s = _tsumo_win(a, s, ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "p1", "p2", "p3", "m5", "m5"], "m5")
        assert s["env"]["fan_pay"] == 1280

    def test_haidilaoyue_zero_for_other_variants(self):
        a = _engine(variant="guangdong", player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s["env"]["wall_count"] = 0
        s = _tsumo_win(a, s, ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "p1", "p2", "p3", "z1", "z1"], "z1")
        # 鸡胡 1 + 平胡 2 = 3 番 → 40（海底捞月对非四川不计番）。
        assert s["env"]["fan_pay"] == 40

    def test_no_chi_in_claim_phase(self):
        a = _engine(variant="sichuan", player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s = _act(a, s, "discard", tile=s["_arrays"]["hand_p0"][0])
        assert s["env"]["phase"] == "claim"
        legal_ids = {x.template_id for x in a.get_legal_actions(s)}
        assert "claim_chi" not in legal_ids
        assert "claim_pass" in legal_ids

    def test_blood_ends_at_player_count_minus_one(self):
        a = _engine(variant="sichuan", player_count=2, seed=1)
        assert a._constants["variant"] == "sichuan"  # noqa: SLF001
        assert a._constants["player_count"] == 2  # noqa: SLF001
        s = _resolve(a, a.create_initial_state())
        # 2p 血战到底: one win ends the round (player_count - 1 == 1).
        # Two-suit hand (缺一门 must hold for the win to be legal).
        hand = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "p1", "p2", "p3", "p3", "p3"]
        s = _tsumo_win(a, s, hand, "p3")
        assert a.is_terminal(s)

    def test_blood_4p_ends_at_three_done(self):
        a = _engine(variant="sichuan", player_count=4, seed=1)
        s = _resolve(a, a.create_initial_state())
        # p0 wins → continues; two more wins → three done → over.
        s = _tsumo_win(a, s, ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "p1", "p2", "p3", "p3", "p3"], "p3")
        assert not a.is_terminal(s)
        assert s["env"]["done"] == ["p0"]
        p0_first = float(s["env"]["payoffs"][0])
        assert p0_first > 0, "p0 首胡应得三家支付的正分"
        # p1 tsumo (two-suit hand)
        s = _resolve(a, s)
        s = _tsumo_win(
            a, s, ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "p2", "p3", "p4", "p4", "p4"], "p4", pid="p1"
        )
        assert not a.is_terminal(s)
        # 累计结算：p1 胡后 p0（已 done）的分数必须原样保留。
        assert float(s["env"]["payoffs"][0]) == p0_first
        # p2 tsumo → three done → over (two-suit hand)
        s = _resolve(a, s)
        s = _tsumo_win(
            a, s, ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "s1", "s2", "s3", "s3", "s3"], "s3", pid="p2"
        )
        assert a.is_terminal(s)
        assert s["env"]["done"] == ["p0", "p1", "p2"]
        # 血战累计结算终态：先前胡家分数保留、三家胡牌、四人零和。
        payoffs = [float(x) for x in s["env"]["payoffs"]]
        assert payoffs[0] == p0_first
        assert payoffs[1] > 0, "第二位胡家 p1 累计应为正（两家中未 done 的 p2/p3 支付）"
        assert s["env"]["winners"] == ["p0", "p1", "p2"]
        assert s["env"]["winner"] is None
        assert abs(sum(payoffs)) < 1e-9, "四次支付累计应零和"

    def test_wall_empty_on_108_deck(self):
        a = _engine(variant="sichuan", player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s = _act(a, s, "discard", tile=s["_arrays"]["hand_p0"][0])
        # Drain the wall by marking all 108 kinds drawn (ground array, bound
        # as $drawn — the wall-empty deck arm reads $drawn, not env.drawn).
        s["_arrays"]["drawn"] = list(a._constants["tile_ids"])  # noqa: SLF001
        while a.get_node_type(s) != "terminal":  # 胡>碰/杠>吃 阶段全过
            assert s["env"]["phase"] == "claim"
            s = _act(a, s, "claim_pass")
        assert s["env"]["last_action"] == "wall_empty"


class TestChangsha:
    def test_108_deck_no_honors(self):
        a = _engine(variant="changsha", player_count=2, seed=1)
        assert len(a._constants["tile_ids"]) == 108  # noqa: SLF001
        s = _resolve(a, a.create_initial_state())
        assert not any(t.startswith("z") for t in s["_arrays"]["hand_p0"])

    def test_xiaohu_requires_258_pair(self):
        a = _engine(variant="changsha", player_count=2, seed=1)
        # Standard 小胡 shape with a non-258 pair (m1) → not legal.
        hand_bad = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "p1", "p2", "p3", "m1", "m1"]
        assert not _eval_win(a, hand_bad)
        # Same shape with a 258 pair (m5) → legal 小胡.
        hand_ok = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "p1", "p2", "p3", "m5", "m5"]
        assert _eval_win(a, hand_ok)

    def test_dahu_any_pair(self):
        a = _engine(variant="changsha", player_count=2, seed=1)
        # 碰碰胡 with a non-258 pair → 乱将 → legal.
        peng = ["m1", "m1", "m1", "m2", "m2", "m2", "p3", "p3", "p3", "s4", "s4", "s4", "m5", "m5"]
        assert _eval_win(a, peng)
        # 清一色 with non-258 pair → legal (runs + pung mixed structure;
        # v5.4: the chi pool supplies each run once per copy, so 一杯口
        # shapes are also selectable).
        qing = ["m1", "m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m6", "m6", "m7", "m8", "m9"]
        assert _eval_win(a, qing)
        # Plain standard hand with non-258 pair and no 大胡 pattern → illegal.
        plain = ["m1", "m2", "m3", "m4", "m5", "m6", "p1", "p2", "p3", "s1", "s2", "s3", "m1", "m1"]
        assert not _eval_win(a, plain)

    def test_seven_pairs_exempt(self):
        a = _engine(variant="changsha", player_count=2, seed=1)
        # 七对 with arbitrary pairs (none 258) → exempt → legal.
        qidui = ["m1", "m1", "m3", "m3", "p4", "p4", "s5", "s5", "s6", "s6", "m7", "m7", "p9", "p9"]
        assert _eval_win(a, qidui)

    def test_jiangjianghu_structure_exempt(self):
        a = _engine(variant="changsha", player_count=2, seed=1)
        # All tiles 2/5/8 but NOT decomposable into 4 melds + pair.
        hand = ["m2", "m2", "m2", "m2", "m2", "m2", "m5", "m5", "m5", "m5", "m5", "m5", "m8", "m8"]
        assert _eval_win(a, hand)

    def test_fan_pay_mapping(self):
        a = _engine(variant="changsha", player_count=2, seed=1)
        # 小胡 (平胡 1 番) → pay 10.
        s = _resolve(a, a.create_initial_state())
        s = _tsumo_win(a, s, ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "p1", "p2", "p3", "m5", "m5"], "m5")
        assert s["env"]["fan_pay"] == 10
        # 清一色 (6) + 七对 (6) = 12 → 番上番 120 (pair of m5, qidui exempt).
        a2 = _engine(variant="changsha", player_count=2, seed=1)
        s2 = _resolve(a2, a2.create_initial_state())
        qing_qidui = ["m2", "m2", "m5", "m5", "m8", "m8", "m1", "m1", "m3", "m3", "m4", "m4", "m6", "m6"]
        s2 = _tsumo_win(a2, s2, qing_qidui, "m6")
        assert s2["env"]["fan_pay"] == 120

    def test_situational_fans_env_gated(self):
        a = _engine(variant="changsha", player_count=2, seed=1)
        # 杠上开花: win right after a gang action (last_action = "gang").
        s = _resolve(a, a.create_initial_state())
        s["env"]["last_action"] = "gang"
        s = _tsumo_win(a, s, ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "p1", "p2", "p3", "m5", "m5"], "m5")
        # 平胡 1 + 杠上开花 6 = 7 → 单大胡 band 60.
        assert s["env"]["fan_pay"] == 60
        # 海底捞月: 小胡 with wall_count 0 → 1+6=7 → 60.
        a2 = _engine(variant="changsha", player_count=2, seed=1)
        s2 = _resolve(a2, a2.create_initial_state())
        s2["env"]["wall_count"] = 0
        s2 = _tsumo_win(
            a2, s2, ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "p1", "p2", "p3", "m5", "m5"], "m5"
        )
        assert s2["env"]["fan_pay"] == 60


class TestTaiwan:
    def test_deal_17_16(self):
        a = _engine(variant="taiwan", player_count=2, seed=1)
        assert a._constants["win_tiles"] == 17  # noqa: SLF001
        assert a._constants["meld_k"] == 5  # noqa: SLF001
        s = _resolve(a, a.create_initial_state())
        assert len(s["_arrays"]["hand_p0"]) == 17
        assert len(s["_arrays"]["hand_p1"]) == 16
        assert s["env"]["wall_count"] == 136 - 33

    def test_five_melds_plus_pair(self):
        a = _engine(variant="taiwan", player_count=2, seed=1)
        # 5 runs + pair = 17 tiles → win.
        hand = [
            "m1",
            "m2",
            "m3",
            "m4",
            "m5",
            "m6",
            "m7",
            "m8",
            "m9",
            "p1",
            "p2",
            "p3",
            "p4",
            "p5",
            "p6",
            "z1",
            "z1",
        ]
        assert _eval_win(a, hand)
        # 4 runs + pair = 14 tiles → NOT a taiwan win.
        short = [
            "m1",
            "m2",
            "m3",
            "m4",
            "m5",
            "m6",
            "m7",
            "m8",
            "m9",
            "p1",
            "p2",
            "p3",
            "z1",
            "z1",
        ]
        assert not _eval_win(a, short)

    def test_licu_seven_pairs_plus_triplet(self):
        a = _engine(variant="taiwan", player_count=2, seed=1)
        # 呖咕呖咕: 7 pairs + 1 triplet = 17 tiles.
        licu = [
            "m1",
            "m1",
            "m2",
            "m2",
            "m3",
            "m3",
            "p4",
            "p4",
            "p5",
            "p5",
            "s6",
            "s6",
            "z2",
            "z2",
            "z7",
            "z7",
            "z7",
        ]
        assert _eval_win(a, licu)

    def test_no_14_tile_seven_pairs(self):
        a = _engine(variant="taiwan", player_count=2, seed=1)
        qidui14 = ["m1", "m1", "m2", "m2", "p3", "p3", "s4", "s4", "z1", "z1", "z2", "z2", "z3", "z3"]
        assert not _eval_win(a, qidui14)

    def test_default_variants_unchanged(self):
        # Cross-check: guangdong keeps the 14-tile win semantics.
        a = _engine(variant="guangdong", player_count=2, seed=1)
        qidui14 = ["m1", "m1", "m2", "m2", "p3", "p3", "s4", "s4", "z1", "z1", "z2", "z2", "z3", "z3"]
        assert _eval_win(a, qidui14)
        std14 = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "p1", "p2", "p3", "p3", "p3"]
        assert _eval_win(a, std14)

    def test_tsumo_pinghu_menqing_selfdraw_fan_pay(self):
        # 平胡 2 + 门清 1 + 自摸 1 = 4 台 → 10 × 2^3 = 80.
        a = _engine(variant="taiwan", player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        hand = [
            "m1",
            "m2",
            "m3",
            "m4",
            "m5",
            "m6",
            "m7",
            "m8",
            "m9",
            "p1",
            "p2",
            "p3",
            "p4",
            "p5",
            "p6",
            "z1",
            "z1",
        ]
        s = _tsumo_win(a, s, hand, "z1")
        assert s["env"]["fan_pay"] == 80

    def test_ron_pinghu_menqing_fan_pay(self):
        # 荣和: 平胡 2 + 门清 1 = 3 台 → 40 (no 自摸).
        a = _engine(variant="taiwan", player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s["_arrays"]["hand_p0"] = ["z1"] + s["_arrays"]["hand_p0"][1:]
        s["_arrays"]["hand_p1"] = [
            "m1",
            "m2",
            "m3",
            "m4",
            "m5",
            "m6",
            "m7",
            "m8",
            "m9",
            "p1",
            "p2",
            "p3",
            "p4",
            "p5",
            "p6",
            "z1",
        ]
        s = _act(a, s, "discard", tile="z1")
        assert s["env"]["phase"] == "claim"
        s = _act(a, s, "claim_win", tile="z1")
        assert a.is_terminal(s)
        assert s["env"]["fan_pay"] == 40

    def test_tsumo_pengpenghu_fan_pay(self):
        # 碰碰胡 4 + 门清 1 + 自摸 1 = 6 台 → 320.
        a = _engine(variant="taiwan", player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        hand = [
            "m1",
            "m1",
            "m1",
            "m2",
            "m2",
            "m2",
            "p3",
            "p3",
            "p3",
            "s4",
            "s4",
            "s4",
            "z5",
            "z5",
            "z5",
            "m5",
            "m5",
        ]
        s = _tsumo_win(a, s, hand, "m5")
        assert s["env"]["fan_pay"] == 320
        # 清一色 8 + 门清 1 + 自摸 1 = 10 台 → clamp 1280.
        a2 = _engine(variant="taiwan", player_count=2, seed=1)
        s2 = _resolve(a2, a2.create_initial_state())
        qing = [
            "m1",
            "m1",
            "m1",
            "m1",
            "m2",
            "m2",
            "m2",
            "m3",
            "m4",
            "m5",
            "m5",
            "m5",
            "m5",
            "m6",
            "m7",
            "m8",
            "m9",
        ]
        s2 = _tsumo_win(a2, s2, qing, "m9")
        assert s2["env"]["fan_pay"] == 1280

    def test_tsumo_ligu_fan_pay(self):
        # 呖咕呖咕 (mixed): 门清 1 + 自摸 1 = 2 台 → 20.
        a = _engine(variant="taiwan", player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        licu = [
            "m1",
            "m1",
            "m2",
            "m2",
            "m3",
            "m3",
            "p4",
            "p4",
            "p5",
            "p5",
            "s6",
            "s6",
            "z2",
            "z2",
            "z7",
            "z7",
            "z7",
        ]
        s = _tsumo_win(a, s, licu, "z7")
        assert s["env"]["fan_pay"] == 20


# ── v5.4 audit fixes (blood 血流成河 / changsha 番表) ────────────────


class TestBloodV54:
    """blood = 血流成河：108 张无字牌、缺一门、无吃、胡家不退场。

    （对比 sichuan 血战到底：胡家退场进 done；blood 靠 winners 守卫禁止
    二胡，终局条件统一为 player_count-1 家胡过或牌墙抽干。）
    """

    TWO_SUIT_WIN = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "p1", "p2", "p3", "m5", "m5"]

    def test_blood_deck_no_honors(self):
        """blood 牌池 = 108 张无字牌（旧版用 136 张默认牌池）。"""
        a = _engine(variant="blood", player_count=2, seed=1)
        assert len(a._constants["tile_ids"]) == 108  # noqa: SLF001
        assert not any(t.startswith("z") for t in a._constants["tile_ids"])  # noqa: SLF001
        s = _resolve(a, a.create_initial_state())
        assert not any(t.startswith("z") for t in s["_arrays"]["hand_p0"])

    def test_blood_missing_suit_gate(self):
        """blood 与四川同款缺一门 gate：三门手牌不胡，两门胡。"""
        a = _engine(variant="blood", player_count=2, seed=1)
        three_suits = ["m1", "m2", "m3", "m4", "m5", "m6", "p1", "p2", "p3", "s1", "s2", "s3", "m7", "m7"]
        assert not _eval_win(a, three_suits)
        assert _eval_win(a, self.TWO_SUIT_WIN)

    def test_blood_no_chi(self):
        """blood 禁吃（川麻无吃，旧版漏了 blood）。"""
        a = _engine(variant="blood", player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s = _act(a, s, "discard", tile=s["_arrays"]["hand_p0"][0])
        assert s["env"]["phase"] == "claim"
        legal_ids = {x.template_id for x in a.get_legal_actions(s)}
        assert "claim_chi" not in legal_ids

    def test_changsha_no_chi(self):
        """changsha 禁吃（长沙麻将只能碰杠，旧版漏了 changsha）。"""
        a = _engine(variant="changsha", player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s = _act(a, s, "discard", tile=s["_arrays"]["hand_p0"][0])
        legal_ids = {x.template_id for x in a.get_legal_actions(s)}
        assert "claim_chi" not in legal_ids

    def test_blood_haidilaoyue(self):
        """blood 海底捞月 8 番（鸡胡1 + 缺一门1 + 海底8 = 10 → clamp 1280）。"""
        a = _engine(variant="blood", player_count=4, seed=1)
        s = _resolve(a, a.create_initial_state())
        s["env"]["wall_count"] = 0
        s = _tsumo_win(a, s, self.TWO_SUIT_WIN, "m5")
        assert s["env"]["fan_pay"] == 1280

    def test_haidilaoyue_deck_exhausted_arm(self):
        """P1-10: 108 张牌型的海底捞月走牌池抽干臂（旧版只查
        wall_count==0，而 wall_count 初值 136 在 108 张局永远到不了 0）。"""
        a = _engine(variant="blood", player_count=4, seed=1)
        s = _resolve(a, a.create_initial_state())
        # 不动 wall_count（保持 136 初值减已发牌数），只把 drawn 抬满牌池：
        # drawn 已含发牌 + 本局摸牌，补齐到全部 108 张。
        s["_arrays"]["drawn"] = list(a._constants["tile_ids"])  # noqa: SLF001
        s = _tsumo_win(a, s, self.TWO_SUIT_WIN, "m5")
        assert s["env"]["fan_pay"] == 1280


class TestChangshaV54:
    def test_fan_12_pays_120(self):
        """P1-9: changsha 番表 12 番 → 120（旧表 [60]*7 在 idx 11 给 60，
        番上番被压成单大胡）。清一色碰碰胡 = 6+6 = 12 番。"""
        a = _engine(variant="changsha", player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        hand = ["m1", "m1", "m1", "m2", "m2", "m2", "m3", "m3", "m3", "m4", "m4", "m4", "m5", "m5"]
        s = _tsumo_win(a, s, hand, "m5")
        assert s["env"]["fan_pay"] == 120
