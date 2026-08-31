"""Tests for hidden_guard (layer4_interface/agent/hidden_guard.py).

Covers the 2026-08 leak-fix additions:

- ``_RANK_ONLY_HOLD``: 无花色前缀的**具体单牌持牌**表述（「手里有张K」
  「拿着一张3」）默认/对手模式都拦；牌力措辞（「一对K」「同花」）不误伤；
- UNO 扫描：六变体规则表 + 具体牌面记法（红5 / 蓝禁止 / +2 / 万能）
  default/teaching/adversarial 三态；
- ``resolve_scan_game``：显式内置 ``game_id`` 优先于观测推断（UNO 与麻将
  视图名撞前缀的既有缺口），custom id 回退观测推断，双缺失 → unknown
  （扫描 fail-soft 跳过）；
- ``HIDDEN_FIELDS``：UNO 高人数变体（2-10 人）的 ``hand_p4…hand_p9``
  隐藏数组已列入黑名单。
"""

from __future__ import annotations

from layer4_interface.agent.hidden_guard import (
    HIDDEN_FIELDS,
    resolve_scan_game,
    scan,
)

# ── rank-only specific single-card holdings ──────────────────────────


class TestRankOnlyHold:
    def test_default_blocks_rank_only_hold(self) -> None:
        """默认（全拦）：无花色前缀的持牌表述也改写（「手里有张K」）。"""
        out = scan("我手里有张K，这手稳了。", "texas_holdem")
        assert "K" not in out.split("。")[0]
        assert "不细说" in out

    def test_default_blocks_various_hold_frames(self) -> None:
        """「手上有张」「拿着一张」「有一张」「正好是」等持牌框架都拦。"""
        for phrase in ("我手上有张3", "他拿着一张5", "你手里有10", "正好是张J"):
            out = scan(f"{phrase}。", "texas_holdem")
            assert "不细说" in out, phrase

    def test_adversarial_blocks_rank_only_hold(self) -> None:
        """对手模式：AI 报**具体单牌点数**（无花色）也是明牌，照拦。"""
        out = scan("我手里有张K，稳一点。", "texas_holdem", adversarial=True)
        assert "不细说" in out

    def test_hand_strength_phrases_not_blocked(self) -> None:
        """牌力措辞不误伤：「一对K」「同花」「三条」无 张/拿着 框架，放行。"""
        out = scan("我手里一对K，这手可以压一压你。", "texas_holdem", adversarial=True)
        assert out == "我手里一对K，这手可以压一压你。"
        out = scan("我这手同花不算大。", "texas_holdem")
        assert out == "我这手同花不算大。"


# ── UNO scan coverage ────────────────────────────────────────────────


class TestUnoScan:
    def test_default_blocks_specific_uno_cards(self) -> None:
        """默认（全拦）：具体牌面（红5/蓝禁止/+2/万能）与「手牌」都改写。"""
        for phrase in ("我的手里有一张红5。", "他打出蓝禁止。", "我手里有张+2。", "我有一张万能四。"):
            out = scan(phrase, "uno")
            assert "不细说" in out, phrase

    def test_default_blocks_hand_word(self) -> None:
        """「手牌」持牌措辞在 UNO 默认态同样拦。"""
        out = scan("我的手牌还不错。", "uno")
        assert "不细说" in out

    def test_teaching_blocks_opponent_hand_allows_player(self) -> None:
        """教学：拦「我的/AI 的 + 手牌」（教练不可知对手牌）；放行玩家自己的牌。"""
        out = scan("我的手牌是清一色红。你的手牌里有张红5。", "uno", teaching=True)
        assert "不细说" in out.split("你的")[0]
        assert "你的手牌里有张红5。" in out

    def test_adversarial_blocks_player_hand_and_specific_cards(self) -> None:
        """对手模式：拦「你的/玩家的 + 手牌」；AI 报具体牌面（红5）也是明牌，照拦。"""
        out = scan("你的手牌里有张红5。我的手牌不小。", "uno", adversarial=True)
        assert "不细说" in out.split("我的手牌")[0]
        assert "我的手牌不小。" in out

    def test_dispatch_via_resolve_scan_game(self) -> None:
        """chat/_guard 用 resolve_scan_game(spec.game_id) 分派后 UNO 变体照扫。"""
        resolved = resolve_scan_game("uno_stacking", {})
        assert resolved == "uno_stacking"
        out = scan("我手里有一张红5。", resolved)
        assert "不细说" in out


# ── resolve_scan_game: builtin id wins over observation inference ────


class TestResolveScanGame:
    def test_builtin_uno_beats_mahjong_inference(self) -> None:
        """UNO 视图名与麻将撞前缀（hand_view_*）：显式内置 id 优先，不再错判麻将。"""
        obs = {"hand_view_p0": {"cards": ["r5"]}}
        assert resolve_scan_game("uno", obs) == "uno"
        assert resolve_scan_game("uno_strict_wild4", obs) == "uno_strict_wild4"

    def test_inference_fallback_without_builtin(self) -> None:
        """custom/未知 id：回退观测形态推断（德州/麻将视图名仍可识别）。"""
        assert resolve_scan_game("custom_texas_v2", {"sb_hole_view": {}}) == "texas_holdem"
        assert resolve_scan_game("custom_uno_x", {"hand_view_p0": {}}) == "mahjong"
        assert resolve_scan_game("moon_chess", {}) == "unknown"

    def test_scan_skips_unknown(self) -> None:
        """双缺失（custom id + 无视图名）→ unknown → 扫描 fail-soft 原样放行。"""
        out = scan("我手里有张K。", resolve_scan_game("moon_chess", {}))
        assert out == "我手里有张K。"


# ── HIDDEN_FIELDS completeness ───────────────────────────────────────


class TestHiddenFields:
    def test_uno_high_player_hands_included(self) -> None:
        """UNO 2-10 人：hand_p4…hand_p9 均纳入黑名单（原只列到 p3）。"""
        for key in ("hand_p4", "hand_p5", "hand_p6", "hand_p7", "hand_p8", "hand_p9"):
            assert key in HIDDEN_FIELDS, key

    def test_core_hidden_keys_still_present(self) -> None:
        """既有核心键未丢失。"""
        assert {
            "sb_hole",
            "bb_hole",
            "my_hole",
            "ai_hole",
            "my_hand",
            "ai_hand",
            "roles",
            "seerResult",
        } <= HIDDEN_FIELDS
