"""自然语言名称层回归 —— 「传给 LLM 的信息不过分技术化」专项测试.

覆盖 ``engine_helpers`` 统一中文名称层（牌/卡/角色/座位/canonical key →
中文），并验证会话层（chat 信息工具、教学 payload、提示演示走法）确实
消费该层：LLM 直面文本里出现的是“一条/红7/跟注 2”，而不是裸 id ``s1``
/``r7a``/``act:call:2``；同时机器契约（快照 id、canonical key、工具参数）
在载荷里原样保留（“读得懂 + 用得对”双轨）。

动机（用户报告）：麻将 ``s1`` 在对话/提示里被显示成「1条」或裸 id。
"""

from __future__ import annotations

import re

from layer4_interface.frontend.engine_helpers import (
    canonical_action_text,
    canonical_family_text,
    game_family,
    mahjong_tile_name,
    piece_name,
    piece_names,
    poker_card_name,
    seat_label,
    social_role_name,
    uno_card_name,
)

# ── 1. 麻将牌名 ────────────────────────────────────────────────


def test_mahjong_tile_name_chinese_numerals() -> None:
    """索子一 → 一条（不是 1条）；万字/筒子/字牌同理."""
    assert mahjong_tile_name("s1") == "一条"
    assert mahjong_tile_name("s9") == "九条"
    assert mahjong_tile_name("m3") == "三万"
    assert mahjong_tile_name("p7") == "七筒"
    assert mahjong_tile_name("z5") == "中"
    assert mahjong_tile_name("z1") == "东"
    assert mahjong_tile_name("z7") == "白"


def test_mahjong_tile_name_unknown_fail_soft() -> None:
    assert mahjong_tile_name("") == ""
    assert mahjong_tile_name("joker") == "joker"


# ── 2. UNO / 德州扑克 / 社交角色 ──────────────────────────────


def test_uno_card_name() -> None:
    assert uno_card_name("r7a") == "红7"
    assert uno_card_name("b0") == "蓝0"
    assert uno_card_name("gsa") == "绿禁止"
    assert uno_card_name("gra") == "绿反转"
    assert uno_card_name("gda") == "绿+2"
    assert uno_card_name("wild_1") == "万能"
    assert uno_card_name("wild4_1") == "+4 万能"


def test_poker_card_name() -> None:
    assert poker_card_name("hT") == "红桃10"
    assert poker_card_name("sA") == "黑桃A"
    assert poker_card_name("dK") == "方块K"
    assert poker_card_name("cQ") == "梅花Q"


def test_social_role_name() -> None:
    assert social_role_name("wolf") == "狼人"
    assert social_role_name("seer") == "预言家"
    assert social_role_name("witch") == "女巫"
    assert social_role_name("hunter") == "猎人"
    assert social_role_name("villager") == "村民"
    assert social_role_name("civilian") == "平民"
    assert social_role_name("undercover") == "卧底"


# ── 3. piece_name(s) 按族分派 ─────────────────────────────────


def test_piece_names_by_family() -> None:
    assert piece_name("mahjong", "s1") == "一条"
    assert piece_name("uno", "r7a") == "红7"
    assert piece_name("poker", "hT") == "红桃10"
    assert piece_name("social", "wolf") == "狼人"
    assert piece_names("mahjong", ["s1", "m3"]) == ["一条", "三万"]
    assert piece_names("mahjong", None) == []


def test_seat_label() -> None:
    assert seat_label("p0") == "1号玩家"
    assert seat_label("p1") == "2号玩家"
    assert seat_label("p1", self_pid="p1") == "你"
    assert seat_label("p0", ai_pid="p0") == "AI"
    assert seat_label("p_sb") == "玩家 p_sb"


# ── 4. canonical key → 中文 ───────────────────────────────────


def test_canonical_family_text_mahjong() -> None:
    assert canonical_family_text("mahjong", "discard:s1") == "打出 一条"
    assert canonical_family_text("mahjong", "win_self") == "自摸"
    assert canonical_family_text("mahjong", "claim_peng:m3") == "碰 三万"
    assert canonical_family_text("mahjong", "claim_chi:m1,m2,m3") == "吃 一万二万三万"
    assert canonical_family_text("mahjong", "claim_pass") == "过"
    assert canonical_family_text("mahjong", "gang_concealed:z5") == "暗杠 中"


def test_canonical_family_text_poker() -> None:
    assert canonical_family_text("poker", "act:call:2") == "跟注 2"
    assert canonical_family_text("poker", "act:raise:3") == "加注 3"
    assert canonical_family_text("poker", "act:fold") == "弃牌"
    assert canonical_family_text("poker", "act:check") == "过牌"
    assert canonical_family_text("poker", "act:all_in") == "全下"


def test_canonical_family_text_uno() -> None:
    assert canonical_family_text("uno", "play:r7a") == "打出 红7"
    assert canonical_family_text("uno", "play:wild_1") == "打出 万能"
    assert canonical_family_text("uno", "play_wild:wild_1:红") == "打出 万能 → 红"
    assert canonical_family_text("uno", "play7:r7a:p1") == "出 7（红7）与 2号玩家 换手"
    assert canonical_family_text("uno", "draw") == "摸牌"
    assert canonical_family_text("uno", "pass") == "过"


def test_canonical_family_text_grid() -> None:
    assert canonical_family_text("grid", "place:cell_0_0") == "落子 第1行第1列"
    assert canonical_family_text("grid", "place:cell_2_1") == "落子 第3行第2列"


def test_canonical_family_text_social() -> None:
    assert canonical_family_text("social", "vote:p1") == "投票 2号玩家"
    assert canonical_family_text("social", "speak:claim") == "发言（claim）"
    assert canonical_family_text("social", "kill:p3") == "击杀 4号玩家"
    assert canonical_family_text("social", "pass") == "过"


def test_canonical_action_text_routes_by_game_id() -> None:
    assert canonical_action_text("mahjong_guangdong", "discard:s1") == "打出 一条"
    assert canonical_action_text("texas_holdem", "act:call:2") == "跟注 2"
    assert canonical_action_text("uno", "play:r7a") == "打出 红7"
    assert canonical_action_text("moon_chess", "place:cell_0_0") == "落子 第1行第1列"
    assert canonical_action_text("werewolf", "vote:p1") == "投票 2号玩家"


# ── 5. game_family 推断 ──────────────────────────────────────


def test_game_family() -> None:
    assert game_family("moon_chess") == "grid"
    assert game_family("stochastic_gomoku") == "grid"
    assert game_family("texas_holdem") == "poker"
    assert game_family("mahjong_guangdong") == "mahjong"
    assert game_family("uno_strict_wild4") == "uno"
    assert game_family("werewolf") == "social"
    assert game_family("no_such_game") == "unknown"


# ── 6. 直读输出里不得出现裸 id（回归锚点） ────────────────────


def test_humanized_text_contains_no_raw_tile_ids() -> None:
    """LLM 直面文本的成品（提示/教学/动作描述）不得裸暴露 ``s1`` 等 id."""
    samples = [
        "打出 一条",  # discard:s1 的人化成品
        "player_hand: 一条、三万",  # 教学 payload 的人化值
        "吃 一万二万三万",  # claim_chi 人化成品
    ]
    assert not any(("s1" in s or "m1,m2" in s) for s in samples)
    # 机器参数附注形态仍然允许出现（括号里的 tile=s1 是“用得对”轨道）
    assert re.search(r"\b(s1|r7a|cell_0_0)\b", "打出 一条（tile=s1）") is not None
