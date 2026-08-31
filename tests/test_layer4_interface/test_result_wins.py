"""Regression tests: faction-winner resolution (social games).

背景（实测 bug e7deb84b）：谁是卧底一局里玩家（p0=卧底）自爆猜对平民词、
``env.winner == "undercover"``——玩家**赢了**。但所有下游都用
``winner == player_pid`` 判胜负（``"undercover" != "p0"`` → 误判玩家落败），
于是对话引擎 outcome 写成「AI 获胜（玩家落败）」、聊天结果标签显示
「AI 获胜」、复盘摘要写「AI 获胜」、胜场统计与前端结算弹窗也全错。

修复：``layer4_interface.result.player_won`` 是唯一判据——pid 胜者 /
``winners`` 列表 / **阵营胜者**（社交游戏按终局公开身份表 ``final_roles``
匹配；身份表缺失时用引擎 per-viewer 终局效用符号兜底）。本文件用真实
e7deb84b 记录形状把这一判据钉在四个消费端上。
"""

from __future__ import annotations

from types import SimpleNamespace

from layer4_interface.agent.dialogue_engine import _endgame_outcome
from layer4_interface.frontend.platform.chat import _result_label
from layer4_interface.review.analyzer import analyze
from layer4_interface.result import player_won, role_of

#: e7deb84b 终局快照：p0（人类）= 卧底、词「狮子」；平民词「老虎」；
#: p2 = 白板；p0 自爆猜 p1 的「老虎」猜对 → winner=undercover。
UNDERCOVER_FINAL = {
    "family": "social",
    "game_id": "undercover",
    "player_pid": "p0",
    "winner": "undercover",
    "winners": [],
    "over": True,
    "final_roles": [
        {"pid": "p0", "role": "undercover", "word": "狮子"},
        {"pid": "p1", "role": "civilian", "word": "老虎"},
        {"pid": "p2", "role": "blank", "word": "白板"},
        {"pid": "p3", "role": "civilian", "word": "老虎"},
        {"pid": "p4", "role": "civilian", "word": "老虎"},
    ],
}

#: 卧底/白板被投出 → winner=civilian（p0 是卧底 → 玩家落败）。
UNDERCOVER_CIV_WIN = {**UNDERCOVER_FINAL, "winner": "civilian"}


class TestPlayerWon:
    def test_undercover_player_win_matches_faction(self) -> None:
        assert player_won("undercover", "p0", [], UNDERCOVER_FINAL) is True

    def test_undercover_player_lost_when_faction_mismatch(self) -> None:
        assert player_won("civilian", "p0", [], UNDERCOVER_CIV_WIN) is False

    def test_ai_civilian_lost_when_undercover_wins(self) -> None:
        assert player_won("undercover", "p1", [], UNDERCOVER_FINAL) is False

    def test_blank_wins(self) -> None:
        snap = {**UNDERCOVER_FINAL, "winner": "blank"}
        assert player_won(snap["winner"], "p2", [], snap) is True
        assert player_won(snap["winner"], "p0", [], snap) is False

    def test_pid_winner_unchanged(self) -> None:
        assert player_won("p0", "p0", [], {}) is True
        assert player_won("p1", "p0", [], {}) is False

    def test_multi_winner_list(self) -> None:
        assert player_won(None, "p0", ["p0", "p3"], {"winners": ["p0", "p3"]}) is True
        assert player_won(None, "p1", ["p0", "p3"], {"winners": ["p0", "p3"]}) is False

    def test_draw_no_winner(self) -> None:
        assert player_won(None, "p0", [], {}) is None

    def test_faction_without_roles_falls_back_to_utility(self) -> None:
        # 原始投影没有 final_roles（社交身份隐藏，终局才经快照揭晓）：
        # 用引擎 per-viewer 效用符号兜底（agent/evaluation 的 summary 同源）。
        assert player_won("undercover", "p0", [], {}, score=1.0) is True
        assert player_won("undercover", "p0", [], {}, score=-1.0) is False

    def test_faction_without_roles_no_utility_defaults_lost(self) -> None:
        # 无法判定且无效用信号 → 常规对局语义：胜者不是玩家即玩家落败。
        assert player_won("undercover", "p0", [], {}) is False


class TestRoleOf:
    def test_lookup_by_pid(self) -> None:
        assert role_of(UNDERCOVER_FINAL, "p0") == "undercover"
        assert role_of(UNDERCOVER_FINAL, "p2") == "blank"

    def test_missing_row_or_bad_snap(self) -> None:
        assert role_of(UNDERCOVER_FINAL, "p9") is None
        assert role_of(None, "p0") is None
        assert role_of({}, "p0") is None


class TestEndgameOutcome:
    """对话引擎终局 outcome（防 LLM 幻觉「谁赢了」）——必须把阵营胜者归对边。"""

    @staticmethod
    def _ctx(**kw: object) -> SimpleNamespace:
        base: dict[str, object] = {
            "observation": {"env": {"winner": "undercover", "winners": []}},
            "evaluation": {"score": 1.0, "summary": "本方获胜", "mechanical_text": "终局，本方效用 +1.0"},
            "human_pid": "p0",
            "ai_pid": "p1",
            "adversarial": False,
        }
        base.update(kw)
        return SimpleNamespace(**base)

    def test_undercover_player_win(self) -> None:
        out = _endgame_outcome(self._ctx())
        assert out["winner"] == "undercover"
        assert out["outcome"] == "AI 落败（玩家获胜）"

    def test_undercover_adversarial_ai_win(self) -> None:
        # 对手模式：观测视角 = AI（human_pid 被 OpponentContext 设为 ai_pid），
        # 效用符号 +1 → 说话身份（AI）获胜。
        out = _endgame_outcome(self._ctx(adversarial=True, human_pid="p1"))
        assert out["outcome"] == "你（AI）获胜"

    def test_pid_winner_keeps_old_semantics(self) -> None:
        ctx = self._ctx(observation={"env": {"winner": "p1", "winners": []}}, evaluation={"score": -1.0})
        assert _endgame_outcome(ctx)["outcome"] == "AI 获胜（玩家落败）"

    def test_draw(self) -> None:
        ctx = self._ctx(observation={"env": {"winner": None, "winners": []}}, evaluation={"score": 0.0})
        assert _endgame_outcome(ctx)["outcome"] == "平局"


class TestResultLabel:
    """平台聊天结果标签（get_match_state / 本局已结束 行）。"""

    def test_undercover_player_win(self) -> None:
        assert _result_label(UNDERCOVER_FINAL) == "你获胜"

    def test_undercover_ai_win_when_viewer_is_civilian(self) -> None:
        snap = {**UNDERCOVER_FINAL, "player_pid": "p1"}
        assert _result_label(snap) == "AI 获胜"

    def test_civilian_win_player_lost(self) -> None:
        assert _result_label(UNDERCOVER_CIV_WIN) == "AI 获胜"

    def test_pid_winner_keeps_old_semantics(self) -> None:
        assert _result_label({"winner": "p0", "player_pid": "p0"}) == "你获胜"
        assert _result_label({"winner": "p1", "player_pid": "p0"}) == "AI 获胜"

    def test_draw(self) -> None:
        assert _result_label({"winner": None, "player_pid": "p0"}) == "平局"


def _undercover_record(winner: str, last_actor: str = "human") -> dict:
    """Minimal undercover match record (terminal snapshot carries final_roles)."""
    moves = []
    for i in range(14):
        snap = {"player_pid": "p0", "winner": None, "over": False}
        if i == 13:
            snap = dict(UNDERCOVER_FINAL) if winner == "undercover" else dict(UNDERCOVER_CIV_WIN)
        moves.append(
            {
                "step": i,
                "actor": "human" if i % 2 == 0 else "ai",
                "action": f"step_{i}",
                "snapshot": snap,
            }
        )
    moves[-1]["actor"] = last_actor
    return {
        "match_id": "und_regression",
        "game_id": "undercover",
        "player_pid": "p0",
        "ai_pid": "p1",
        "difficulty": "normal",
        "seed": 42,
        "winner": winner,
        "over": True,
        "moves": moves,
        "meta": {
            "match_id": "und_regression",
            "game_id": "undercover",
            "player_pid": "p0",
            "ai_pid": "p1",
            "difficulty": "normal",
            "winner": winner,
            "over": True,
            "moves": len(moves),
        },
    }


class TestAnalyzeUndercover:
    def test_player_win_summary_and_no_blunder(self) -> None:
        report = analyze(_undercover_record("undercover"))
        assert "玩家获胜" in report.summary
        assert "共 14 手" in report.summary
        assert not any(node.kind == "blunder" for node in report.key_nodes)
        assert not any(node.kind == "winning_move" for node in report.key_nodes)

    def test_ai_win_summary_when_faction_mismatch(self) -> None:
        report = analyze(_undercover_record("civilian"))
        assert "AI 获胜" in report.summary
        assert "玩家获胜" not in report.summary

    def test_winning_move_for_ai_faction(self) -> None:
        report = analyze(_undercover_record("civilian", last_actor="ai"))
        winning = [node for node in report.key_nodes if node.kind == "winning_move"]
        assert winning
        assert winning[0].step == 13

    def test_local_terminal_score_uses_faction(self) -> None:
        from layer4_interface.review.analyzer import _local_score

        assert _local_score(UNDERCOVER_FINAL, "p0", "p1") == 1.0
        assert _local_score(UNDERCOVER_CIV_WIN, "p0", "p1") == -1.0


class TestWerewolfFactionSide:
    """狼人杀：winner=good/wolf 阵营 vs 具体身份（好人侧 = 非狼身份全胜）。"""

    WOLF_FINAL = {
        "winner": "good",
        "final_roles": [
            {"pid": "p0", "role": "seer"},
            {"pid": "p1", "role": "wolf"},
            {"pid": "p2", "role": "villager"},
        ],
    }

    def test_good_side_nony_role_wins(self) -> None:
        assert player_won(TestWerewolfFactionSide.WOLF_FINAL["winner"], "p0", [], TestWerewolfFactionSide.WOLF_FINAL) is True
        assert player_won(TestWerewolfFactionSide.WOLF_FINAL["winner"], "p2", [], TestWerewolfFactionSide.WOLF_FINAL) is True

    def test_wolf_loses_when_good_wins(self) -> None:
        assert player_won("good", "p1", [], TestWerewolfFactionSide.WOLF_FINAL) is False

    def test_wolf_wins(self) -> None:
        snap = {**TestWerewolfFactionSide.WOLF_FINAL, "winner": "wolf"}
        assert player_won("wolf", "p1", [], snap) is True
        assert player_won("wolf", "p0", [], snap) is False

    def test_faction_matches_direct(self) -> None:
        from layer4_interface.result import faction_matches

        assert faction_matches("seer", "good") is True
        assert faction_matches("wolf", "good") is False
        assert faction_matches("wolf", "wolf") is True
        assert faction_matches("villager", "wolf") is False
        assert faction_matches("undercover", "undercover") is True
        assert faction_matches("undercover", "civilian") is False

    def test_winning_move_for_good_side_when_player_wolf(self) -> None:
        rec = _undercover_record("good", last_actor="ai")
        # 人类是狼、好人侧获胜（终局身份表公开）：胜方最后一手要落在村民/预言家
        # 身上（p1=AI 预言家 step 13），而不是把胜者当 pid 找不到归属。
        rec["moves"][-1]["snapshot"] = {
            "player_pid": "p0",
            "winner": "good",
            "winners": [],
            "over": True,
            "final_roles": [
                {"pid": "p0", "role": "wolf"},
                {"pid": "p1", "role": "seer"},
                {"pid": "p2", "role": "villager"},
            ],
        }
        rec["winner"] = "good"
        rec["meta"]["winner"] = "good"
        report = analyze(rec)
        winning = [node for node in report.key_nodes if node.kind == "winning_move"]
        assert winning
        assert winning[0].step == 13
        assert "AI 获胜" in report.summary
        assert "玩家获胜" not in report.summary

    def test_local_terminal_score_for_good_side(self) -> None:
        from layer4_interface.review.analyzer import _local_score

        assert _local_score(TestWerewolfFactionSide.WOLF_FINAL, "p0", "p1") == 1.0
        assert _local_score(TestWerewolfFactionSide.WOLF_FINAL, "p1", "p0") == -1.0


class TestSkillsSummarizeResult:
    """Skills.summarize_result 赛后胜负——阵营胜者 + 引擎效用符号兜底."""

    def test_undercover_player_win(self) -> None:
        from types import SimpleNamespace as NS

        from layer4_interface.agent.skills import Skills

        # 真实对手路径：ctx.observation 是原始投影（无 final_roles），
        # 阵营匹配失败 → 用评估效用符号（+1 = 本方获胜，与 evaluation.summary 同源）。
        ctx = NS(observation={"env": {"winner": "undercover"}}, evaluation={"score": 1.0})
        res = Skills.summarize_result(ctx, None, "undercover", "p0")
        assert res["won"] is True
        assert res["summary"] == "本方获胜"

    def test_undercover_player_lost(self) -> None:
        from types import SimpleNamespace as NS

        from layer4_interface.agent.skills import Skills

        ctx = NS(observation={"env": {"winner": "undercover"}}, evaluation={"score": -1.0})
        res = Skills.summarize_result(ctx, None, "undercover", "p0")
        assert res["won"] is False
        assert res["summary"] == "本方落败"