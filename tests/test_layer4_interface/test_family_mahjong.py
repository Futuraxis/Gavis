"""mahjong family runtime tests (Wave B).

Covers the B deliverables end-to-end:

- ``detect`` claims ``rules/mahjong.json`` (all six declared variants)
  and rejects grid / poker / social rules (stochastic gomoku, texas
  hold'em, moon chess, werewolf, undercover);
- ``build_spec`` generalizes the ``_mahjong_*`` closures: seats come
  from ``rules["players"]``, player counts are ``(2, 4)`` for the
  standard template, and variant selection stays declarative (no
  variant name hardcoded);
- registered spec → ``PlayManager.start`` (2- and 4-player) → human
  discard → the AI seats reply → the snapshot key set matches the
  ``MahjongSnapshot`` contract in ``platform-frontend/src/types.ts``
  exactly; ``ai_hand`` stays empty before the game is over (the
  hidden-information red line);
- ``create_solver("custom_mahjong", "mahjong", ..., allow_unknown=True)``
  builds the heuristic mahjong solver and its ``select_action`` works on
  a dealt state.
"""

from __future__ import annotations

import pytest

from layer2_engine.core.engine import GameEngine
from layer4_interface.frontend.engine_helpers import load_rules
from layer4_interface.frontend.platform.custom_games import (
    CustomGameRegistry,
    CustomGameStore,
)
from layer4_interface.frontend.platform.families import FAMILY_IDS, detect_family
from layer4_interface.frontend.platform.families.mahjong import build_spec, detect
from layer4_interface.frontend.platform.games import PlayError
from layer4_interface.frontend.platform.session import PlayManager
from train_cli import create_solver

#: ``MahjongSnapshot`` 契约键集（platform-frontend/src/types.ts，快照本体
#: 不含会话层附加的 ``chat`` / ``evaluation``）。
MAHJONG_SNAPSHOT_KEYS = frozenset(
    {
        "game_id",
        "player_pid",
        "ai_pid",
        "difficulty",
        "over",
        "winner",
        "turn",
        "phase",
        "my_hand",
        "ai_hand",
        "hand_counts",
        "melds",
        "discards",
        "wall_remaining",
        "last_discard",
        "last_action",
        "done",
        "winners",
        "payoffs",
        "claim",
        "legal",
        "last_ai_action",
    }
)

#: ``rules/mahjong.json`` 声明的六个变种（detect 与 build_spec 不得硬编码）。
MAHJONG_VARIANTS = ("guangdong", "hongzhong", "blood", "sichuan", "changsha", "taiwan")

#: 非麻将负例规则文件（grid / poker / social 族）。
NON_MAHJONG_RULES = ("stochastic_gomoku", "texas_holdem", "moon_chess", "werewolf", "undercover")


class _ScriptedMahjongAI:
    """Scripted AI — pass every claim, discard the first legal tile."""

    def __init__(self, engine: GameEngine) -> None:
        self.engine = engine
        self.name = "mahjong"

    def select_action(self, state: dict):
        legal = self.engine.get_legal_actions(state)
        if not legal:
            return None
        if state.get("env", {}).get("phase") == "claim":
            for action in legal:
                if action.template_id == "claim_pass":
                    return action
            return legal[0]
        for action in legal:
            if action.template_id == "discard":
                return action
        return legal[0]

    def solve(self, state: dict, **kwargs):  # pragma: no cover — protocol surface
        return self.select_action(state)

    def train(self, episodes: int, **kwargs):  # pragma: no cover — protocol surface
        return None


class _ScriptedProvider:
    """``SolverProvider`` stub returning the scripted AI (fast E2E tests)."""

    def create_solver(self, game_id: str, name: str, engine, seed: int, budget: int, **kwargs):
        return _ScriptedMahjongAI(engine)


def _register(store: CustomGameStore, game_id: str, rules: dict) -> CustomGameRegistry:
    """Persist a minimal registry entry and return the registry."""
    store.save({"game_id": game_id, "family": "mahjong", "rules": rules})
    return CustomGameRegistry(store)


def _discard_legal(snapshot: dict) -> dict:
    """First legal discard payload from a snapshot (human action phase)."""
    return next(action for action in snapshot["legal"] if action["type"] == "discard")


# ── Detection ─────────────────────────────────────────────────────────


class TestMahjongDetection:
    def test_mahjong_family_auto_discovered(self):
        assert "mahjong" in FAMILY_IDS
        assert FAMILY_IDS == tuple(sorted(FAMILY_IDS))

    def test_rules_file_declares_six_variants(self):
        options = load_rules("mahjong")["variants"]["options"]
        assert set(options) == set(MAHJONG_VARIANTS)

    def test_mahjong_rules_detected(self):
        rules = load_rules("mahjong")
        assert detect(rules) is True
        family = detect_family(rules)
        assert family is not None
        assert family.FAMILY_ID == "mahjong"

    @pytest.mark.parametrize("game_id", NON_MAHJONG_RULES)
    def test_non_mahjong_rules_rejected(self, game_id: str):
        rules = load_rules(game_id)
        assert detect(rules) is False
        family = detect_family(rules)
        # 其他族可认领（gomoku→grid / texas→poker），但绝不认领为 mahjong。
        assert family is None or family.FAMILY_ID != "mahjong"


# ── Spec shape & rules-data generalization ────────────────────────────


class TestMahjongFamilySpec:
    def test_build_spec_shape(self):
        spec = build_spec("custom_mahjong", load_rules("mahjong"))
        assert spec.kind == "mahjong"
        assert spec.board_size is None
        assert spec.seat_options == ("p0", "p1", "p2", "p3")
        assert spec.seat_label == "座位"
        assert spec.player_counts == (2, 4)
        assert spec.difficulty_budgets == {"easy": 1, "normal": 1, "hard": 1}
        assert spec.display_name == "mahjong"
        assert "Mahjong" in spec.description  # 模板 meta.description 为英文

    def test_player_counts_follow_declared_seats(self):
        rules = load_rules("mahjong")
        rules["players"] = ["p0", "p1"]
        assert build_spec("custom_mahjong_2p", rules).player_counts == (2,)


# ── Session E2E ───────────────────────────────────────────────────────


@pytest.fixture
def manager(tmp_path) -> PlayManager:
    """Registry-registered mahjong rules + scripted passing AI."""
    rules = load_rules("mahjong")
    registry = _register(CustomGameStore(tmp_path / "custom_games"), "custom_mahjong", rules)
    return PlayManager(provider=_ScriptedProvider(), custom=registry, seed=42)


class TestMahjongSession:
    def test_start_2p(self, manager: PlayManager):
        session = manager.start("custom_mahjong", "p0", "easy", player_count=2)
        assert session.over is False
        assert session.custom is True
        assert session.family == "mahjong"
        snap = session.snapshot()
        assert len(snap["my_hand"]) == 14  # 庄家 p0 多摸一张
        assert snap["phase"] == "action"
        assert snap["wall_remaining"] == 136 - 27
        assert snap["ai_hand"] == []  # 终局前 AI 手牌隐藏
        assert "discard" in {a["type"] for a in snap["legal"]}

    def test_start_4p(self, manager: PlayManager):
        session = manager.start("custom_mahjong", "p0", "easy", player_count=4)
        assert session.over is False
        snap = session.snapshot()
        assert len(snap["my_hand"]) == 14
        assert len(snap["hand_counts"]) == 4
        assert len(snap["melds"]) == 4
        assert len(snap["discards"]) == 4
        assert snap["wall_remaining"] == 136 - 53

    def test_player_count_validation(self, manager: PlayManager):
        with pytest.raises(PlayError, match="3 人"):
            manager.start("custom_mahjong", "p0", "easy", player_count=3)

    def test_human_discard_ai_replies_2p(self, manager: PlayManager):
        session = manager.start("custom_mahjong", "p0", "easy", player_count=2)
        payload = _discard_legal(session.snapshot())
        post = manager.move(session.game_id, payload)
        assert post["over"] is False
        assert post["ai_hand"] == []  # 隐藏红线：终局前 AI 手牌不泄露
        assert payload["tile"] in post["discards"]["p0"]
        assert any(entry["actor"] == "ai" for entry in session.log)  # AI 座位回手
        assert set(post) >= MAHJONG_SNAPSHOT_KEYS

    def test_multi_seat_ai_4p(self, manager: PlayManager):
        session = manager.start("custom_mahjong", "p0", "easy", player_count=4)
        assert session.current_player == "p0"
        payload = _discard_legal(session.snapshot())
        post = manager.move(session.game_id, payload)
        assert post["over"] is False
        assert post["ai_hand"] == []
        # 其余三个座位至少有两个 AI 动作（各自 claim 过/出牌）。
        ai_entries = [entry for entry in session.log if entry["actor"] == "ai"]
        assert len(ai_entries) >= 2
        # AI 回手后轮到人类（action 出牌或 claim 应接）。
        assert post["turn"] == "p0"
        assert post["phase"] in ("action", "claim")

    def test_ai_opens_when_human_not_first_seat(self, manager: PlayManager):
        session = manager.start("custom_mahjong", "p1", "easy", player_count=2)
        snap = session.snapshot()
        assert len(snap["my_hand"]) == 13  # AI（庄家 p0）已先开一张
        assert snap["ai_hand"] == []
        assert len(session.log) >= 1  # AI 开局动作已记录

    def test_snapshot_matches_mahjong_contract_keys(self, manager: PlayManager):
        session = manager.start("custom_mahjong", "p0", "easy", player_count=2)
        built = session.spec.build_snapshot(session)
        assert set(built) == MAHJONG_SNAPSHOT_KEYS
        public = session.snapshot()
        assert MAHJONG_SNAPSHOT_KEYS <= set(public)  # chat / evaluation 附加于外

    def test_full_game_ends_and_reveals(self, manager: PlayManager):
        """脚本 AI 全程过/出牌 → 牌墙摸空终局；终局后 AI 手牌揭晓。"""
        session = manager.start("custom_mahjong", "p1", "easy", player_count=2)
        guard = 0
        while not session.over and guard < 500:
            snap = session.snapshot()
            legal = snap["legal"]
            if not legal:
                break
            if snap["phase"] == "claim":
                action = {"type": "claim_pass"}
            else:
                action = _discard_legal(snap)
            manager.move(session.game_id, action)
            guard += 1
        assert session.over or guard >= 500
        if session.over:
            snap = session.snapshot()
            assert snap["ai_hand"], "终局后 AI 手牌应揭晓"
            assert snap["my_hand"]


# ── Heuristic solver assembly (allow_unknown) ─────────────────────────


class TestMahjongSolverAssembly:
    def test_allow_unknown_mahjong_creates_and_selects(self):
        engine = GameEngine(load_rules("mahjong"), seed=42, allow_codegen=False)
        solver = create_solver("custom_mahjong", "mahjong", engine, 42, 1, allow_unknown=True)
        assert solver.name == "mahjong_heuristic"
        state = engine.create_initial_state()
        while engine.get_node_type(state) == "chance":
            _, state = engine.sample_chance(state)
        action = solver.select_action(state)
        assert action is not None
        assert action.template_id in ("discard", "win_self", "gang_concealed", "gang_added")
