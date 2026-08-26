"""poker family runtime tests (Wave B1).

Covers the B1 deliverables end-to-end:

- ``detect`` claims ``rules/texas_holdem.json`` (hole/community arrays +
  raise/call/fold actions) and rejects grid / mahjong rules;
- ``build_spec`` generalizes the ``_poker_*`` closures: seats, hole
  arrays, env stack/committed/folded keys and constants are all read
  from the rules dict — a deep-copied variant with a smaller
  ``stack_size``, a trimmed ``raise_grid`` and a declared
  ``street_names`` table still yields a working spec;
- registered spec → ``PlayManager.start`` → human move → AI reply →
  terminal: both a showdown path (AI hole revealed) and a fold path
  (AI hole stays hidden — the hidden-information red line);
- the snapshot key set matches the ``PokerSnapshot`` contract in
  ``platform-frontend/src/types.ts`` exactly;
- ``create_solver("custom_poker", "hybrid", ..., allow_unknown=True)``
  builds a real Hybrid solver and its ``select_action`` works on a
  dealt state.
"""

from __future__ import annotations

import copy

import pytest

from layer2_engine.core.engine import GameEngine
from layer4_interface.frontend.engine_helpers import load_rules
from layer4_interface.frontend.platform.custom_games import (
    CustomGameRegistry,
    CustomGameStore,
)
from layer4_interface.frontend.platform.families import FAMILY_IDS, detect_family
from layer4_interface.frontend.platform.families.poker import build_spec, detect
from layer4_interface.frontend.platform.session import PlayManager
from train_cli import create_solver

#: ``PokerSnapshot`` 契约键集（platform-frontend/src/types.ts，快照本体不含
#: 会话层附加的 ``chat`` / ``evaluation``）。
POKER_SNAPSHOT_KEYS = frozenset(
    {
        "game_id",
        "player_pid",
        "ai_pid",
        "difficulty",
        "over",
        "winner",
        "turn",
        "phase",
        "street",
        "street_name",
        "pot",
        "community",
        "my_hole",
        "ai_hole",
        "revealed",
        "my_stack",
        "ai_stack",
        "my_committed",
        "ai_committed",
        "my_folded",
        "ai_folded",
        "last_actor",
        "last_action",
        "last_ai_action",
        "call_to",
        "my_hand_name",
        "ai_hand_name",
        "payoff",
        "legal",
        "raise_amounts",
    }
)


class _CallingSolver:
    """Scripted AI — always call when possible (drives a showdown)."""

    def __init__(self, engine: GameEngine) -> None:
        self.engine = engine
        self.name = "hybrid"

    def select_action(self, state: dict):
        legal = self.engine.get_legal_actions(state)
        for action in legal:
            if action.params.get("choice") == "call":
                return action
        return legal[0] if legal else None

    def solve(self, state: dict, **kwargs):  # pragma: no cover — protocol surface
        return self.select_action(state)

    def train(self, episodes: int, **kwargs):  # pragma: no cover — protocol surface
        return None


class _ScriptedProvider:
    """``SolverProvider`` stub returning the scripted AI (fast E2E tests)."""

    def create_solver(self, game_id: str, name: str, engine, seed: int, budget: int, **kwargs):
        return _CallingSolver(engine)


def _register(store: CustomGameStore, game_id: str, rules: dict) -> CustomGameRegistry:
    """Persist a minimal registry entry and return the registry."""
    store.save({"game_id": game_id, "family": "poker", "rules": rules})
    return CustomGameRegistry(store)


# ── Detection ─────────────────────────────────────────────────────────


class TestPokerDetection:
    def test_poker_family_auto_discovered(self):
        assert "poker" in FAMILY_IDS
        assert FAMILY_IDS == tuple(sorted(FAMILY_IDS))

    def test_texas_holdem_detected(self):
        rules = load_rules("texas_holdem")
        assert detect(rules) is True
        family = detect_family(rules)
        assert family is not None
        assert family.FAMILY_ID == "poker"

    def test_grid_rules_not_poker(self):
        rules = load_rules("stochastic_gomoku")
        assert detect(rules) is False
        assert detect_family(rules).FAMILY_ID == "grid"

    def test_mahjong_rules_not_poker(self):
        rules = load_rules("mahjong")
        assert detect(rules) is False


# ── Spec shape & constants generalization ─────────────────────────────


class TestPokerFamilySpec:
    def test_build_spec_shape(self):
        spec = build_spec("custom_poker", load_rules("texas_holdem"))
        assert spec.kind == "poker"
        assert spec.board_size is None
        assert spec.seat_options == ("p_sb", "p_bb")
        assert spec.seat_label == "座位"
        assert spec.player_counts == (2,)
        assert spec.difficulty_budgets == {"easy": 150, "normal": 500, "hard": 1200}
        assert spec.display_name == "texas_holdem"
        assert "Hold'em" in spec.description

    def test_modified_constants_variant(self, tmp_path):
        """stack_size / raise_grid / street_names 全部从 rules 常量读取。"""
        rules = copy.deepcopy(load_rules("texas_holdem"))
        rules["constants"]["stack_size"] = 50
        rules["constants"]["raise_grid"] = [0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50]
        rules["constants"]["street_names"] = ["翻前", "翻牌", "转牌", "河牌"]
        assert build_spec("custom_poker", rules).kind == "poker"  # 直接构造路径可用
        registry = _register(CustomGameStore(tmp_path / "custom_games"), "custom_poker", rules)
        manager = PlayManager(provider=_ScriptedProvider(), custom=registry, seed=42)
        session = manager.start("custom_poker", "p_sb", "easy")
        snap = session.snapshot()
        assert snap["street_name"] == "翻前"
        assert snap["raise_amounts"], "翻前应有合法加注额"
        assert all(a <= 50 for a in snap["raise_amounts"])
        assert set(snap["raise_amounts"]) <= set(rules["constants"]["raise_grid"])
        # 全下 50 → 脚本 AI 跟注 → 双方 all-in → showdown
        post = manager.move(session.game_id, {"choice": "raise", "amount": 50})
        assert post["over"] is True
        assert post["revealed"] is True
        assert len(post["ai_hole"]) == 2
        assert post["payoff"] is not None


# ── Session E2E ───────────────────────────────────────────────────────


@pytest.fixture
def showdown_manager(tmp_path) -> PlayManager:
    """Registry-registered texas rules + scripted calling AI."""
    rules = load_rules("texas_holdem")
    registry = _register(CustomGameStore(tmp_path / "custom_games"), "custom_poker", rules)
    return PlayManager(provider=_ScriptedProvider(), custom=registry, seed=42)


class TestPokerSession:
    def test_start_then_all_in_showdown(self, showdown_manager):
        manager = showdown_manager
        session = manager.start("custom_poker", "p_sb", "easy")
        assert session.over is False
        assert session.custom is True
        assert session.family == "poker"
        pre = session.snapshot()
        assert pre["revealed"] is False
        assert pre["ai_hole"] == []  # 未揭晓前 AI 底牌隐藏
        assert len(pre["my_hole"]) == 2
        assert pre["turn"] == "p_sb"
        assert pre["street_name"] == "street 0"
        assert pre["pot"] == 3  # 小盲 1 + 大盲 2
        post = manager.move(session.game_id, {"choice": "raise", "amount": 100})
        assert post["over"] is True
        assert post["revealed"] is True
        assert len(post["ai_hole"]) == 2  # showdown 揭晓后给出 AI 底牌
        assert post["winner"] in ("p_sb", "p_bb", None)
        assert post["payoff"] is not None
        assert post["my_hand_name"] is not None
        assert post["ai_hand_name"] is not None

    def test_fold_end_keeps_ai_hole_hidden(self, tmp_path):
        rules = load_rules("texas_holdem")
        registry = _register(CustomGameStore(tmp_path / "custom_games"), "custom_poker", rules)
        manager = PlayManager(provider=_ScriptedProvider(), custom=registry, seed=42)
        session = manager.start("custom_poker", "p_bb", "easy")
        # AI（小盲）翻前跟注 2 → 轮到大盲人类
        assert session.current_player == "p_bb"
        post = manager.move(session.game_id, {"choice": "fold"})
        assert post["over"] is True
        assert post["winner"] == "p_sb"
        assert post["revealed"] is False
        assert post["ai_hole"] == []  # 弃牌终局不翻牌，保持隐藏

    def test_snapshot_matches_poker_contract_keys(self, showdown_manager):
        session = showdown_manager.start("custom_poker", "p_sb", "easy")
        built = session.spec.build_snapshot(session)
        assert set(built) == POKER_SNAPSHOT_KEYS
        public = session.snapshot()
        assert POKER_SNAPSHOT_KEYS <= set(public)  # chat / evaluation 附加于外


# ── Hybrid solver assembly (allow_unknown) ────────────────────────────


class TestHybridSolverAssembly:
    def test_allow_unknown_hybrid_creates_and_selects(self):
        engine = GameEngine(load_rules("texas_holdem"), seed=42, allow_codegen=False)
        solver = create_solver(
            "custom_poker",
            "hybrid",
            engine,
            42,
            150,
            allow_unknown=True,
            imperfect_information=True,
            opponent_model="uniform",
        )
        assert "Hybrid" in solver.name
        state = engine.create_initial_state()
        while engine.get_node_type(state) == "chance":
            _, state = engine.sample_chance(state)
        action = solver.select_action(state)
        assert action is not None
        assert action.params.get("choice") in ("fold", "call", "raise")
