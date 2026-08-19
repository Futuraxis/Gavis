"""Tests for Texas Hold'em (Layer 2, v5.1 — zero builtins).

The game logic (hand evaluation, betting rules, payoff) lives in
``rules/texas_holdem.json`` as expression aliases.  Every expected
value vector from the pre-refactor suite is ported verbatim — the
tests assert engine-level behavior (``env.winner`` / ``get_utility`` /
``get_observation``) and evaluate the aliases through the engine, so
the JSON stays the single source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from layer2_engine.games.texas_holdem.texas_env_adapter import TexasHoldemAdapter
from layer2_engine.interfaces.solver_adapter import SolverAdapter

RULES_PATH = Path(__file__).resolve().parent.parent.parent / "rules" / "texas_holdem.json"


@pytest.fixture
def adapter() -> TexasHoldemAdapter:
    return TexasHoldemAdapter(seed=42)


def _resolve_chance(adapter: TexasHoldemAdapter, state: dict) -> dict:
    while adapter.get_node_type(state) == "chance":
        _, state = adapter.sample_chance(state)
    return state


def _act(adapter: TexasHoldemAdapter, state: dict, choice: str, amount=None) -> dict:
    """Apply the unique (or amount-matching) legal action for ``choice``."""
    cands = [a for a in adapter.get_legal_actions(state) if a.params.get("choice") == choice]
    if amount is None:
        assert len(cands) == 1, f"choice {choice} not unique"
        return adapter.apply_action(state, cands[0])
    for a in cands:
        if a.params.get("amount") == amount:
            return adapter.apply_action(state, a)
    raise AssertionError(f"not legal: {choice} {amount}")


def _legal_keys(adapter: TexasHoldemAdapter, state: dict) -> set[str]:
    return {a.canonical_key for a in adapter.get_legal_actions(state)}


def _crafted(adapter: TexasHoldemAdapter, env: dict, arrays: dict) -> dict:
    """Build a terminal-ish state via load_state (schema defaults fill gaps)."""
    return adapter.load_state({"_arrays": arrays, "env": env})


def _call(adapter: TexasHoldemAdapter, name: str, state: dict, *args) -> object:
    """Evaluate one of the rules aliases (winner/payoff/...) on ``state``."""
    ctx = adapter._build_context(state)  # noqa: SLF001 — engine-layer test
    spec = {"call": [name, *({"const": a} for a in args)]}
    return adapter.expr.eval(spec, ctx)


def _eval_five(adapter: TexasHoldemAdapter, cards: list) -> list:
    """Best-5 value list [category, tiebreaks...] for a card list."""
    ctx = {"$constants": adapter._constants, "$env": {}}
    return adapter.expr.eval({"call": ["best5", {"const": list(cards)}]}, ctx)


# ── Hand evaluation (alias vectors, ported from the builtin suite) ─────


class TestHandEvaluator:
    def test_rank_alias(self, adapter):
        ctx = adapter._build_context(adapter.create_initial_state())
        for card, expected in [("s2", 2), ("sT", 10), ("hJ", 11), ("dQ", 12), ("cK", 13), ("hA", 14)]:
            assert adapter.expr.eval({"call": ["rank", {"const": card}]}, ctx) == expected

    def test_contains_expression(self, adapter):
        spec = {"contains": [{"const": ["sA", "h2"]}, {"const": "sA"}]}
        assert adapter.expr.eval(spec, {}) is True
        spec = {"contains": [{"const": ["sA", "h2"]}, {"const": "d3"}]}
        assert adapter.expr.eval(spec, {}) is False

    def test_categories(self, adapter):
        """Ported verbatim from poker_hand_value vectors: category + first tiebreak."""
        cases = [
            (["sA", "sK", "sQ", "sJ", "sT"], (8, 14)),  # royal
            (["s2", "s3", "s4", "s5", "s6"], (8, 6)),  # straight flush
            (["hA", "sA", "dA", "cA", "sK"], (7, 14)),  # quads
            (["c5", "s5", "h5", "dK", "cK"], (6, 5)),  # full house
            (["h9", "h7", "h5", "h3", "h2"], (5, 9)),  # flush
            (["s2", "s3", "d4", "d5", "c6"], (4, 6)),  # straight
            (["hA", "sA", "dA", "sK", "d2"], (3, 14)),  # trips
            (["sA", "sK", "hA", "hK", "d2"], (2, 14)),  # two pair
            (["sA", "sK", "hA", "d2", "c3"], (1, 14)),  # pair
            (["sA", "sK", "hQ", "dJ", "c9"], (0, 14)),  # high card
        ]
        for cards, (cat, s1) in cases:
            value = _eval_five(adapter, cards)
            assert value[0] == cat, f"{cards}: category"
            assert value[1] == s1, f"{cards}: first tiebreak"

    def test_best_five_of_seven(self, adapter):
        # Community + hole: picks the straight flush over the pair of aces
        cards = ["sA", "s2", "s3", "s4", "s5", "h2", "c9"]
        assert _eval_five(adapter, cards)[:2] == [8, 5]
        # Wheel (A-2-3-4-5) beats a pair
        cards = ["hA", "sA", "s2", "s3", "d4", "d5", "c9"]
        assert _eval_five(adapter, cards)[:2] == [4, 5]

    def test_two_pair_kicker_ordering(self, adapter):
        """Two pair with the same pairs is broken by the kicker."""
        a = _eval_five(adapter, ["sA", "sK", "hA", "hK", "d2", "c3", "s4"])
        b = _eval_five(adapter, ["sA", "sK", "hA", "hK", "d2", "c3", "s3"])
        assert a > b  # kicker 4 > 3

    def test_hand_name(self, adapter):
        assert adapter.hand_name(["sA", "sK", "sQ", "sJ", "sT"]) == "同花顺"
        assert adapter.hand_name(["c5", "s5", "h5", "dK", "cK"]) == "葫芦"
        assert adapter.hand_name(["sA", "sK", "hQ", "dJ", "c9"]) == "高牌"


# ── Engine basics ─────────────────────────────────────────────────────


class TestTexasHoldemBasics:
    def test_create_initial_state(self, adapter: TexasHoldemAdapter):
        state = adapter.create_initial_state()
        env = state["env"]
        assert env["phase"] == "deal_sb1"
        assert env["sb_stack"] == 99 and env["bb_stack"] == 98  # blinds posted
        assert env["sb_committed"] == 1 and env["bb_committed"] == 2
        assert state["_arrays"]["sb_hole"] == []
        assert state["_arrays"]["community"] == []

    def test_initial_node_is_chance(self, adapter: TexasHoldemAdapter):
        state = adapter.create_initial_state()
        assert adapter.get_node_type(state) == "chance"
        assert adapter.get_current_player(state) is None

    def test_dealing(self, adapter: TexasHoldemAdapter):
        state = _resolve_chance(adapter, adapter.create_initial_state())
        assert len(state["_arrays"]["sb_hole"]) == 2
        assert len(state["_arrays"]["bb_hole"]) == 2
        assert len(state["_arrays"]["drawn"]) == 4
        drawn = state["_arrays"]["drawn"]
        assert len(set(drawn)) == 4  # no duplicate cards
        assert adapter.get_node_type(state) == "player"
        assert adapter.get_current_player(state) == "p_sb"

    def test_is_solver_adapter(self, adapter: TexasHoldemAdapter):
        assert isinstance(adapter, SolverAdapter)

    def test_protocol_methods(self, adapter: TexasHoldemAdapter):
        for m in (
            "create_initial_state",
            "get_node_type",
            "get_current_player",
            "get_legal_actions",
            "apply_action",
            "get_chance_outcomes",
            "apply_chance",
            "is_terminal",
            "get_utility",
            "get_observation",
            "get_info_set_key",
            "load_state",
            "project_observation",
        ):
            assert hasattr(adapter, m), f"Missing method: {m}"


# ── Betting rounds ────────────────────────────────────────────────────


class TestBetting:
    def test_preflop_legal_actions(self, adapter: TexasHoldemAdapter):
        state = _resolve_chance(adapter, adapter.create_initial_state())
        keys = _legal_keys(adapter, state)
        assert "act:fold:0" in keys
        assert "act:call:2" in keys  # call the blind
        assert "act:raise:4" in keys and "act:raise:2" not in keys  # min-raise = 2×BB
        assert "act:raise:100" in keys  # all-in always available

    def test_min_raise_progression(self, adapter: TexasHoldemAdapter):
        state = _resolve_chance(adapter, adapter.create_initial_state())
        state = _act(adapter, state, "call")  # SB calls → BB acts
        state = _act(adapter, state, "raise", 6)  # BB raises to 6
        keys = _legal_keys(adapter, state)
        assert "act:call:6" in keys
        assert "act:raise:10" in keys  # 6 + max(4, 2)
        assert "act:raise:8" not in keys

    def test_check_check_advances_street(self, adapter: TexasHoldemAdapter):
        state = _resolve_chance(adapter, adapter.create_initial_state())
        state = _act(adapter, state, "call")  # SB call
        state = _act(adapter, state, "call")  # BB check
        assert state["env"]["phase"] == "deal_flop1"
        state = _resolve_chance(adapter, state)
        assert state["env"]["phase"] == "betting"
        assert len(state["_arrays"]["community"]) == 3
        assert adapter.get_current_player(state) == "p_bb"  # BB leads postflop

    def test_fold_ends_hand(self, adapter: TexasHoldemAdapter):
        state = _resolve_chance(adapter, adapter.create_initial_state())
        state = _act(adapter, state, "fold")
        assert adapter.is_terminal(state)
        assert state["env"]["winner"] == "p_bb"
        assert adapter.get_utility(state, "p_sb") == -1.0
        assert adapter.get_utility(state, "p_bb") == 1.0

    def test_allin_fold_or_call_only(self, adapter: TexasHoldemAdapter):
        """Against an all-in the opponent can only fold or call (no raise)."""
        state = _resolve_chance(adapter, adapter.create_initial_state())
        state = _act(adapter, state, "raise", 100)  # SB shoves
        assert state["env"]["turn"] == "p_bb"
        assert _legal_keys(adapter, state) == {"act:call:100", "act:fold:0"}
        state = _act(adapter, state, "fold")
        assert state["env"]["winner"] == "p_sb"
        assert adapter.get_utility(state, "p_bb") == -2.0  # BB loses the blind

    def test_utility_is_zero_sum(self, adapter: TexasHoldemAdapter):
        state = _resolve_chance(adapter, adapter.create_initial_state())
        state = _act(adapter, state, "raise", 30)
        state = _act(adapter, state, "raise", 60)
        state = _act(adapter, state, "call")
        while not adapter.is_terminal(state):
            state = _resolve_chance(adapter, state)
            if adapter.get_node_type(state) == "player":
                state = _act(adapter, state, "call")  # check/check or call
        assert adapter.is_terminal(state)
        assert len(state["_arrays"]["community"]) == 5
        u_sb = adapter.get_utility(state, "p_sb")
        u_bb = adapter.get_utility(state, "p_bb")
        assert u_sb + u_bb == 0.0
        assert abs(u_sb) <= 60


# ── Payoffs (fold / refund / split) — vectors ported verbatim ─────────


class TestPayoffs:
    def _state(self, adapter: TexasHoldemAdapter, env: dict, arrays: dict) -> dict:
        env = {"phase": "game_over", "street": 3, **env}
        return _crafted(adapter, env, arrays)

    def test_fold_payoff(self, adapter: TexasHoldemAdapter):
        state = self._state(
            adapter,
            {
                "sb_committed": 10,
                "bb_committed": 2,
                "sb_stack": 90,
                "bb_stack": 98,
                "sb_folded": False,
                "bb_folded": True,
            },
            {"sb_hole": ["sA", "sK"], "bb_hole": ["hA", "hK"], "community": []},
        )
        assert _call(adapter, "winner", state) == "p_sb"
        assert adapter.get_utility(state, "p_sb") == 2.0  # pot 12 - committed 10
        assert adapter.get_utility(state, "p_bb") == -2.0

    def test_showdown_winner(self, adapter: TexasHoldemAdapter):
        # SB royal flush vs BB straight flush — SB wins the full pot
        state = self._state(
            adapter,
            {
                "sb_committed": 50,
                "bb_committed": 50,
                "sb_stack": 50,
                "bb_stack": 50,
                "sb_folded": False,
                "bb_folded": False,
            },
            {
                "sb_hole": ["sA", "sK"],
                "bb_hole": ["hA", "hK"],
                "community": ["sQ", "sJ", "sT", "d2", "c3"],
            },
        )
        assert _call(adapter, "winner", state) == "p_sb"
        assert adapter.get_utility(state, "p_sb") == 50.0
        assert adapter.get_utility(state, "p_bb") == -50.0

    def test_allin_refund(self, adapter: TexasHoldemAdapter):
        # SB all-in 40, BB over-committed 100 → 60 refunded regardless of winner
        state = self._state(
            adapter,
            {
                "sb_committed": 40,
                "bb_committed": 100,
                "sb_stack": 60,
                "bb_stack": 0,
                "sb_folded": False,
                "bb_folded": False,
            },
            {
                "sb_hole": ["sA", "sK"],
                "bb_hole": ["hA", "hK"],
                "community": ["sQ", "sJ", "sT", "d2", "c3"],
            },
        )
        assert _call(adapter, "winner", state) == "p_sb"
        assert adapter.get_utility(state, "p_sb") == 40.0  # main pot 80 - committed 40
        assert adapter.get_utility(state, "p_bb") == -40.0  # -100 + refund 60

    def test_split_pot(self, adapter: TexasHoldemAdapter):
        # Both make the same two pair (A-A-K-K, kicker 4) → split
        state = self._state(
            adapter,
            {
                "sb_committed": 30,
                "bb_committed": 30,
                "sb_stack": 70,
                "bb_stack": 70,
                "sb_folded": False,
                "bb_folded": False,
            },
            {
                "sb_hole": ["sA", "sK"],
                "bb_hole": ["hA", "hK"],
                "community": ["dA", "dK", "c2", "c3", "c4"],
            },
        )
        assert _call(adapter, "winner", state) is None
        assert adapter.get_utility(state, "p_sb") == 0.0
        assert adapter.get_utility(state, "p_bb") == 0.0

    def test_showdown_fold_winner_payoff(self, adapter: TexasHoldemAdapter):
        """Wheel (A-2-3-4-5) must beat a pair of nines at showdown."""
        state = self._state(
            adapter,
            {
                "sb_committed": 20,
                "bb_committed": 20,
                "sb_stack": 80,
                "bb_stack": 80,
                "sb_folded": False,
                "bb_folded": False,
            },
            {
                "sb_hole": ["hA", "s2"],
                "bb_hole": ["s9", "d9"],
                "community": ["s3", "d4", "c5", "d2", "c8"],
            },
        )
        assert _call(adapter, "winner", state) == "p_sb"
        assert adapter.get_utility(state, "p_sb") == 20.0


# ── Imperfect information ─────────────────────────────────────────────


class TestObservations:
    def test_opponent_hole_hidden(self, adapter: TexasHoldemAdapter):
        state = _resolve_chance(adapter, adapter.create_initial_state())
        obs_sb = adapter.project_observation(state, "p_sb")
        assert len(obs_sb["sb_hole_view"]) == 2 and all("id" in c for c in obs_sb["sb_hole_view"])
        assert len(obs_sb["bb_hole_view"]) == 2 and all("id" not in c for c in obs_sb["bb_hole_view"])
        obs_bb = adapter.project_observation(state, "p_bb")
        assert all("id" in c for c in obs_bb["bb_hole_view"])
        assert all("id" not in c for c in obs_bb["sb_hole_view"])

    def test_info_set_ignores_opponent_hole(self, adapter: TexasHoldemAdapter):
        """Two states differing only in the opponent's hole share an info set."""
        base = _resolve_chance(adapter, adapter.create_initial_state())
        key = adapter.get_info_set_key(base, "p_sb")
        clone = adapter.load_state(
            {
                "_arrays": {
                    "sb_hole": base["_arrays"]["sb_hole"],
                    "bb_hole": ["s2", "s3"],
                    "community": base["_arrays"]["community"],
                    "drawn": base["_arrays"]["drawn"],
                },
                "env": dict(base["env"]),
            }
        )
        assert adapter.get_info_set_key(clone, "p_sb") == key

    def test_adapter_observation(self, adapter: TexasHoldemAdapter):
        state = _resolve_chance(adapter, adapter.create_initial_state())
        obs = adapter.get_observation(state, "p_sb")
        assert obs["hole"] == state["_arrays"]["sb_hole"]
        assert obs["opponent_hole"] == []  # hidden pre-showdown
        assert obs["pot"] == 3
        assert obs["street_name"] == "翻前"
        assert obs["my_stack"] == 99


# ── Rules JSON sanity (v5.1 aliases) ──────────────────────────────────


class TestRulesJSON:
    def test_rules_parse_with_alias_definitions(self):
        rules = json.load(open(RULES_PATH, encoding="utf-8"))
        assert rules["meta"]["gameId"] == "texas_holdem"
        assert len(rules["constants"]["card_ids"]) == 52
        for name, defn in rules["functions"].items():
            assert isinstance(defn.get("params"), list), name
            assert isinstance(defn.get("expr"), dict), name
        declared = set(rules["functions"])
        assert {"eval_five", "best5", "call_to", "min_raise_to", "round_over", "winner", "payoff", "rank"} <= declared

    def test_no_builtin_calls_remain(self):
        text = json.dumps(json.load(open(RULES_PATH, encoding="utf-8")))
        assert "poker_" not in text
        assert "check_line" not in text

    def test_engine_loads_rules(self, adapter: TexasHoldemAdapter):
        state = adapter.create_initial_state()
        assert adapter.get_node_type(state) == "chance"
        outcomes = adapter.get_chance_outcomes(state)
        assert len(outcomes) == 52  # first deal: uniform over 52 cards
        probs = {o.probability for o in outcomes}
        assert probs == {1.0 / 52.0}
