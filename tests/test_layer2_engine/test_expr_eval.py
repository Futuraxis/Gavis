"""Tests for the v5.1 expression language (Layer 2).

Covers the pure-math primitives (choose/range/sort/group/distinct/
contains/sum/max/min/at), alias ``call`` (rules ``functions`` as real
definitions, recursion banned), and — critically — interpreter vs
compiled-closure consistency: both paths must produce identical output
for the same input.
"""

from __future__ import annotations

import pytest

from layer2_engine.core.expr_eval import ExprEvaluator


@pytest.fixture
def ev() -> ExprEvaluator:
    return ExprEvaluator()


def _both(ev: ExprEvaluator, spec, ctx):
    """(interpreter result, compiled result) for the same spec+ctx."""
    compiled = ev.compile(spec)
    return ev.eval(spec, ctx), compiled(ctx)


def _assert_consistent(ev: ExprEvaluator, spec, ctx=None):
    """Assert interpreter and compiled paths agree (and return the value)."""
    interp, compiled = _both(ev, spec, ctx or {})
    assert interp == compiled, f"consistency: {spec}\ninterp={interp}\ncompiled={compiled}"
    return interp


# ── range / sort / group / distinct / contains / aggregates ───────────


class TestCollectionPrimitives:
    def test_range_basic(self, ev):
        assert _assert_consistent(ev, {"range": {"from": 0, "to": 5}}) == [0, 1, 2, 3, 4]

    def test_range_step_and_dynamic(self, ev):
        ctx = {"top": 7, "s": 2}
        assert _assert_consistent(ev, {"range": {"from": 1, "to": {"var": "$top"}, "step": {"var": "$s"}}}, ctx) == [
            1,
            3,
            5,
        ]

    def test_range_zero_step(self, ev):
        assert _assert_consistent(ev, {"range": {"from": 0, "to": 5, "step": 0}}) == []

    def test_sort_default(self, ev):
        assert _assert_consistent(ev, {"sort": {"list": [3, 1, 2]}}) == [1, 2, 3]

    def test_sort_reverse(self, ev):
        assert _assert_consistent(ev, {"sort": {"list": [3, 1, 2], "reverse": True}}) == [3, 2, 1]

    def test_sort_by_key(self, ev):
        spec = {"sort": {"list": ["m3", "m1", "m9", "m5"], "by": {"var": "$node"}}}
        assert _assert_consistent(ev, spec) == ["m1", "m3", "m5", "m9"]

    def test_group(self, ev):
        spec = {"group": {"list": ["m1", "m1", "m2", "m1", "m2"], "by": {"var": "$node"}}}
        groups = _assert_consistent(ev, spec)
        assert [g["key"] for g in groups] == ["m1", "m2"]
        assert [g["count"] for g in groups] == [3, 2]
        assert groups[0]["items"] == ["m1", "m1", "m1"]

    def test_group_by_field(self, ev):
        spec = {
            "group": {
                "list": [{"s": "m", "r": 1}, {"s": "m", "r": 2}, {"s": "p", "r": 1}],
                "by": {"get": ["$node", "s"]},
            }
        }
        groups = _assert_consistent(ev, spec)
        assert [g["key"] for g in groups] == ["m", "p"]
        assert [g["count"] for g in groups] == [2, 1]

    def test_distinct(self, ev):
        assert _assert_consistent(ev, {"distinct": [1, 2, 2, 3, 1]}) == [1, 2, 3]
        assert _assert_consistent(ev, {"distinct": {"var": "$hand"}}, {"hand": ["m1", "m1", "m2"]}) == ["m1", "m2"]

    def test_contains(self, ev):
        spec = {"contains": [{"var": "$list"}, {"var": "$item"}]}
        assert _assert_consistent(ev, spec, {"list": ["m1", "m2"], "item": "m1"})
        assert not _assert_consistent(ev, spec, {"list": ["m1", "m2"], "item": "m3"})
        assert not _assert_consistent(ev, spec, {"list": None, "item": "m1"})

    def test_sum_max_min(self, ev):
        ctx = {"lst": [3, 7, 2]}
        assert _assert_consistent(ev, {"sum": {"var": "$lst"}}, ctx) == 12
        assert _assert_consistent(ev, {"max": {"var": "$lst"}}, ctx) == 7
        assert _assert_consistent(ev, {"min": {"var": "$lst"}}, ctx) == 2
        assert _assert_consistent(ev, {"sum": {"var": "$empty"}}, {}) == 0
        assert _assert_consistent(ev, {"max": {"var": "$empty"}}, {}) is None

    def test_at_list_index(self, ev):
        ctx = {"board": ["a", "b", "c"], "i": 1}
        assert _assert_consistent(ev, {"at": [{"var": "$board"}, {"var": "$i"}]}, ctx) == "b"
        # Out-of-bounds → None (the "natural filter" contract for win checks);
        # negative indices are out of bounds too — no wrap-around.
        assert _assert_consistent(ev, {"at": [{"var": "$board"}, {"const": -1}]}, ctx) is None
        assert _assert_consistent(ev, {"at": [{"var": "$board"}, {"const": 9}]}, ctx) is None
        assert _assert_consistent(ev, {"at": [{"var": "$board"}, {"const": -9}]}, ctx) is None

    def test_at_dict_key(self, ev):
        ctx = {"table": {"m1": 3, "p2": 1}}
        assert _assert_consistent(ev, {"at": [{"var": "$table"}, {"const": "m1"}]}, ctx) == 3
        assert _assert_consistent(ev, {"at": [{"var": "$table"}, {"const": "z9"}]}, ctx) is None


# ── choose ────────────────────────────────────────────────────────────


class TestChoose:
    def test_existence(self, ev):
        spec = {"choose": {"items": [1, 2, 3, 4], "k": 2, "where": {"gt": [{"sum": {"var": "$c"}}, {"const": 6}]}}}
        assert _assert_consistent(ev, spec) is True  # 3+4 = 7

    def test_no_satisfying_combo(self, ev):
        spec = {"choose": {"items": [1, 2, 3], "k": 2, "where": {"gt": [{"sum": {"var": "$c"}}, {"const": 100}]}}}
        assert _assert_consistent(ev, spec) is False

    def test_dedupe_semantics(self, ev):
        """C(unique,k): four copies of m1 count as one tile kind."""
        items = ["m1", "m1", "m1", "m1", "m2", "m3"]
        spec = {
            "choose": {
                "items": items,
                "k": 2,
                "as": "$c",
                "where": {
                    "all": {"list": {"var": "$c"}, "as": "$t", "where": {"eq": [{"var": "$t"}, {"const": "m1"}]}}
                },
            }
        }
        # Only one m1 in the pool → no pair of m1s is possible.
        assert _assert_consistent(ev, spec) is False

    def test_dedupe_still_selects_different_kinds(self, ev):
        items = ["m1", "m1", "m1", "m1", "m2", "m3"]
        spec = {"choose": {"items": items, "k": 2, "where": {"contains": [{"var": "$c"}, {"const": "m2"}]}}}
        assert _assert_consistent(ev, spec) is True

    def test_prefix_prunes_branch(self, ev):
        """Prefix failing on a partial prunes every extension of it.

        The monotone contract: prefix is a necessary condition on
        partials — if it fails on a partial, extensions cannot satisfy
        ``where``.  Here the prefix requires the first chosen item to be
        'a', so branches starting with 'b'/'c' are never explored.
        """
        spec = {
            "choose": {
                "items": ["a", "b", "c"],
                "k": 2,
                "as": "$c",
                "prefix": {"eq": [{"at": [{"var": "$c"}, {"const": 0}]}, {"const": "a"}]},
                "where": {"contains": [{"var": "$c"}, {"const": "b"}]},
            },
        }
        assert _assert_consistent(ev, spec) is True  # [a, b] passes where

    def test_prefix_blocks_everything(self, ev):
        spec = {
            "choose": {
                "items": ["a", "b", "c"],
                "k": 2,
                "as": "$c",
                "prefix": {"eq": [{"at": [{"var": "$c"}, {"const": 0}]}, {"const": "x"}]},
                "where": {"const": True},
            },
        }
        assert _assert_consistent(ev, spec) is False

    def test_prefix_sound_with_sum(self, ev):
        """Sum-style pruning: partial sums > target can never recover."""
        items = [1, 2, 3, 4, 5, 20]
        where = {"eq": [{"sum": {"var": "$c"}}, {"const": 6}]}
        base = {"items": items, "k": 3, "as": "$c", "where": where}
        no_prefix = _assert_consistent(ev, {"choose": {**base}})  # (1,2,3)
        with_prefix = _assert_consistent(
            ev, {"choose": {**base, "prefix": {"lte": [{"sum": {"var": "$c"}}, {"const": 6}]}}}
        )
        assert no_prefix is True and with_prefix is True

    def test_prefix_never_vetoes_full_combo(self, ev):
        """``where`` is the sole authority on full-length combinations."""
        spec = {
            "choose": {
                "items": ["a", "b", "c"],
                "k": 3,
                "as": "$c",
                "prefix": {"lt": [{"count": {"var": "$c"}}, {"const": 3}]},
                "where": {"eq": [{"count": {"var": "$c"}}, {"const": 3}]},
            },
        }
        assert _assert_consistent(ev, spec) is True

    def test_prefix_blocks_all(self, ev):
        # A prefix that always fails → no combination ever completes.
        spec = {"choose": {"items": [1, 2, 3], "k": 2, "prefix": {"const": False}, "where": {"const": True}}}
        assert _assert_consistent(ev, spec) is False

    def test_then_agg_max(self, ev):
        # Poker best5 shape: max of the value over all 3-combos.
        spec = {"choose": {"items": [1, 2, 3, 4], "k": 3, "as": "$c", "then": {"sum": {"var": "$c"}}, "agg": "max"}}
        assert _assert_consistent(ev, spec) == 9  # 2+3+4

    def test_then_agg_min(self, ev):
        spec = {"choose": {"items": [1, 2, 3, 4], "k": 2, "as": "$c", "then": {"sum": {"var": "$c"}}, "agg": "min"}}
        assert _assert_consistent(ev, spec) == 3

    def test_choose_sorted_input(self, ev):
        # Input is sorted by value before dedupe — order must not matter.
        a = {
            "choose": {
                "items": ["p2", "m1", "z5"],
                "k": 1,
                "as": "$c",
                "where": {"contains": [{"var": "$c"}, {"const": "m1"}]},
            }
        }
        b = {
            "choose": {
                "items": ["m1", "p2", "z5"],
                "k": 1,
                "as": "$c",
                "where": {"contains": [{"var": "$c"}, {"const": "m1"}]},
            }
        }
        assert _assert_consistent(ev, a) is True
        assert _assert_consistent(ev, b) is True


# ── alias call ────────────────────────────────────────────────────────

_ALIASES = {
    "rank": {
        "params": ["card"],
        "expr": {"at": [{"var": "$constants.rank_of"}, {"var": "$card"}]},
    },
    "double": {
        "params": ["x"],
        "expr": {"expr": "$x * 2"},
    },
    "fib": {  # would recurse — must be rejected statically
        "params": ["n"],
        "expr": {"call": ["fib", {"var": "$n"}]},
    },
    "two_step": {  # indirect recursion A→B→A
        "params": ["x"],
        "expr": {"call": ["two_step_b", {"var": "$x"}]},
    },
    "two_step_b": {
        "params": ["x"],
        "expr": {"call": ["two_step", {"var": "$x"}]},
    },
    "sum3": {
        "params": ["a", "b", "c"],
        "expr": {"expr": "$a + $b + $c"},
    },
}


def _ev_with_aliases() -> ExprEvaluator:
    ev = ExprEvaluator()
    ev.set_functions(_ALIASES)
    return ev


class TestAliasCall:
    def test_binds_params_and_evaluates_body(self, ev):
        ev.set_functions({"double": _ALIASES["double"]})
        assert ev.eval({"call": ["double", {"const": 21}]}, {}) == 42
        assert ev.compile({"call": ["double", {"const": 21}]})({}) == 42

    def test_multi_param(self, ev):
        ev.set_functions({"sum3": _ALIASES["sum3"]})
        spec = {"call": ["sum3", {"const": 1}, {"const": 2}, {"const": 39}]}
        assert _assert_consistent(ev, spec) == 42

    def test_body_uses_caller_context(self, ev):
        ev.set_functions({"rank": _ALIASES["rank"]})
        ctx = {"$constants": {"rank_of": {"m1": 1, "p2": 2}}}
        spec = {"call": ["rank", {"const": "m1"}]}
        assert _assert_consistent(ev, spec, ctx) == 1

    def test_nested_calls_compile(self, ev):
        ev.set_functions({"double": _ALIASES["double"], "sum3": _ALIASES["sum3"]})
        spec = {"call": ["sum3", {"call": ["double", {"const": 3}]}, {"const": 4}, {"const": 5}]}
        assert _assert_consistent(ev, spec) == 15

    def test_unknown_function(self, ev):
        with pytest.raises(ValueError, match="Unknown function"):
            ev.eval({"call": ["nope", {"const": 1}]}, {})

    def test_direct_recursion_rejected(self, ev):
        ev.set_functions({"fib": _ALIASES["fib"]})
        with pytest.raises(RecursionError, match="fib"):
            ev.eval({"call": ["fib", {"const": 5}]}, {})
        with pytest.raises(RecursionError, match="fib"):
            ev.compile({"call": ["fib", {"const": 5}]})({})

    def test_indirect_recursion_rejected(self, ev):
        ev.set_functions({"two_step": _ALIASES["two_step"], "two_step_b": _ALIASES["two_step_b"]})
        with pytest.raises(RecursionError, match="two_step"):
            ev.eval({"call": ["two_step", {"const": 1}]}, {})

    def test_healthy_alias_survives_cycle_detection(self, ev):
        ev.set_functions({"double": _ALIASES["double"], "fib": _ALIASES["fib"]})
        assert ev.eval({"call": ["double", {"const": 2}]}, {}) == 4

    def test_declaration_stub_skipped(self, ev):
        """v5.0-style metadata-only entries are skipped, not fatal."""
        ev.set_functions({"declared_only": {"description": "x", "pure": True}})
        with pytest.raises(ValueError, match="Unknown function"):
            ev.eval({"call": ["declared_only"]}, {})

    def test_wrong_arity(self, ev):
        ev.set_functions({"double": _ALIASES["double"]})
        with pytest.raises(ValueError, match="expected 1 args"):
            ev.eval({"call": ["double", {"const": 1}, {"const": 2}]}, {})


# ── v5.1 win-check shape (gomoku sliding window) ──────────────────────


# Row/col-decomposed win check around the last placed cell (3×3 slice).
# For each direction (dr, dc), a window of ``win_length`` cells through
# the last cell must all hold ``piece``.  Cell (r', c') is only
# evaluated when in bounds — pure index-step arithmetic would wrap
# across board edges (e.g. index 3 + step 2 → index 5 on a 3×3 board:
# (1,0) → (1,2), not collinear), so the guard is mandatory.
def _win_spec() -> dict:
    return {
        "any": {
            "list": {"var": "$dirs"},
            "as": "$d",
            "where": {
                "any": {
                    "list": {"range": {"from": 0, "to": {"var": "$win_length"}}},
                    "as": "$k",
                    "where": {
                        "all": {
                            "list": {"range": {"from": 0, "to": {"var": "$win_length"}}},
                            "as": "$j",
                            "where": {
                                "and": [
                                    {
                                        "and": [
                                            {"gte": [{"expr": "c // size + (j - k) * d.dr"}, {"const": 0}]},
                                            {"lt": [{"expr": "c // size + (j - k) * d.dr"}, {"var": "$size"}]},
                                            {"gte": [{"expr": "c % size + (j - k) * d.dc"}, {"const": 0}]},
                                            {"lt": [{"expr": "c % size + (j - k) * d.dc"}, {"var": "$size"}]},
                                        ]
                                    },
                                    {
                                        "eq": [
                                            {
                                                "at": [
                                                    {"var": "$board"},
                                                    {
                                                        "expr": "(c // size + (j - k) * d.dr) * size + c % size + (j - k) * d.dc"
                                                    },
                                                ]
                                            },
                                            {"var": "piece"},
                                        ]
                                    },
                                ],
                            },
                        }
                    },
                }
            },
        }
    }


def _win_ctx(board, c, piece="p_black", size=3, win_length=3):
    return {
        "$board": board,
        "piece": piece,
        "c": c,
        "size": size,
        "win_length": win_length,
        "dirs": [{"dr": 0, "dc": 1}, {"dr": 1, "dc": 0}, {"dr": 1, "dc": 1}, {"dr": 1, "dc": -1}],
    }


class TestWinCheckShape:
    def test_sliding_window_win(self, ev):
        board = ["p_black", "p_black", "p_black", None, None, None, None, None, None]
        assert _assert_consistent(ev, _win_spec(), _win_ctx(board, c=0)) is True
        # Win through the last cell, not anchored at 0 (vertical: 1, 4, 7).
        board = [None, "p_black", None, None, "p_black", None, None, "p_black", None]
        assert _assert_consistent(ev, _win_spec(), _win_ctx(board, c=1)) is True

    def test_sliding_window_no_win(self, ev):
        board = ["p_black", "p_black", "p_white", None, None, None, None, None, None]
        assert _assert_consistent(ev, _win_spec(), _win_ctx(board, c=0)) is False

    def test_window_near_board_edge(self, ev):
        """c=8 (corner): windows run off the edge → guard filters them."""
        board = ["p_black", "p_black", None, None, None, None, None, None, "p_black"]
        assert _assert_consistent(ev, _win_spec(), _win_ctx(board, c=8)) is False

    def test_anti_diagonal_no_wrap_regression(self, ev):
        """The moon-chess false win: pieces at indices 1, 3, 5 on a 3×3
        board — (0,1), (1,0), (1,2) — are NOT collinear.  Plain index-step
        arithmetic (step 2) would report a win; the row/col guard must not."""
        board = [None, "p_black", None, "p_black", None, "p_black", None, None, None]
        assert _assert_consistent(ev, _win_spec(), _win_ctx(board, c=5)) is False

    def test_anti_diagonal_true_win(self, ev):
        """A genuine anti-diagonal (2,4,6) must still be detected."""
        board = [None, None, "p_black", None, "p_black", None, "p_black", None, None]
        assert _assert_consistent(ev, _win_spec(), _win_ctx(board, c=4)) is True

    def test_diagonal_true_win(self, ev):
        board = ["p_black", None, None, None, "p_black", None, None, None, "p_black"]
        assert _assert_consistent(ev, _win_spec(), _win_ctx(board, c=8)) is True
