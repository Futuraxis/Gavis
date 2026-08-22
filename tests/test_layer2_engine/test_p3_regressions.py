"""Regression tests for the Layer2 core engine P3 review batch (2026-08-22).

Covers the behavior-changing P3 fixes from .docs/review/layer2_engine.md:
dead code removal (15), $var arithmetic codegen (16), trigger cascades (17),
branch/inc/remove/forEach None guards (18/19), non-square trimByKey (20),
count(dict) (21), numeric chance canonicalKey (22), cycle-detection scope (23),
length-expr longest-name replacement (25), unknown-effector warning (26),
sample_chance protocol symmetry (27), cartesian cap (28), view cache (29),
compile-without-engine guard (30), view field call (31).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from layer2_engine.core.engine import _MAX_ACTION_COMBINATIONS, GameEngine, _cartesian_product
from layer2_engine.core.expr_eval import ExprEvaluator
from layer2_engine.core.rules_compiler import RulesCompiler, UnsupportedShapeError, _Gen
from layer2_engine.core.state_graph import _eval_length_expr
from layer2_engine.interfaces.solver_adapter import SolverAdapter

RULES_DIR = Path(__file__).resolve().parent.parent.parent / "rules"


def _load(game: str) -> GameEngine:
    with open(RULES_DIR / f"{game}.json", "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=42)


def _player_rules() -> dict:
    """Synthetic rules with a single player phase and no chance nodes."""
    return {
        "constants": {},
        "players": [{"id": "p0"}],
        "groundState": {"env": {"type": "env", "fields": {"phase": {"type": "str", "initial": "p1"}}}},
        "derivedViews": {},
        "phases": [{"id": "p1", "actions": ["go"]}],
        "actions": [],
        "effectors": {},
        "chance": [],
        "terminal": [],
        "utility": [],
        "visibility": {},
        "queries": {},
        "functions": {},
        "triggers": [],
    }


# ── 15: 死代码已删除（无独立测试；_Gen 不再有 _with_item / 重复副本） ──


def test_with_item_removed():
    assert not hasattr(_Gen, "_with_item")


# ── 16: 含 $ 变量的算术串可以编译（此前必然回退解释器） ──────────────


def test_arith_with_prefixed_var_compiles():
    gen = _Gen({"two": 2}, {}, {}, {"cell": "node"})
    src = gen.expr({"expr": "$cell.x * 2 + two"})
    assert eval(compile(src, "<arith>", "eval"), {"node": {"x": 3}}) == 8


def test_arith_unresolvable_var_falls_back():
    # 上下文不可解析的 $ 名 → UnsupportedShapeError（调用方回退解释器）
    with pytest.raises(UnsupportedShapeError):
        _Gen({}, {}, {}, {}).expr({"expr": "$unknown.thing + 1"})
    # 非数值裸标识符同样回退
    with pytest.raises(UnsupportedShapeError):
        _Gen({}, {}, {}, {}).expr({"expr": "hand_len + 1"})


# ── 17: 触发链级联事件会被继续处理 ─────────────────────────────────────


def test_trigger_cascade_processes_nested_events():
    rules = _player_rules()
    rules["groundState"]["log"] = {"type": "array", "length": {"const": 4}, "mutable": True}
    rules["actions"] = [{"id": "go", "type": "action", "phases": ["p1"], "effectRef": "emit_a",
                         "canonicalKey": {"const": "go"}, "legal": {"const": True}}]
    rules["effectors"] = {
        "emit_a": {"ops": [{"op": "emit", "event": "a", "payload": {}}]},
        "emit_b": {"ops": [{"op": "emit", "event": "b", "payload": {}}]},
        "log_b": {"ops": [{"op": "append", "array": "log", "value": {"const": "b-seen"}}]},
    }
    rules["triggers"] = [
        {"event": "a", "effectRef": "emit_b"},
        {"event": "b", "effectRef": "log_b"},
    ]
    engine = GameEngine(rules, seed=1)
    state = engine.apply_action(
        engine.create_initial_state(),
        __import__("layer2_engine.interfaces.solver_adapter", fromlist=["ActionInstance"]).ActionInstance(
            "go", "action", "p0", {}, "go"
        ),
    )
    # a → trigger → emit b → trigger → append；级联事件被处理而非丢弃
    assert state["_arrays"]["log"] == ["b-seen"]
    assert state.get("_pending_events", []) == []
    assert state.get("_pending_effects", []) == []


# ── 18/19: branch 缺 then / inc / forEach 的 None 守卫 ─────────────────


def test_branch_missing_then_no_crash():
    rules = _player_rules()
    rules["actions"] = [{"id": "go", "type": "action", "phases": ["p1"], "effectRef": "br",
                         "canonicalKey": {"const": "go"}, "legal": {"const": True}}]
    rules["effectors"] = {"br": {"ops": [{"op": "branch", "if": {"const": True}}]}}
    engine = GameEngine(rules, seed=1)
    from layer2_engine.interfaces.solver_adapter import ActionInstance

    state = engine.apply_action(engine.create_initial_state(), ActionInstance("go", "action", "p0", {}, "go"))
    assert state["env"]["phase"] == "p1"  # 分支无操作、不崩溃


def test_inc_remove_foreach_none_guards():
    rules = _player_rules()
    rules["groundState"]["hand"] = {"type": "array", "length": {"const": 3}, "mutable": False}
    rules["groundState"]["env"]["fields"]["turns"] = {"type": "int", "initial": 0}
    rules["actions"] = [{"id": "go", "type": "action", "phases": ["p1"], "effectRef": "ops",
                         "canonicalKey": {"const": "go"}, "legal": {"const": True}}]
    rules["effectors"] = {
        "ops": {
            "ops": [
                {"op": "inc", "key": "turns", "by": {"var": "$nonexistent"}},
                {"op": "remove", "array": "hand", "value": {"const": 1}, "count": {"var": "$nonexistent"}},
                {"op": "forEach", "list": {"var": "$nonexistent"}, "do": []},
            ]
        }
    }
    engine = GameEngine(rules, seed=1)
    from layer2_engine.interfaces.solver_adapter import ActionInstance

    state = engine.create_initial_state()
    state["_arrays"]["hand"] = [1, 2, 1]
    state = engine.apply_action(state, ActionInstance("go", "action", "p0", {}, "go"))
    assert state["env"]["turns"] == 1  # inc by None → 默认 1
    assert state["_arrays"]["hand"] == [2, 1]  # remove count None → 默认 1


# ── 20: trimByKey 非正方形棋盘（cols 来自 grid 视图而非 sqrt） ─────────


def test_trim_by_key_non_square_cols():
    rules = _player_rules()
    rules["groundState"]["board"] = {"type": "array", "length": {"const": 12}, "mutable": False}
    rules["groundState"]["order"] = {"type": "array", "length": {"const": 8}, "mutable": True}
    rules["derivedViews"] = {
        "cell": {"from": {"type": "grid", "array": "board", "cols": {"const": 4}},
                 "fields": {"id": {"template": "cell_{$row}_{$col}"}}}
    }
    rules["actions"] = [{"id": "go", "type": "action", "phases": ["p1"], "effectRef": "do_trim",
                         "canonicalKey": {"const": "go"}, "legal": {"const": True}}]
    rules["effectors"] = {
        "do_trim": {
            "ops": [
                {"op": "trimByKey", "array": "order", "key": "player_id", "value": {"const": "p0"}, "max": {"const": 1},
                 "onEvict": [{"op": "setIndex", "array": "board", "at": {"var": "evicted_index"}, "value": None}]}
            ]
        }
    }
    engine = GameEngine(rules, seed=1)
    from layer2_engine.interfaces.solver_adapter import ActionInstance

    state = engine.create_initial_state()
    state["_arrays"]["board"] = [1] * 12
    state["_arrays"]["order"] = [
        {"cell_id": "cell_1_2", "player_id": "p0"},  # 3×4 方阵下 flat 索引 = 1*4+2 = 6
        {"cell_id": "cell_2_0", "player_id": "p0"},
    ]
    state = engine.apply_action(state, ActionInstance("go", "action", "p0", {}, "go"))
    assert [e["cell_id"] for e in state["_arrays"]["order"]] == ["cell_2_0"]
    assert state["_arrays"]["board"][6] is None  # sqrt 猜测 (1*3+2=5) 会写错位置


# ── 21: 编译版 count 接受 dict（与解释器一致） ─────────────────────────


def test_compiled_count_accepts_dict():
    gen = _Gen({}, {}, {}, {})
    src = gen.expr({"count": {"const": {"a": 1, "b": 2}}})
    assert eval(compile(src, "<count>", "eval"), {}) == 2
    assert eval(compile(gen.expr({"count": {"const": [1, 2, 3]}}), "<count>", "eval"), {}) == 3
    assert eval(compile(gen.expr({"count": {"const": "abc"}}), "<count>", "eval"), {}) == 0


# ── 22: 数值 outcome 的 canonicalKey 不再被字符串化 ────────────────────


def test_chance_canonical_key_numeric_outcome():
    rules = _player_rules()
    rules["chance"] = [{"phases": ["p1"], "probability": {"explicit": [
        {"outcome": 3, "prob": 0.5}, {"outcome": 4, "prob": 0.5}]},
        "canonicalKey": {"eq": [{"var": "$outcome"}, {"const": 3}]}}]
    engine = GameEngine(rules, seed=1)
    state = engine.create_initial_state()
    compiled = engine._compiled
    assert compiled is not None and compiled.chance_outcomes is not None
    keys = [o.canonical_key for o in compiled.chance_outcomes(state)]
    assert keys == [True, False]  # 此前编译版绑定了 '3'（str），恒 False


# ── 23: 环检测只标记环成员，不误伤调用链上层 ───────────────────────────


def test_cycle_detection_marks_only_cycle_members():
    ev = ExprEvaluator()
    ev.set_functions({
        "d": {"params": ["x"], "expr": {"call": ["c", {"var": "$x"}]}},
        "c": {"params": ["x"], "expr": {"call": ["b", {"var": "$x"}]}},
        "b": {"params": ["x"], "expr": {"call": ["a", {"var": "$x"}]}},
        "a": {"params": ["x"], "expr": {"call": ["b", {"var": "$x"}]}},
    })
    # C→B→A→B：只有 A/B 是环成员；C/D 只是可达环，不应被标记
    assert ev._cyclic == {"a", "b"}
    assert "c" not in ev._cyclic and "d" not in ev._cyclic
    # 调用环仍被拦截
    with pytest.raises(RecursionError):
        ev.eval({"call": ["d", {"const": 1}]}, {})


# ── 25: 长度表达式常量替换按长名优先 + 词边界 ──────────────────────────


def test_length_expr_longest_name_first():
    constants = {"size": 2, "board_size": 9}
    assert _eval_length_expr({"expr": "board_size * size"}, constants) == 18
    assert _eval_length_expr({"expr": "board_size + size"}, constants) == 11
    # 词边界：短名不误伤长名内部
    assert _eval_length_expr({"expr": "board_size"}, {"size": 2, "board_size": 9}) == 9
    assert _eval_length_expr({"expr": "1 + 1"}, {"one": 1}) == 2


# ── 26: 未知 effector 打日志而非静默吞掉 ───────────────────────────────


def test_unknown_effector_logs_warning(caplog):
    engine = _load("moon_chess")
    engine._execute_effector("no_such_effector", {"$state": {}}, engine.create_initial_state())
    assert any("no_such_effector" in r.message for r in caplog.records)


# ── 27: sample_chance 进入 Protocol（可选，带默认抛错实现） ─────────────


def test_sample_chance_optional_in_protocol():
    assert hasattr(SolverAdapter, "sample_chance")
    with pytest.raises(NotImplementedError):
        SolverAdapter.sample_chance(None, {})  # 默认实现为占位
    # 引擎实现可用
    engine = _load("texas_holdem")
    assert callable(engine.sample_chance)


# ── 28: 笛卡尔积规模上限 ──────────────────────────────────────────────


def test_cartesian_product_cap():
    with pytest.raises(ValueError):
        _cartesian_product({"a": list(range(300)), "b": list(range(300))})  # 90000 > 65536
    assert len(_cartesian_product({"a": [1, 2], "b": [3, 4]})) == 4
    assert _MAX_ACTION_COMBINATIONS >= 65536


# ── 29: 视图缓存命中 + 变更失效 ───────────────────────────────────────


def test_view_cache_hit_and_invalidation():
    engine = _load("moon_chess")
    state = engine.create_initial_state()
    first = engine._materialize_view(state, "cell")
    assert engine._materialize_view(state, "cell") is first  # 缓存命中
    # 原地变更数组 → 缓存失效
    engine._execute_op({"op": "setIndex", "array": "board", "at": {"const": 0}, "value": {"const": "p_black"}},
                       {"$state": state}, state)
    second = engine._materialize_view(state, "cell")
    assert second is not first
    assert second[0]["value"] == "p_black"
    # clone 不携带缓存
    from layer2_engine.core.state_graph import clone_state

    assert "_view_cache" not in clone_state(state)


def test_project_observation_partial_info_builds_ctx_once():
    # 部分可观测路径下每次调用不重复重建完整 context（行为不变，仅结构）
    engine = _load("texas_holdem")
    state = engine.create_initial_state()
    obs = engine.project_observation(state, "p_sb")
    assert "phase" in obs["env"]  # 结构完整


# ── 30: 无 engine 编译时 skipped 模板不再生成 _engine 引用 ─────────────


def test_compile_without_engine_skips_skipped_templates():
    rules = _player_rules()
    rules["actions"] = [{"id": "multi", "type": "action", "phases": ["p1"], "effectRef": "do_x",
                         "params": {"a": {"domain": [1, 2]}, "b": {"domain": [3]}},
                         "legal": {"const": True}, "canonicalKey": {"template": "m{$a}{$b}"}}]
    artifacts = RulesCompiler().compile(rules)  # engine=None
    assert artifacts.legal_actions is None  # 无法运行时 fallback → 整件禁用

    engine = GameEngine(rules, seed=1)
    compiled = engine._compiled
    assert compiled is not None and compiled.legal_actions is not None
    state = engine.create_initial_state()
    mine = {a.canonical_key for a in compiled.legal_actions(state)}
    theirs = {a.canonical_key for a in engine._interp_legal_actions(state)}
    assert mine == theirs == {"m13", "m23"}


# ── 31: 视图字段 call 使用规则 alias（不再 Unknown function 崩溃） ─────


def test_view_field_call_with_alias():
    rules = _player_rules()
    rules["groundState"]["a"] = {"type": "array", "length": {"const": 3}, "mutable": False}
    rules["derivedViews"] = {
        "nums": {"from": {"type": "enum", "array": "a"},
                 "fields": {"d": {"call": ["double", {"get": ["$self", "value"]}]}}}
    }
    rules["functions"] = {"double": {"params": ["x"], "expr": {"mul": [{"var": "$x"}, {"const": 2}]}}}
    engine = GameEngine(rules, seed=1)
    state = engine.create_initial_state()
    state["_arrays"]["a"] = [1, 2, 3]
    entities = engine._materialize_view(state, "nums")
    assert [e["d"] for e in entities] == [2, 4, 6]
