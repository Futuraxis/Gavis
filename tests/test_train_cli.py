"""Tests for the train-cli game registry and the unified training script.

Covered:

1. Registry integrity — every registered game resolves to an existing
   rules file, builds a ``GameEngine`` (variant/player-count applied via
   the v5.2 ``variants`` declaration), and its seats are declared in the
   rules' ``players``.
2. Generic runtime factory (``create_solver``) — data-driven availability
   (CFR excluded for Texas Hold'em, ``imperfect_information`` enabled for
   its Hybrid via registry config), unknown game/solver → ``ValueError``.
3. Baseline ``RandomSolver`` plays legal actions.
4. The generic ``evaluate`` loop (vs uniform random, seat rotation) runs
   end-to-end on Moon Chess without any training.

These tests deliberately avoid actual training (CFR iters / MARL episodes
are minutes-long); they pin the registrar's data and the trainer's generic
plumbing.  The ``# noqa: F401, I001`` on the first import is required: the
``train-cli/`` directory is hyphenated, so it is reachable only after the
``train_cli`` bridge puts it on ``sys.path``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

import train_cli  # noqa: F401, I001 — 导入桥副作用：train-cli/ 进入 sys.path
from train_cli import GAMES, RandomSolver, build_engine, create_solver, evaluate

RULES_DIR = Path(__file__).resolve().parent.parent / "rules"

PLATFORM_GAME_IDS = (
    "moon_chess",
    "stochastic_gomoku",
    "texas_holdem",
    "mahjong_guangdong",
    "mahjong_hongzhong",
    "mahjong_blood",
    "mahjong_sichuan",
    "mahjong_changsha",
    "mahjong_taiwan",
)

UNO_GAME_IDS = (
    "uno",
    "uno_seven_zero",
    "uno_jump_in",
    "uno_stacking",
    "uno_draw_until",
    "uno_strict_wild4",
)


# ── 注册表完整性 ──────────────────────────────────────────────────


def test_registered_games_cover_platform_plus_deduction_games() -> None:
    assert "werewolf" in GAMES
    assert "undercover" in GAMES  # 谁是卧底（text 发言 + 投票桌游）
    for game_id in PLATFORM_GAME_IDS:
        assert game_id in GAMES
    for game_id in UNO_GAME_IDS:
        assert game_id in GAMES
    #: 平台内置 3 + 麻将 6 变种 + 狼人杀 + 谁是卧底 + UNO 6 变种 = 17。
    assert len(GAMES) == 17


def test_every_game_rules_file_exists() -> None:
    for spec in GAMES.values():
        assert (RULES_DIR / spec.engine.rules).is_file(), f"{spec.game_id}: {spec.engine.rules}"


def test_every_game_builds_engine() -> None:
    for spec in GAMES.values():
        engine = build_engine(spec, seed=42)
        engine.create_initial_state()


def test_players_declared_in_rules() -> None:
    for spec in GAMES.values():
        with open(RULES_DIR / spec.engine.rules, encoding="utf-8") as f:
            rules = json.load(f)
        declared = {p if isinstance(p, str) else p["id"] for p in rules["players"]}
        for seat in spec.players:
            assert seat in declared, f"{spec.game_id}: 座位 {seat} 不在 rules 声明中"


def _resolve_to_player(engine, state, max_steps: int = 80):
    """按通用对局循环推进 chance 节点直到出现 player 节点（验证引擎装配）。"""
    rng = random.Random(1)
    while engine.get_node_type(state) != "player" and not engine.is_terminal(state) and max_steps > 0:
        if engine.get_node_type(state) == "chance":
            outcomes = engine.get_chance_outcomes(state)
            if not outcomes:
                break
            probs = [float(getattr(o, "probability", 0.0) or 0.0) for o in outcomes]
            if sum(probs) <= 0:
                state = engine.apply_chance(state, rng.choice(outcomes))
            else:
                state = engine.apply_chance(state, rng.choices(outcomes, weights=probs, k=1)[0])
        else:
            break
        max_steps -= 1
    return state


def test_mahjong_variants_select_variant_and_player_count() -> None:
    for spec in (
        GAMES["mahjong_guangdong"],
        GAMES["mahjong_hongzhong"],
        GAMES["mahjong_blood"],
        GAMES["mahjong_sichuan"],
        GAMES["mahjong_changsha"],
        GAMES["mahjong_taiwan"],
    ):
        assert spec.engine.rules == "mahjong.json"
        # 麻将标准 4 人：引擎装配、训练（MARL）与评估全部按 4 座位进行。
        assert spec.engine.player_count == 4
        assert spec.players == ("p0", "p1", "p2", "p3")
        engine = build_engine(spec, seed=42)
        assert engine.variant == spec.engine.variant
        assert engine.player_count == spec.engine.player_count
        state = _resolve_to_player(engine, engine.create_initial_state())
        assert engine.get_current_player(state) in spec.players


# ── 通用运行时工厂（create_solver）────────────────────────────────


def test_uno_variants_select_variant_and_player_count() -> None:
    for spec in (GAMES[gid] for gid in UNO_GAME_IDS):
        assert spec.engine.rules == "uno.json"
        assert spec.engine.player_count == 4
        engine = build_engine(spec, seed=42)
        assert engine.variant == spec.engine.variant
        assert engine.player_count == spec.engine.player_count
        state = _resolve_to_player(engine, engine.create_initial_state())
        assert engine.get_current_player(state) in spec.players


def test_create_solver_moon_chess_all_runtime_solvers() -> None:
    engine = build_engine(GAMES["moon_chess"], seed=42)
    for name in ("mcts", "cfr", "hybrid", "random"):
        solver = create_solver("moon_chess", name, engine, seed=42, budget=500)
        assert solver is not None
    mcts = create_solver("moon_chess", "mcts", engine, seed=42, budget=500)
    assert mcts.config.budget == 500


def test_create_solver_texas_hybrid_enables_imperfect_information() -> None:
    engine = build_engine(GAMES["texas_holdem"], seed=42)
    solver = create_solver("texas_holdem", "hybrid", engine, seed=42, budget=500)
    assert solver.config.imperfect_information is True


def test_create_solver_cfr_excluded_for_texas_holdem() -> None:
    engine = build_engine(GAMES["texas_holdem"], seed=42)
    with pytest.raises(ValueError, match="不适用"):
        create_solver("texas_holdem", "cfr", engine, seed=42, budget=500)


def test_create_solver_unknown_game_or_solver() -> None:
    engine = build_engine(GAMES["moon_chess"], seed=42)
    with pytest.raises(ValueError, match="未知游戏"):
        create_solver("no_such_game", "mcts", engine, seed=42, budget=500)
    with pytest.raises(ValueError, match="未知求解器"):
        create_solver("moon_chess", "no_such_solver", engine, seed=42, budget=500)


# ── 基准求解器与通用评估 ──────────────────────────────────────────


def test_random_solver_selects_legal_action() -> None:
    engine = build_engine(GAMES["moon_chess"], seed=42)
    state = engine.create_initial_state()
    legal = engine.get_legal_actions(state)
    solver = RandomSolver(engine, seed=7)
    action = solver.select_action(state)
    assert action in legal


def test_evaluate_loop_runs_end_to_end() -> None:
    spec = GAMES["moon_chess"]
    engine = build_engine(spec, seed=42)
    result = evaluate(engine, spec, None, None, episodes=4, base_seed=1)
    # 默认返回按对手的 dict（random 列是下限基准）。
    assert set(result) == {"random"}
    r = result["random"]
    assert r["episodes"] == 4
    assert r["wins"] + r["draws"] + r["losses"] == 4
    assert 0.0 <= r["win_rate"] <= 1.0
    assert r["seconds"] >= 0.0


def test_evaluate_opponent_seats_play_random_not_stop() -> None:
    """回归：评估时非 own 座位（owners 中 None）必须均匀随机落子而非中止。

    曾是 play_episode 的 bug：``solver is None → break`` 使对手回合立即截断，
    月亮棋评估出现"全平"（utility 恒 0）。own 座位是 RandomSolver、对手 None 时，
    月亮棋每局必须分出胜负（wins+losses==episodes），不允许 0 胜负全平。
    """
    from train_cli import RandomSolver

    spec = GAMES["moon_chess"]
    engine = build_engine(spec, seed=42)
    result = evaluate(engine, spec, RandomSolver(engine, seed=7), None, episodes=6, base_seed=1)
    r = result["random"]
    assert r["episodes"] == 6
    assert r["wins"] + r["losses"] == 6, f"对手座位未参与对局: {result}"


def test_evaluate_self_and_mcts_columns() -> None:
    """多对手评估：self（自博弈镜像）与 mcts（基线）列必须真实对局（月亮棋均分出胜负）。"""
    from train_cli import EVAL_MCTS_BUDGET, RandomSolver

    spec = GAMES["moon_chess"]
    engine = build_engine(spec, seed=42)
    solver = RandomSolver(engine, seed=7)
    result = evaluate(engine, spec, solver, None, episodes=4, base_seed=1, opponents=("random", "self", "mcts"))
    assert set(result) == {"random", "self", "mcts"}
    for opp, r in result.items():
        assert r["episodes"] == 4
        assert r["wins"] + r["losses"] == 4, f"{opp} 列未到达终局: {result}"
    assert EVAL_MCTS_BUDGET == 300  # 基线预算与 Hybrid 自身 mcts_budget 同量级


def test_evaluate_mcts_column_skipped_when_unavailable() -> None:
    """数据驱动：游戏未登记 mcts 运行时求解器时，mcts 评估列自动跳过（狼人杀）。"""
    from train_cli import RandomSolver

    spec = GAMES["werewolf"]
    engine = build_engine(spec, seed=42)
    result = evaluate(
        engine, spec, RandomSolver(engine, seed=7), None, episodes=2, base_seed=1, opponents=("random", "mcts")
    )
    assert set(result) == {"random"}  # mcts 不在 werewolf.runtime_solvers → 跳过
    assert result["random"]["episodes"] == 2
