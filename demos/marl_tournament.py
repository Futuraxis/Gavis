#!/usr/bin/env python3
"""MARL 单循环赛 — 已训练的 QMix / HAPPO / MAAC 两两对抗并记录详细结果.

对每个游戏、每对求解器 (A, B) 打主/客场两轮：一轮 A 执先手、一轮 B 执先手。
逐局新建 adapter（种子逐局推进）保证发牌多样性；对局用轻量循环直接驱动
``select_action``（不做训练式 transition 编码），记录 payoff、步数与胜者。

产物保存到 ``<out-dir>/<game>.json``：
  - ``matchups``: 每对求解器 × 方向的逐局记录（种子、步数、双方 payoff、胜者）
  - ``summary``:  胜负平汇总、平均 payoff、平均步数
  - ``players``:  先手玩家 id 顺序

Usage:  python -m demos.marl_tournament [--game all|moon_chess|texas_holdem|mahjong_2p]
                                         [--games-per-match N] [--seed N]
                                         [--model-dir models/marl]
                                         [--out-dir data/marl_tournament] [--verbose]
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path

from layer2_engine.games.mahjong.mahjong_adapter import MahjongAdapter
from layer2_engine.games.moon_chess.moon_env_adapter import MoonChessAdapter
from layer2_engine.games.texas_holdem.texas_env_adapter import TexasHoldemAdapter
from layer3_solvers import HAPPOConfig, HAPPOSolver, MAACConfig, MAACSolver, QMixConfig, QMixSolver

GAMES = ("moon_chess", "texas_holdem", "mahjong_2p")
SOLVERS = ("qmix", "happo", "maac")

# 默认每场对抗局数（按本机实测耗时标定，命令行可覆盖）：
#   moon_chess:  ~5-15 ms/局 → 100 局/方向
#   texas_holdem: ~10-50 ms/局 → 100 局/方向
#   mahjong_2p:  ~3.5 s/局 → 30 局/方向（约 10 min）
DEFAULT_GAMES_PER_MATCH: dict[str, int] = {
    "moon_chess": 100,
    "texas_holdem": 100,
    "mahjong_2p": 30,
}

SOLVER_CLASSES = {"qmix": QMixSolver, "happo": HAPPOSolver, "maac": MAACSolver}


def make_config(name: str, seed: int = 0, device: str = "cpu"):
    """按求解器名构造默认配置（device 在加载前解析好）。"""
    if name == "qmix":
        return QMixConfig(seed=seed, device=device)
    if name == "happo":
        return HAPPOConfig(seed=seed, device=device)
    return MAACConfig(seed=seed, device=device)


def make_adapter(game: str, seed: int):
    """构造游戏适配器（mahjong 用 guangdong 变种 2 人）。"""
    if game == "moon_chess":
        return MoonChessAdapter(seed=seed)
    if game == "texas_holdem":
        return TexasHoldemAdapter(seed=seed)
    return MahjongAdapter(variant="guangdong", player_count=2, seed=seed)


def play_game(adapter, owners: dict[str, str], rng: random.Random, max_steps: int = 2000) -> dict:
    """轻量对局：按 owner 把当前玩家分派给对应求解器的 ``select_action``。

    返回 {steps, payoffs, winner}；payoffs 为 None 表示对局中断（平局兜底）。
    """
    state = adapter.create_initial_state()
    steps = 0
    while True:
        if steps >= max_steps:
            return {"steps": steps, "payoffs": None, "winner": None}
        steps += 1
        node = adapter.get_node_type(state)
        if node == "chance":
            outcomes = adapter.get_chance_outcomes(state)
            if not outcomes:
                return {"steps": steps, "payoffs": None, "winner": None}
            probs = [float(getattr(o, "probability", 0.0) or 0.0) for o in outcomes]
            if sum(probs) <= 0:
                state = adapter.apply_chance(state, rng.choice(outcomes))
            else:
                state = adapter.apply_chance(state, rng.choices(outcomes, weights=probs, k=1)[0])
            continue
        if node != "player":
            break
        current = adapter.get_current_player(state)
        if current is None or current not in owners:
            break
        legal = adapter.get_legal_actions(state)
        if not legal:
            break
        solver = owners[current]
        action = solver.select_action(state)
        if action is None:
            break
        try:
            state = adapter.apply_action(state, action)
        except Exception:
            # 规则/引擎边缘情况（如麻将退化 chi 链）→ 按平局处理
            return {"steps": steps, "payoffs": None, "winner": None}
        if adapter.is_terminal(state):
            break

    payoffs = {p: float(adapter.get_utility(state, p)) for p in owners}
    winner = max(payoffs, key=payoffs.get)
    if payoffs[winner] <= 0:
        winner = None  # 双方收益均非正 → 平局
    elif any(v == payoffs[winner] and p != winner for p, v in payoffs.items()):
        winner = None  # 收益并列 → 平局
    return {"steps": steps, "payoffs": payoffs, "winner": winner}


def run_match(game: str, a: str, b: str, a_first: bool, n_games: int, base_seed: int, model_dir: Path) -> dict:
    """(A, B) 一轮对抗：``a_first=True`` 时 A 执先手，否则 B 执先手。

    每局新建 adapter（种子推进），求解器实例只建一次，逐局替换其
    ``adapter`` 引用 —— 编码器/动作空间由同类 adapter 构建，对局中仅依赖
    state，替换安全。
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter0 = make_adapter(game, 0)
    solver_a = SOLVER_CLASSES[a](adapter0, make_config(a, device=device))
    solver_b = SOLVER_CLASSES[b](adapter0, make_config(b, device=device))
    solver_a.load(str(model_dir / game / f"{a}.pt"))
    solver_b.load(str(model_dir / game / f"{b}.pt"))

    # 先手玩家：moon_chess p_black / texas p_sb / mahjong p0
    first = "p_black" if game == "moon_chess" else ("p_sb" if game == "texas_holdem" else "p0")
    order = [first, "p_white" if game == "moon_chess" else ("p_bb" if game == "texas_holdem" else "p1")]

    games: list[dict] = []
    for i in range(n_games):
        seed = base_seed + i
        adapter = make_adapter(game, seed)
        solver_a.adapter = adapter
        solver_b.adapter = adapter
        owners = {order[0]: solver_a if a_first else solver_b, order[1]: solver_b if a_first else solver_a}
        rng = random.Random(seed * 31 + 7)
        rec = play_game(adapter, owners, rng)
        winner = rec["winner"]
        games.append(
            {
                "seed": seed,
                "steps": rec["steps"],
                "payoff_a": rec["payoffs"][order[0]] if rec["payoffs"] else None,
                "payoff_b": rec["payoffs"][order[1]] if rec["payoffs"] else None,
                "winner": ("a" if winner == order[0] else "b") if winner else None,
            }
        )

    a_wins = sum(1 for g in games if g["winner"] == "a")
    b_wins = sum(1 for g in games if g["winner"] == "b")
    draws = sum(1 for g in games if g["winner"] is None)
    pa = [g["payoff_a"] for g in games if g["payoff_a"] is not None]
    pb = [g["payoff_b"] for g in games if g["payoff_b"] is not None]
    return {
        "solver_a": a,
        "solver_b": b,
        "a_first": a_first,
        "games": games,
        "summary": {
            "n_games": n_games,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "draws": draws,
            "a_win_rate": a_wins / n_games,
            "b_win_rate": b_wins / n_games,
            "avg_steps": sum(g["steps"] for g in games) / n_games,
            "avg_payoff_a": sum(pa) / len(pa) if pa else 0.0,
            "avg_payoff_b": sum(pb) / len(pb) if pb else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gavis MARL 单循环赛")
    parser.add_argument("--game", type=str, default="all", choices=["all", *GAMES])
    parser.add_argument("--games-per-match", type=int, default=0, help="覆盖每方向局数（0=按游戏默认）")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--model-dir", type=str, default="models/marl")
    parser.add_argument("--out-dir", type=str, default="data/marl_tournament")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    games = GAMES if args.game == "all" else [args.game]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    pairs = list(itertools.combinations(SOLVERS, 2))
    print(f"\n{'█' * 60}")
    print(f"  Gavis MARL 单循环赛  games={games}")
    print(f"  对阵: {pairs}  × 主/客场")
    print(f"{'█' * 60}")

    for game in games:
        n = args.games_per_match or DEFAULT_GAMES_PER_MATCH[game]
        all_matchups = []
        t0 = time.time()
        for a, b in pairs:
            for a_first in (True, False):
                tag = "A 执先" if a_first else "B 执先"
                print(f"\n── {game}: {a} vs {b}（{tag}，{n} 局）──", flush=True)
                m = run_match(game, a, b, a_first, n, args.seed, model_dir)
                s = m["summary"]
                print(
                    f"  {a} {s['a_wins']:3d} 胜 / {s['b_wins']:3d} 胜 / {s['draws']:3d} 平"
                    f"  avg_payoff {s['avg_payoff_a']:+.2f} / {s['avg_payoff_b']:+.2f}"
                    f"  avg_steps {s['avg_steps']:.1f}"
                )
                all_matchups.append(m)

        # 汇总本游戏：合并主客场后的纯胜负（去掉先手影响）
        agg = {}
        for m in all_matchups:
            key = tuple(sorted((m["solver_a"], m["solver_b"])))
            agg.setdefault(key, {"a_wins": 0, "b_wins": 0, "draws": 0, "n": 0})
            e = agg[key]
            s = m["summary"]
            if m["a_first"]:
                e["a_wins"] += s["a_wins"]
                e["b_wins"] += s["b_wins"]
            else:
                e["a_wins"] += s["b_wins"]
                e["b_wins"] += s["a_wins"]
            e["draws"] += s["draws"]
            e["n"] += s["n_games"]
        agg_out = {
            f"{a}_{b}": {"a": a, "b": b, **v, "a_win_rate": v["a_wins"] / max(1, v["n"])} for (a, b), v in agg.items()
        }

        doc = {
            "game": game,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "games_per_direction": n,
            "matchups": all_matchups,
            "aggregate": agg_out,
        }
        out_path = out_root / f"{game}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        print(f"\n  → {out_path}  (耗时 {time.time() - t0:.0f}s)")

    print(f"\n{'█' * 60}\n  循环赛完成 → {out_root}\n{'█' * 60}")


if __name__ == "__main__":
    main()
