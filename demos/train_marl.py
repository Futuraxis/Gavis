#!/usr/bin/env python3
"""MARL 求解器训练 — 在三个游戏上训练 QMix / HAPPO / MAAC 并保存产物.

三个游戏共享同一个 adapter 实例（同一发牌序列），保证各求解器在完全
相同的对局分布上训练，横评公平。

产物保存到 ``<out-dir>/<game>/``：
  - ``{qmix,happo,maac}.pt``  各求解器 checkpoint
  - ``metrics.json``          每个求解器的训练指标（win_rate / avg_return / 耗时）
  - ``config.json``           训练所用全部配置

Usage:  python -m demos.train_marl [--game all|moon_chess|texas_holdem|mahjong_2p]
                                    [--solver all|qmix|happo|maac]
                                    [--episodes N] [--seed N] [--device cpu|cuda|auto]
                                    [--out-dir models/marl] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from layer2_engine.games.mahjong.mahjong_adapter import MahjongAdapter
from layer2_engine.games.moon_chess.moon_env_adapter import MoonChessAdapter
from layer2_engine.games.texas_holdem.texas_env_adapter import TexasHoldemAdapter

from layer3_solvers import HAPPOConfig, HAPPOSolver, MAACConfig, MAACSolver, QMixConfig, QMixSolver

GAMES = ("moon_chess", "texas_holdem", "mahjong_2p")
SOLVERS = ("qmix", "happo", "maac")

# 默认训练局数（按本机实测耗时标定，命令行可覆盖）：
#   moon_chess:  ~5-15 ms/局   600 局 ≈ 10 s
#   texas_holdem: ~10-50 ms/局 400 局 ≈ 30 s
#   mahjong_2p:  ~3.5 s/局（随机对局 220 步打满墙） 200 局 ≈ 12 min
DEFAULT_EPISODES: dict[str, int] = {
    "moon_chess": 600,
    "texas_holdem": 400,
    "mahjong_2p": 200,
}

SOLVER_CLASSES = {
    "qmix": (QMixSolver, QMixConfig),
    "happo": (HAPPOSolver, HAPPOConfig),
    "maac": (MAACSolver, MAACConfig),
}


def make_adapter(game: str, seed: int):
    """构造游戏适配器（mahjong 用 guangdong 变种 2 人）。"""
    if game == "moon_chess":
        return MoonChessAdapter(seed=seed)
    if game == "texas_holdem":
        return TexasHoldemAdapter(seed=seed)
    return MahjongAdapter(variant="guangdong", player_count=2, seed=seed)


def resolve_device(device: str) -> str:
    """'auto' → cuda 可用时用 cuda，否则 cpu。"""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gavis MARL 求解器多游戏训练")
    parser.add_argument("--game", type=str, default="all", choices=["all", *GAMES])
    parser.add_argument("--solver", type=str, default="all", choices=["all", *SOLVERS])
    parser.add_argument("--episodes", type=int, default=0, help="覆盖默认局数（0=按游戏默认）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-dir", type=str, default="models/marl")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    games = GAMES if args.game == "all" else [args.game]
    solvers = SOLVERS if args.solver == "all" else [args.solver]
    out_root = Path(args.out_dir)

    print(f'\n{"█" * 60}')
    print(f"  Gavis MARL 训练  games={games}  solvers={solvers}  device={device}")
    print(f'{"█" * 60}')

    for game in games:
        episodes = args.episodes or DEFAULT_EPISODES[game]
        # 共享同一 adapter：三个求解器在同一发牌序列上训练
        adapter = make_adapter(game, args.seed)
        game_dir = out_root / game
        game_dir.mkdir(parents=True, exist_ok=True)

        metrics_all: dict[str, dict] = {}
        for name in solvers:
            cls, cfg_cls = SOLVER_CLASSES[name]
            cfg = cfg_cls(seed=args.seed, device=device)
            solver = cls(adapter, cfg)
            print(f'\n── 训练 {name} @ {game}  ({episodes} 局)  ──')
            t0 = time.time()
            metrics = solver.train(episodes=episodes, verbose=args.verbose)
            elapsed = time.time() - t0
            metrics_all[name] = {
                "episodes": metrics.episodes,
                "win_rate": metrics.win_rate,
                "avg_return": metrics.avg_return,
                "extra": metrics.extra,
                "elapsed_s": round(elapsed, 1),
                "config": asdict(cfg),
            }
            solver.save(str(game_dir / f"{name}.pt"))
            print(f"  完成: win_rate={metrics.win_rate:.3f}  avg_return={metrics.avg_return:.3f}"
                  f"  {elapsed:.1f}s  → {game_dir / (name + '.pt')}")

        # 合并写入：并行训练多个求解器时各自的 metrics.json 不会互相覆盖
        prev_path = game_dir / "metrics.json"
        merged = metrics_all
        if prev_path.exists():
            with open(prev_path, encoding="utf-8") as f:
                merged = {**json.load(f), **metrics_all}
        with open(prev_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        with open(game_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump({"seed": args.seed, "device": device}, f, ensure_ascii=False, indent=2)
        print(f"  指标 → {game_dir / 'metrics.json'}")

    print(f'\n{"█" * 60}\n  训练完成 → {out_root}\n{"█" * 60}')


if __name__ == "__main__":
    main()
