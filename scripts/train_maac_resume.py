#!/usr/bin/env python3
"""scripts/train_maac_resume.py — continue MAAC mahjong training from a checkpoint.

``train.py`` 的注册表管线不支持“续训”（每次新建实例）；本脚本复用其装配
（GameSpec → ``build_engine`` → ``make_solver``），在已保存的 ``maac.pt`` 上
``solver.train(episodes)`` 续跑，然后把更新后的检查点写回同一产物路径。

注意：MAAC 的 ``save()`` 不持久化对手池快照——续训会重新走 PFSP warmup（纯
自博弈起步），对延长训练可接受（与完整首次训练的行为一致）。

Usage::

    python scripts/train_maac_resume.py --game mahjong_guangdong --episodes 400
    python scripts/train_maac_resume.py --game mahjong_sichuan --episodes 400 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "train-cli"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from games import GAMES  # noqa: E402
from train import apply_preset, build_engine, make_solver  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="从已保存检查点续训麻将 MAAC")
    parser.add_argument("--game", default="mahjong_guangdong")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=str(_ROOT / "models" / "train"))
    args = parser.parse_args()

    spec = GAMES[args.game]
    engine = build_engine(spec, args.seed)
    pipeline = apply_preset(spec.solvers["maac"], "full")
    out_dir = Path(args.out_dir) / spec.game_id
    out_dir.mkdir(parents=True, exist_ok=True)

    solver = make_solver("maac", engine, pipeline, args.seed, args.device, out_dir)
    ckpt = out_dir / "maac.pt"
    if ckpt.exists():
        solver.load(str(ckpt))
        print(f"已加载检查点: {ckpt}")
    else:
        print(f"[提示] 无检查点: {ckpt} — 从头训练")

    metrics = solver.train(episodes=args.episodes, verbose=True)
    solver.save(str(ckpt))
    print(f"已保存: {ckpt}")
    print(f"metrics: episodes={metrics.episodes} win_rate={metrics.win_rate:.4f} "
          f"avg_return={metrics.avg_return:.4f} steps={metrics.extra.get('steps')}")


if __name__ == "__main__":
    main()