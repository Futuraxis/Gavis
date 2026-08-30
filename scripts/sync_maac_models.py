#!/usr/bin/env python3
"""scripts/sync_maac_models.py — copy a trained MAAC checkpoint to sibling variants.

Mahjong variants share the observation/action space within two tile
groups, so one trained model serves both:

- **136 tiles (34 kinds)**: mahjong_guangdong / mahjong_hongzhong /
  mahjong_taiwan  ← source `models/train/mahjong_guangdong/maac.pt`
- **108 tiles (27 kinds)**: mahjong_sichuan / mahjong_blood /
  mahjong_changsha ← source `models/train/mahjong_sichuan/maac.pt`

The platform's default mahjong AI resolves ``maac`` per game id (the
registry injects ``models/train/<game_id>/maac.pt``); copying the
checkpoint makes every variant in the group use the trained policy
instead of falling back to the heuristic.  Only same-tile-set copies are
performed — cross-group copies would load with a shape mismatch and are
refused by construction (the groups above never mix).

Usage::

    python scripts/sync_maac_models.py             # copy whatever exists
    python scripts/sync_maac_models.py --dry-run   # show what would happen
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_GROUPS = {
    "mahjong_guangdong": ("mahjong_hongzhong", "mahjong_taiwan"),
    "mahjong_sichuan": ("mahjong_blood", "mahjong_changsha"),
}

_MODELS_DIR = _ROOT / "models" / "train"


def main() -> None:
    parser = argparse.ArgumentParser(description="同步 MAAC 检查点到同牌组变体")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    done = missing = 0
    for source, targets in _GROUPS.items():
        src = _MODELS_DIR / source / "maac.pt"
        if not src.exists():
            print(f"[缺失] {source}: 无 {src} — 跳过该组")
            missing += 1
            continue
        for target in targets:
            dst = _MODELS_DIR / target / "maac.pt"
            if args.dry_run:
                print(f"[dry] {src} -> {dst}")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            print(f"[复制] {src} -> {dst}")
            done += 1
    print(f"完成: copied={done} missing_groups={missing}")


if __name__ == "__main__":
    main()