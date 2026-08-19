#!/usr/bin/env python3
"""MARL 循环赛结果分析 — 汇总胜负矩阵、先手优势、平局率、对局长度等指标.

读取 ``<out-dir>/<game>.json``（循环赛产物）与 ``<model-dir>/<game>/metrics.json``
（训练指标），生成 Markdown 分析报告 ``<out-dir>/REPORT.md`` 并打印控制台摘要。

Usage:  python -m demos.marl_report [--data-dir data/marl_tournament]
                                     [--model-dir models/marl]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SOLVERS = ("qmix", "happo", "maac")
SOLVER_LABEL = {"qmix": "QMix", "happo": "HAPPO", "maac": "MAAC"}


def load_tournament(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_metrics(game_dir: Path) -> dict:
    """训练指标 {solver: {win_rate, avg_return, elapsed_s, ...}}。"""
    path = game_dir / "metrics.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def match_win_matrix(doc: dict) -> dict:
    """合并主客场后的 {pair: {a_win, b_win, draw, n}}（a 恒为 pair 前一个）。

    无论谁执先，solver_a 的胜场恒为 ``summary.a_wins``；按 pair 排序后
    对齐到 e["a"] / e["b"] 即可。
    """
    agg = {}
    for m in doc["matchups"]:
        a, b = m["solver_a"], m["solver_b"]
        key = tuple(sorted((a, b)))
        e = agg.setdefault(key, {"a": key[0], "b": key[1], "a_wins": 0, "b_wins": 0, "draws": 0, "n": 0})
        s = m["summary"]
        if a == e["a"]:
            e["a_wins"] += s["a_wins"]
            e["b_wins"] += s["b_wins"]
        else:
            e["a_wins"] += s["b_wins"]
            e["b_wins"] += s["a_wins"]
        e["draws"] += s["draws"]
        e["n"] += s["n_games"]
    return agg


def first_player_stats(doc: dict) -> list[dict]:
    """每个方向记录的（先手方胜率、后手方胜率、平局率）。"""
    out = []
    for m in doc["matchups"]:
        s = m["summary"]
        first_wins = s["a_wins"] if m["a_first"] else s["b_wins"]
        second_wins = s["b_wins"] if m["a_first"] else s["a_wins"]
        out.append(
            {
                "match": f"{SOLVER_LABEL[m['solver_a']]} vs {SOLVER_LABEL[m['solver_b']]}",
                "first_side": SOLVER_LABEL[m["solver_a"]] if m["a_first"] else SOLVER_LABEL[m["solver_b"]],
                "first_wins": first_wins,
                "second_wins": second_wins,
                "draws": s["draws"],
                "n": s["n_games"],
            }
        )
    return out


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gavis MARL 循环赛结果分析")
    parser.add_argument("--data-dir", type=str, default="data/marl_tournament")
    parser.add_argument("--model-dir", type=str, default="models/marl")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    games = sorted(p.stem for p in data_dir.glob("*.json") if p.stem != "REPORT")

    lines: list[str] = []
    lines.append("# MARL 单循环赛分析报告")
    lines.append("")
    lines.append(f"- 数据目录: `{data_dir}`  模型目录: `{model_dir}`")
    lines.append("- 参赛求解器: QMix / HAPPO / MAAC（每对主客场两轮）")
    lines.append("")

    # ── 训练指标 ──────────────────────────────────────────────────
    lines.append("## 1. 训练指标")
    lines.append("")
    lines.append("| 游戏 | 求解器 | 局数 | 训练胜率 | 平均收益 | 训练耗时 |")
    lines.append("|------|--------|------|----------|----------|----------|")
    for game in games:
        metrics = load_metrics(model_dir / game)
        for name in SOLVERS:
            m = metrics.get(name)
            if not m:
                continue
            lines.append(
                f"| {game} | {SOLVER_LABEL[name]} | {m['episodes']} | "
                f"{fmt_pct(m['win_rate'])} | {m['avg_return']:+.3f} | {m['elapsed_s']:.0f}s |"
            )
    lines.append("")

    # ── 每游戏胜负矩阵 ────────────────────────────────────────────
    overall = {s: {"wins": 0, "losses": 0, "draws": 0, "n": 0} for s in SOLVERS}
    lines.append("## 2. 各游戏胜负矩阵（主客场合并）")
    lines.append("")
    for game in games:
        doc = load_tournament(data_dir / f"{game}.json")
        agg = match_win_matrix(doc)
        lines.append(f"### {game}")
        lines.append("")
        lines.append("| 求解器 A | 求解器 B | A 胜 | B 胜 | 平 | A 胜率 | 平均步数 |")
        lines.append("|----------|----------|------|------|-----|--------|----------|")
        for key in sorted(agg):
            e = agg[key]
            a_wr = e["a_wins"] / max(1, e["n"])
            lines.append(
                f"| {SOLVER_LABEL[e['a']]} | {SOLVER_LABEL[e['b']]} | {e['a_wins']} | {e['b_wins']} "
                f"| {e['draws']} | {fmt_pct(a_wr)} | — |"
            )
        lines.append("")
        # 累计到总榜
        for key, e in agg.items():
            overall[e["a"]]["wins"] += e["a_wins"]
            overall[e["a"]]["losses"] += e["b_wins"]
            overall[e["a"]]["draws"] += e["draws"]
            overall[e["a"]]["n"] += e["n"]
            overall[e["b"]]["wins"] += e["b_wins"]
            overall[e["b"]]["losses"] += e["a_wins"]
            overall[e["b"]]["draws"] += e["draws"]
            overall[e["b"]]["n"] += e["n"]

    # ── 先手优势 ──────────────────────────────────────────────────
    lines.append("## 3. 先手优势")
    lines.append("")
    lines.append("| 游戏 | 对阵 | 先手方 | 先手胜率 | 后手胜率 | 平局率 |")
    lines.append("|------|------|--------|----------|----------|--------|")
    for game in games:
        doc = load_tournament(data_dir / f"{game}.json")
        for st in first_player_stats(doc):
            lines.append(
                f"| {game} | {st['match']} | {st['first_side']} | "
                f"{fmt_pct(st['first_wins'] / st['n'])} | {fmt_pct(st['second_wins'] / st['n'])} | "
                f"{fmt_pct(st['draws'] / st['n'])} |"
            )
    lines.append("")

    # ── 总榜 ──────────────────────────────────────────────────────
    lines.append("## 4. 总榜（跨游戏合并）")
    lines.append("")
    lines.append("| 求解器 | 胜 | 负 | 平 | 胜率 |")
    lines.append("|--------|----|----|-----|------|")
    ranked = sorted(SOLVERS, key=lambda s: -overall[s]["wins"] / max(1, overall[s]["n"]))
    for s in ranked:
        o = overall[s]
        lines.append(
            f"| {SOLVER_LABEL[s]} | {o['wins']} | {o['losses']} | {o['draws']} | {fmt_pct(o['wins'] / max(1, o['n']))} |"
        )
    lines.append("")

    report = "\n".join(lines)
    out_path = data_dir / "REPORT.md"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n报告 → {out_path}")


if __name__ == "__main__":
    main()
