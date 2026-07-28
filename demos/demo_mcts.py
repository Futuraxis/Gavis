#!/usr/bin/env python3
"""MCTS Demo — plays Stochastic Gomoku using MCTS.

Usage:  python -m demos.demo_mcts [--budget N] [--size N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from layer2_engine.core.engine import GameEngine
from layer3_solvers.mcts import MCTS
from layer3_solvers.base import SolverConfig

SYMBOLS = {'p_black': '●', 'p_white': '○', None: '·'}
COLOR_NAMES = {'p_black': '黑方 ●', 'p_white': '白方 ○'}


def render_board(state: dict) -> str:
    bs = state['board_size']
    board = state['_board']
    lines = ['   ' + ''.join(f'{i:2}' for i in range(bs))]
    for y in range(bs):
        row = f'{y:2} '
        for x in range(bs):
            row += ' ' + SYMBOLS.get(board[y * bs + x], '?')
        lines.append(row)
    return '\n'.join(lines)


def play_one_game(engine, mcts, verbose=True):
    state = engine.create_initial_state()
    move_count = 0

    if verbose:
        print('═' * 50)
        print(f'MCTS 随机五子棋 — 棋盘 {state["board_size"]}×{state["board_size"]}')
        print(f'预算: {mcts.budget}')
        print('═' * 50)

    while not engine.is_terminal(state):
        nt = engine.get_node_type(state)
        if nt == 'player':
            move_count += 1
            current = engine.get_current_player(state)
            if verbose:
                print(f'\n── 第{move_count}手 | {COLOR_NAMES.get(current, current)} ──')
            t0 = time.perf_counter()
            action = mcts.select_action(state)
            elapsed = time.perf_counter() - t0
            if action is None:
                break
            state = engine.apply_action(state, action)
            if verbose:
                print(f'  落子 {action.canonical_key}  [{elapsed*1000:.0f}ms]')
        elif nt == 'chance':
            _, state = engine.sample_chance(state)
        else:
            break

    if verbose:
        print()
        print(render_board(state))
        winner = state['env'].get('winner')
        if winner:
            print(f'\n🏆 胜者: {COLOR_NAMES.get(winner, winner)}')
        else:
            print('\n🤝 平局')
    return state


def main():
    parser = argparse.ArgumentParser(description='MCTS Demo — Stochastic Gomoku')
    parser.add_argument('--budget', type=int, default=5000, help='MCTS budget')
    parser.add_argument('--size', type=int, default=9, help='Board size')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    rules_path = Path(__file__).resolve().parent.parent / 'rules' / 'stochastic_gomoku.json'
    with open(rules_path, 'r', encoding='utf-8') as f:
        rules = json.load(f)
    rules['constants']['board_size'] = args.size

    engine = GameEngine(rules, seed=args.seed)
    mcts = MCTS(engine, SolverConfig(seed=args.seed))
    mcts.budget = args.budget

    play_one_game(engine, mcts, verbose=True)


if __name__ == '__main__':
    main()
