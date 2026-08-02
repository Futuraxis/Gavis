#!/usr/bin/env python3
"""Texas Hold'em Demo — Hybrid (opponent-model search) vs random one hand.

Usage:  python -m demos.demo_texas_holdem [--budget N] [--seed N]
              [--solver mcts|hybrid] [--cfr-iters N]
"""

from __future__ import annotations

import argparse
import random
import time

from layer2_engine.games.texas_holdem import TexasHoldemAdapter
from layer3_solvers import HybridConfig, HybridSolver
from layer3_solvers.base import SolverConfig
from layer3_solvers.mcts import MCTS

SUITS = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
RANKS = {'T': '10', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A'}
STREETS = ['翻前', '翻牌', '转牌', '河牌']
ACTION_NAMES = {'fold': '弃牌', 'call': '跟注', 'check': '过牌', 'raise': '加注'}


def card_str(card_id: str) -> str:
    rank = RANKS.get(card_id[1], card_id[1])
    suit = SUITS.get(card_id[0], '?')
    color = '\033[91m' if card_id[0] in 'hd' else '\033[0m'
    return f'{color}{rank}{suit}\033[0m'


def render_cards(cards: list) -> str:
    return ' '.join(card_str(c) for c in cards) if cards else '—'


def play_one_hand(engine, solver, verbose=True):
    rng = random.Random(0)
    state = engine.create_initial_state()
    state = engine.resolve_chance(state)

    if verbose:
        print('═' * 56)
        print(f'德州扑克 · {solver.name} vs 随机 — 盲注 1/2, 筹码 100')
        print('═' * 56)
        print(f'你的底牌(SB): {render_cards(state["_arrays"]["sb_hole"])}'
              f'   AI底牌(BB): {render_cards(state["_arrays"]["bb_hole"])}')

    moves = 0
    while not engine.is_terminal(state):
        nt = engine.get_node_type(state)
        if nt == 'chance':
            before = len(state['_arrays']['community'])
            state = engine.resolve_chance(state)
            after = len(state['_arrays']['community'])
            if verbose and after > before:
                print(f'\n  {STREETS[state["env"]["street"]]}公共牌: '
                      f'{render_cards(state["_arrays"]["community"])}')
            continue
        if nt != 'player':
            break

        current = engine.get_current_player(state)
        is_ai = current == 'p_bb'
        if is_ai:
            t0 = time.perf_counter()
            action = solver.select_action(state)
            elapsed = time.perf_counter() - t0
            if action is None:
                break
        else:
            actions = engine.get_legal_actions(state)
            action = rng.choice(actions)
            elapsed = None

        env = state['env']
        state = engine.apply_action(state, action)
        choice = action.params.get('choice', '?')
        amount = action.params.get('amount')
        if verbose:
            who = 'AI' if is_ai else '随机'
            kind = ACTION_NAMES.get(choice, choice)
            if choice == 'raise':
                desc = f'加注到 {amount}'
            elif choice == 'call' and amount == 0:
                desc = '过牌'
            else:
                desc = kind
            timing = f'  [{elapsed*1000:.0f}ms]' if elapsed is not None else ''
            print(f'  {who:4s} {desc:10s} 底池 {env["sb_committed"] + env["bb_committed"]:3d}{timing}')
        moves += 1

    env = state['env']
    if verbose:
        print()
        print(f'公共牌: {render_cards(state["_arrays"]["community"])}')
        print(f'SB: {render_cards(state["_arrays"]["sb_hole"])}'
              f'   BB: {render_cards(state["_arrays"]["bb_hole"])}')
        winner = env.get('winner')
        if winner == 'p_sb':
            print(f'🏆 SB 获胜  ({engine.get_utility(state, "p_sb"):+.0f})')
        elif winner == 'p_bb':
            print(f'🏆 BB 获胜  ({engine.get_utility(state, "p_bb"):+.0f})')
        else:
            print('🤝 平局')
        print(f'（共 {moves} 手行动, 底池 {env["sb_committed"] + env["bb_committed"]}）')
    return state


def main():
    parser = argparse.ArgumentParser(description='Texas Hold\'em Demo — hybrid/MCTS vs random')
    parser.add_argument('--budget', type=int, default=1500, help='search budget (iterations)')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--solver', type=str, choices=['mcts', 'hybrid'], default='hybrid',
                        help='AI solver (hybrid = opponent-model search over sampled worlds)')
    parser.add_argument('--cfr-iters', type=int, default=0,
                        help='train a CFR prior for the hybrid (takes seconds to minutes); '
                             '0 = no prior, uniform opponent model')
    args = parser.parse_args()

    engine = TexasHoldemAdapter(seed=args.seed)
    if args.solver == 'hybrid':
        solver = HybridSolver(engine, HybridConfig(
            seed=args.seed,
            mode='search',
            imperfect_information=True,
            mcts_budget=args.budget,
            opponent_model='cfr' if args.cfr_iters > 0 else 'uniform',
        ))
        if args.cfr_iters > 0:
            print(f'训练 CFR 先验 ({args.cfr_iters} iters)...')
            solver.train(1, verbose=True)
    else:
        solver = MCTS(engine, SolverConfig(seed=args.seed))
        solver.budget = args.budget
    play_one_hand(engine, solver, verbose=True)


if __name__ == '__main__':
    main()
