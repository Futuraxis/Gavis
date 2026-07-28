#!/usr/bin/env python3
"""CFR Demo — 在随机五子棋上训练 CFR 并评估策略质量

用法:  python demo_cfr.py [--iters N] [--size N] [--games N]
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gavis.core.engine import GameEngine
from gavis.core.state_graph import clone_state
from gavis.solvers.cfr import CFR, estimate_exploitability


# ---------------------------------------------------------------------------
# Board display
# ---------------------------------------------------------------------------

SYMBOLS = {'black': '●', 'white': '○', None: '·'}


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='CFR — 随机五子棋训练与评估')
    parser.add_argument('--iters', type=int, default=1000, help='CFR 迭代次数')
    parser.add_argument('--size', type=int, default=5, help='棋盘大小 (默认 5，CFR 适合小板)')
    parser.add_argument('--games', type=int, default=200, help='评估对局数')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-cfr-plus', action='store_true', help='禁用 CFR+')
    args = parser.parse_args()

    # Load rules
    rules_path = Path(__file__).resolve().parent / 'gavis' / 'games' / 'stochastic_gomoku' / 'rules.json'
    with open(rules_path, 'r', encoding='utf-8') as f:
        rules = json.load(f)
    rules['constants']['board_size'] = args.size

    engine = GameEngine(rules, seed=args.seed)
    state = engine.create_initial_state()

    print(f'{"="*60}')
    print(f'CFR 训练 — 随机五子棋 {args.size}×{args.size}  消失率=50%')
    print(f'迭代: {args.iters}  CFR+: {not args.no_cfr_plus}')
    print(f'{"="*60}')

    # ------------------------------------------------------------------
    # Train CFR
    # ------------------------------------------------------------------
    cfr = CFR(
        engine,
        iterations=args.iters,
        use_cfr_plus=not args.no_cfr_plus,
        seed=args.seed,
    )

    print('\n训练中...\n')
    t0 = time.perf_counter()
    strategy = cfr.solve(state, verbose=True)
    elapsed = time.perf_counter() - t0

    print(f'\n训练完成: {elapsed:.1f}s  info_sets={len(cfr.info_sets)}')

    # ------------------------------------------------------------------
    # Show top actions at root
    # ------------------------------------------------------------------
    actions = engine.get_legal_actions(state)
    print(f'\n根节点策略 (当前玩家={engine.get_current_player(state)}):')
    print(f'{"─"*50}')
    sorted_strat = sorted(strategy.items(), key=lambda kv: kv[1], reverse=True)
    for key, prob in sorted_strat[:10]:
        bar = '█' * int(prob * 40)
        print(f'  {key:20s}  {prob:.4f}  {bar}')
    if len(sorted_strat) > 10:
        print(f'  ... ({len(sorted_strat) - 10} more actions)')

    # ------------------------------------------------------------------
    # Evaluate vs random baseline
    # ------------------------------------------------------------------
    print(f'\n{"="*60}')
    print(f'评估: CFR (黑方) vs Random (白方), {args.games} 局')
    print(f'{"="*60}')

    results = estimate_exploitability(engine, cfr, state, n_games=args.games, seed=args.seed)
    print(f'\n  黑方 (CFR)    胜: {results["cfr_wins"]:3d}  ({results["cfr_wins"]/args.games*100:.1f}%)')
    print(f'  白方 (Random) 胜: {results["cfr_losses"]:3d}  ({results["cfr_losses"]/args.games*100:.1f}%)')
    print(f'  平局:            {results["draws"]:3d}  ({results["draws"]/args.games*100:.1f}%)')

    # ------------------------------------------------------------------
    # Play one demo game
    # ------------------------------------------------------------------
    print(f'\n{"="*60}')
    print(f'演示对局: CFR vs Random')
    print(f'{"="*60}')

    import random as rand_mod
    demo_rng = rand_mod.Random(args.seed + 9999)

    sim_state = clone_state(state)
    move = 0
    while not engine.is_terminal(sim_state):
        nt = engine.get_node_type(sim_state)

        if nt == 'player':
            move += 1
            current = engine.get_current_player(sim_state)
            actions = engine.get_legal_actions(sim_state)

            if current == 'p_black':
                action = cfr.get_action(sim_state)
                label = 'CFR'
            else:
                action = demo_rng.choice(actions) if actions else None
                label = 'RND'

            if action is None:
                break
            sim_state = engine.apply_action(sim_state, action)

            # Extract coordinates
            cell = action.params.get('cell', {})
            if isinstance(cell, dict):
                x, y = cell['props']['x'], cell['props']['y']
            else:
                x, y = '?', '?'
            print(f'  Move {move:2d}: {label:4s} → ({x},{y})')

        elif nt == 'chance':
            outcome, sim_state = engine.sample_chance(sim_state)
            if outcome.key == 'vanish':
                print(f'         🎲 消失!')
            # else: print nothing for keep

        else:
            break

    print()
    print(render_board(sim_state))
    winner = sim_state['env'].get('winner')
    if winner:
        print(f'\n🏆 胜者: {"CFR (黑方)" if winner == "p_black" else "Random (白方)"}')
    else:
        print('\n🤝 平局')


if __name__ == '__main__':
    main()
