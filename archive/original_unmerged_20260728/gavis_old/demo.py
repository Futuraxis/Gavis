#!/usr/bin/env python3
"""Stochastic Gomoku — 随机五子棋 最小 Demo

游戏规则: 经典五子棋，但落子后有 50% 概率棋子消失。
这引入了 chance node，可以验证 MCTS 对随机博弈的处理能力。

用法:  python demo.py [--games N] [--budget N] [--size N]
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gavis.core.engine import GameEngine
from gavis.core.state_graph import cell_xy
from gavis.solvers.mcts import MCTS


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

SYMBOLS = {
    'black': '●',
    'white': '○',
    None: '·',
}

COLOR_NAMES = {
    'p_black': '黑方 ●',
    'p_white': '白方 ○',
}


def render_board(state: dict) -> str:
    """Render the board as ASCII art."""
    bs = state['board_size']
    board = state['_board']
    lines = []
    # Column header
    header = '   ' + ''.join(f'{i:2}' for i in range(bs))
    lines.append(header)
    for y in range(bs):
        row = f'{y:2} '
        for x in range(bs):
            idx = y * bs + x
            occupant = board[idx]
            row += ' ' + SYMBOLS.get(occupant, '?')
        lines.append(row)
    return '\n'.join(lines)


def render_move_info(action) -> str:
    """Render move information from action params (may be node dicts)."""
    cell_param = action.params.get('cell', {})
    if isinstance(cell_param, dict):
        x = cell_param.get('props', {}).get('x', '?')
        y = cell_param.get('props', {}).get('y', '?')
        return f'落子 ({x},{y})'
    return f'{action.canonical_key}'


# ---------------------------------------------------------------------------
# Single game player
# ---------------------------------------------------------------------------

def play_one_game(engine: GameEngine, mcts: MCTS, verbose: bool = True) -> dict:
    """Play a single game with MCTS for both sides. Returns game stats."""
    state = engine.create_initial_state()
    move_count = 0
    history = []

    if verbose:
        print('═' * 50)
        print('随机五子棋 — 对局开始')
        print(f'棋盘: {state["board_size"]}×{state["board_size"]}  消失概率: 50%  MCTS 预算: {mcts.budget}')
        print('═' * 50)

    while not engine.is_terminal(state):
        node_type = engine.get_node_type(state)

        if node_type == 'player':
            move_count += 1
            current = engine.get_current_player(state)
            if verbose:
                print(f'\n── Move {move_count} | {COLOR_NAMES.get(current, current)} ──')

            # MCTS search
            t0 = time.perf_counter()
            action = mcts.search(state)
            elapsed = time.perf_counter() - t0

            if action is None:
                if verbose:
                    print('  (无合法动作)')
                break

            # Get stats
            stats = mcts.action_stats(state)
            top_visits = stats[0][1] if stats else 0
            top_value = stats[0][2] if stats else 0.0

            # Apply action
            state = engine.apply_action(state, action)

            # Extract cell coordinates (params now stores full node dicts)
            cell_param = action.params.get('cell', {})
            if isinstance(cell_param, dict):
                x = cell_param.get('props', {}).get('x', '?')
                y = cell_param.get('props', {}).get('y', '?')
            else:
                # Fallback for string cell id
                parts = str(cell_param).split('_')
                x, y = (parts[1], parts[2]) if len(parts) >= 3 else ('?', '?')

            history.append({
                'move': move_count,
                'player': current,
                'cell': f'{x},{y}',
                'canonical': action.canonical_key,
                'mcts_visits': top_visits,
                'mcts_value': top_value,
                'mcts_time_ms': round(elapsed * 1000),
            })

            if verbose:
                print(f'  MCTS 选择 {top_visits} visits, val={top_value:+.3f}  [{elapsed*1000:.0f}ms]')

        elif node_type == 'chance':
            outcome, state = engine.sample_chance(state)
            last_action = state['env'].get('lastAction', '?')

            if outcome.key == 'vanish':
                if verbose:
                    last_cell = state['env'].get('lastPlacedCell', '?')
                    # last_cell is a string like "cell_3_5"
                    if isinstance(last_cell, str) and '_' in last_cell:
                        parts = last_cell.split('_')
                        xy = f'({parts[-2]},{parts[-1]})' if len(parts) >= 3 else '?'
                    else:
                        xy = str(last_cell)
                    print(f'  🎲 棋子消失! {xy} 上的棋子蒸发了…')

                # Record vanish
                if history:
                    history[-1]['vanished'] = True
            else:
                if verbose:
                    print(f'  🎲 棋子保留 ✓')

                if history:
                    history[-1]['vanished'] = False

        else:
            break

    # Game over
    if verbose:
        print()
        print(render_board(state))
        print()

    winner = state['env'].get('winner')
    if verbose:
        if winner:
            print(f'🏆 胜者: {COLOR_NAMES.get(winner, winner)}')
        else:
            print('🤝 平局')

    return {
        'winner': winner,
        'move_count': move_count,
        'final_state': state,
        'history': history,
    }


# ---------------------------------------------------------------------------
# Tournament mode
# ---------------------------------------------------------------------------

def run_tournament(engine: GameEngine, mcts: MCTS, n_games: int = 10):
    """Run multiple games and report statistics."""
    import json

    results = {'p_black': 0, 'p_white': 0, 'draw': 0}
    vanish_count = 0
    keep_count = 0

    print(f'\n{"="*60}')
    print(f'批量对局: {n_games} 局')
    print(f'{"="*60}')

    for i in range(n_games):
        # MCTS with fresh RNG each game
        mcts.rng.seed(i * 12345 + 67890)
        engine.rng.seed(i * 54321 + 9876)

        result = play_one_game(engine, mcts, verbose=False)
        winner = result['winner']
        if winner == 'p_black':
            results['p_black'] += 1
        elif winner == 'p_white':
            results['p_white'] += 1
        else:
            results['draw'] += 1

        # Count vanish/keep
        for h in result['history']:
            if h.get('vanished') is True:
                vanish_count += 1
            elif h.get('vanished') is False:
                keep_count += 1

        print(f'  局 {i+1:2d}: 胜={COLOR_NAMES.get(winner, "平局"):12s}  '
              f'步数={result["move_count"]:2d}  '
              f'MCTS平均={sum(h["mcts_time_ms"] for h in result["history"])//max(1,len(result["history"])):4d}ms/步')

    print(f'\n{"─"*60}')
    print(f'统计: 黑胜={results["p_black"]}  白胜={results["p_white"]}  平={results["draw"]}')
    win_rate = (results['p_black'] + results['p_white']) / n_games * 100
    print(f'胜率 (非平局): {win_rate:.0f}%')
    if vanish_count + keep_count > 0:
        actual_vanish = vanish_count / (vanish_count + keep_count) * 100
        print(f'棋子消失率: {actual_vanish:.1f}% (理论 50%)')
    print(f'{"─"*60}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='随机五子棋 — 自适应棋牌 AI Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python demo.py                  # 单局演示
  python demo.py --games 20       # 20局统计
  python demo.py --budget 10000   # 提高 MCTS 预算
  python demo.py --size 7         # 7x7 棋盘
  python demo.py --seed 42        # 固定随机种子
        """,
    )
    parser.add_argument('--games', type=int, default=1, help='对局数量 (默认 1，即单局演示)')
    parser.add_argument('--budget', type=int, default=5000, help='MCTS 每步迭代预算 (默认 5000)')
    parser.add_argument('--size', type=int, default=9, help='棋盘大小 (默认 9)')
    parser.add_argument('--seed', type=int, default=None, help='全局随机种子')
    parser.add_argument('--vanish', type=float, default=0.5, help='棋子消失概率 (默认 0.5)')
    parser.add_argument('--ucb-c', type=float, default=1.414, help='MCTS 探索系数')
    args = parser.parse_args()

    # Load rules
    rules_path = Path(__file__).resolve().parent / 'gavis' / 'games' / 'stochastic_gomoku' / 'rules.json'
    if not rules_path.exists():
        print(f'错误: 找不到规则文件 {rules_path}')
        sys.exit(1)

    # Load + override constants
    import json
    with open(rules_path, 'r', encoding='utf-8') as f:
        rules = json.load(f)
    rules['constants']['board_size'] = args.size
    rules['constants']['vanish_probability'] = args.vanish

    # Update chance probability
    for ct in rules.get('chance', []):
        if ct['id'] == 'vanish':
            ct['probability']['explicit'][0]['probability']['const'] = args.vanish
            ct['probability']['explicit'][1]['probability']['const'] = 1.0 - args.vanish

    engine = GameEngine(rules, seed=args.seed)
    mcts = MCTS(
        engine,
        budget=args.budget,
        ucb_c=args.ucb_c,
        seed=args.seed,
    )

    print(f'Gavis — 随机五子棋 Demo')
    print(f'棋盘: {args.size}×{args.size} | 消失率: {args.vanish} | MCTS预算: {args.budget} | UCB_C: {args.ucb_c}')

    if args.games == 1:
        play_one_game(engine, mcts, verbose=True)
    else:
        run_tournament(engine, mcts, args.games)


if __name__ == '__main__':
    main()
