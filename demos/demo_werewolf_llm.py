#!/usr/bin/env python3
"""狼人杀 LLM 自对弈演示 — 9 名玩家全部由本地 ollama (qwen3:8b) 扮演.

Usage:  python -m demos.demo_werewolf_llm [--players 9] [--seed N] [--max-steps 300]
                                          [--model qwen3:8b] [--verbose]

每步 LLM 调用 6-10s（热态），一局约 5-8 分钟；``--verbose`` 打印全部发言。
"""

from __future__ import annotations

import argparse
import random
from dataclasses import replace
from pathlib import Path

from layer2_engine.games.werewolf.werewolf_adapter import WerewolfAdapter
from layer3_solvers import OllamaConfig, OllamaSolver


def main() -> None:
    parser = argparse.ArgumentParser(description='Werewolf LLM self-play demo')
    parser.add_argument('--players', type=int, default=9)
    parser.add_argument('--wolves', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--model', type=str, default='qwen3:8b')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    adapter = WerewolfAdapter(seed=args.seed, players=args.players, wolves=args.wolves)
    pids = adapter._constants['player_ids']
    rng = random.Random(args.seed)

    solvers = {
        pid: OllamaSolver(adapter, OllamaConfig(model=args.model), player_id=pid)
        for pid in pids
    }

    state = adapter.create_initial_state()
    print(f'{"█" * 60}')
    print(f'  狼人杀 LLM 自对弈  players={args.players}  model={args.model}')
    print(f'{"█" * 60}')

    role_names = {pid: None for pid in pids}
    steps = 0
    while True:
        steps += 1
        if steps > args.max_steps:
            print(f'步数上限 {args.max_steps} 到达，中断')
            break
        nt = adapter.get_node_type(state)
        phase = state['env']['phase']
        if nt == 'chance':
            outs = adapter.get_chance_outcomes(state)
            if not outs:
                break
            state = adapter.apply_chance(state, rng.choices(outs, weights=[o.probability for o in outs], k=1)[0])
            if phase.startswith('deal_'):
                # 发牌后补记角色
                arr = state['_arrays'].get('roles', [])
                for i, pid in enumerate(pids):
                    if i < len(arr) and role_names[pid] is None:
                        role_names[pid] = arr[i]
            continue
        if nt != 'player':
            break
        cur = adapter.get_current_player(state)
        if cur is None or cur not in solvers:
            break
        legal = adapter.get_legal_actions(state)
        if not legal:
            break
        print(f'  [步 {steps}] {cur} ({role_names[cur]}) 阶段={phase} … ', end='', flush=True)
        action = solvers[cur].select_action(state)
        if action is None:
            action = rng.choice(legal)
        if action.template_id == 'speak':
            print(f'发言[{action.params.get("intent", {}).get("id")}]: {action.params.get("text", "")}')
        else:
            print(action.canonical_key)
        state = adapter.apply_action(state, action)
        if adapter.is_terminal(state):
            break

    winner = state['env'].get('winner')
    print(f'\n{"█" * 60}')
    print(f'  终局: winner={winner}  round={state["env"].get("round")}  steps={steps}')
    print(f'  角色: {role_names}')
    print(f'  存活: {[p for i, p in enumerate(pids) if state["_arrays"]["alive"][i] == 1]}')
    print(f'{"█" * 60}')


if __name__ == '__main__':
    main()
