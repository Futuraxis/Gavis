"""Tests for the werewolf play session (human vs AI table).

The AI solvers are stubbed with fixed-action solvers so the session
mechanics (turn rotation, AI driving, snapshots) run without ollama.
"""

from __future__ import annotations

import random

import pytest

from layer2_engine.games.werewolf.werewolf_adapter import WerewolfAdapter
from layer4_interface.frontend.play_werewolf.session import GameSession, PlayManager


class _StubSolver:
    """AI stub bound to the session engine: random legal action."""

    def __init__(self, player_id: str, engine: WerewolfAdapter):
        self.player_id = player_id
        self.engine = engine
        self._rng = random.Random(hash(player_id) & 0xFFFF)

    def select_action(self, state) -> object | None:
        from dataclasses import replace

        legal = self.engine.get_legal_actions(state)
        if not legal:
            return None
        a = self._rng.choice(legal)
        if a.template_id == 'speak':
            a = replace(a, params={**a.params, 'text': f'{self.player_id}说：我观察了很久'})
        return a


def _make_session(seed: int = 3) -> GameSession:
    engine = WerewolfAdapter(seed=seed)
    pids = engine._constants['player_ids']
    human = 'p0'
    ai = {pid: _StubSolver(pid, engine) for pid in pids if pid != human}
    session = GameSession(
        game_id='test1', human_pid=human, engine=engine,
        ai_solvers=ai, model='stub',
    )
    session._resolve_chance()
    session._ai_turns()
    return session


def test_session_start_reaches_human_turn():
    session = _make_session()
    assert session.over or session.my_turn
    snap = session.snapshot()
    assert snap['game_id'] == 'test1'
    assert snap['my_pid'] == 'p0'
    assert snap['my_role'] is not None
    assert len(snap['players']) == 9
    # 只有自己的身份可见
    visible_roles = [p['role'] for p in snap['players']]
    assert sum(1 for r in visible_roles if r is not None) == 1


def test_human_speech_records_text():
    session = None
    for s in range(30):
        cand = _make_session(seed=10 + s)
        if (not cand.over and cand.my_turn
                and cand.state['env']['phase'] == 'day_speech'):
            session = cand
            break
    if session is None:
        pytest.skip('no game reached human speech phase')
    session.human_move('speak', {'intent': 'claim', 'text': '我是预言家，昨晚验到狼'})
    log = session.state['_arrays']['speechLog']
    assert log and log[-1]['speaker'] == 'p0'
    assert log[-1]['text'] == '我是预言家，昨晚验到狼'
    assert log[-1]['intent'] == 'claim'
    # AI 回合推进后应轮到真人或终局
    assert session.over or session.my_turn


def test_snapshot_hides_other_roles_and_shows_dead_roles():
    session = _make_session()
    snap = session.snapshot()
    for p in snap['players']:
        if p['id'] == 'p0':
            assert p['role'] is not None
        else:
            assert p['role'] is None
    # 已死玩家公布身份
    dead = [p for p in snap['players'] if not p['alive']]
    for p in dead:
        assert p['dead_role'] is not None


def test_illegal_move_rejected():
    session = _make_session()
    if session.over or not session.my_turn:
        pytest.skip('human not at a decision')
    from layer4_interface.frontend.play_werewolf.session import PlayError

    with pytest.raises(PlayError):
        session.human_move('vote', {'target': 'p0'})  # 非投票阶段或非法目标
