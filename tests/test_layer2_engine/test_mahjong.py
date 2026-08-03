"""Tests for Mahjong (Layer 2, v5.1 — one JSON, all variants).

Covers dealing, draw/discard loops, chi/peng/gang claims, concealed and
added gangs, self-win (tsumo) and ron wins, 7 pairs / thirteen orphans,
hongzhong wild-tile coverage, blood-variant done-skip, wall-empty draws,
and claim-queue rotation.
"""

from __future__ import annotations

from layer2_engine.games.mahjong.mahjong_adapter import MahjongAdapter


def _resolve(adapter, state: dict) -> dict:
    while adapter.get_node_type(state) == 'chance':
        _, state = adapter.sample_chance(state)
    return state


def _act(adapter, state: dict, template: str, **params) -> dict:
    for a in adapter.get_legal_actions(state):
        if a.template_id == template and all(
                a.params.get(k) == v for k, v in params.items()):
            return adapter.apply_action(state, a)
    raise AssertionError(f'not legal: {template} {params} at {state["env"]["phase"]}')


def _resolve_after(adapter, state: dict) -> dict:
    state = _resolve(adapter, state)
    return state


def _win_legal(adapter, state: dict) -> bool:
    """Is a win legal for the current turn (self-win or ron)?"""
    for a in adapter.get_legal_actions(state):
        if a.template_id in ('win_self', 'claim_win'):
            return True
    return False


def _eval_win(adapter, hand: list) -> bool:
    ctx = adapter._build_context(adapter.create_initial_state())  # noqa: SLF001
    return bool(adapter.expr.eval({'call': ['is_win_hand', {'const': hand}]}, ctx))


# ── Dealing ───────────────────────────────────────────────────────────

class TestDeal:
    def test_deal_2p(self):
        a = MahjongAdapter(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        assert len(s['_arrays']['hand_p0']) == 14
        assert len(s['_arrays']['hand_p1']) == 13
        assert s['env']['wall_count'] == 136 - 27
        assert s['env']['phase'] == 'action'
        assert s['env']['turn'] == 'p0'

    def test_deal_4p(self):
        a = MahjongAdapter(player_count=4, seed=1)
        s = _resolve(a, a.create_initial_state())
        assert [len(s['_arrays'][f'hand_p{i}']) for i in range(4)] == [14, 13, 13, 13]
        assert s['env']['wall_count'] == 136 - 53

    def test_no_duplicate_tiles(self):
        a = MahjongAdapter(player_count=4, seed=2)
        s = _resolve(a, a.create_initial_state())
        drawn = s['_arrays']['drawn']
        assert len(drawn) == 53
        assert max(drawn.count(t) for t in set(drawn)) <= 4  # kinds, 4 copies


# ── Draw / discard loop ───────────────────────────────────────────────

class TestFlow:
    def test_discard_then_draw(self):
        a = MahjongAdapter(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s = _act(a, s, 'discard', tile=s['_arrays']['hand_p0'][0])
        assert s['env']['phase'] == 'claim'
        assert s['env']['claim_queue'] == ['p1']
        # both pass → draw
        s = _act(a, s, 'claim_pass')
        assert s['env']['phase'] == 'draw'
        s = _resolve_after(a, s)
        assert s['env']['phase'] == 'action'
        assert s['env']['turn'] == 'p1'
        assert len(s['_arrays']['hand_p1']) == 14

    def test_claim_queue_rotation_4p(self):
        a = MahjongAdapter(player_count=4, seed=1)
        s = _resolve(a, a.create_initial_state())
        s = _act(a, s, 'discard', tile=s['_arrays']['hand_p0'][0])
        assert s['env']['claim_queue'] == ['p1', 'p2', 'p3']
        s = _act(a, s, 'claim_pass')
        s = _act(a, s, 'claim_pass')
        s = _act(a, s, 'claim_pass')
        assert s['env']['phase'] == 'draw'
        s = _resolve_after(a, s)
        assert s['env']['turn'] == 'p1'

    def test_peng_claims_and_melds(self):
        a = MahjongAdapter(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        tile = s['_arrays']['hand_p0'][0]
        s = _act(a, s, 'discard', tile=tile)
        # Force a peng by p1: plant two copies into p1's hand
        s['_arrays']['hand_p1'] = [tile, tile] + s['_arrays']['hand_p1'][:-2]
        s = _act(a, s, 'claim_peng', tile=tile)
        assert s['env']['phase'] == 'draw'
        assert s['env']['turn'] == 'p1'
        melds = s['_arrays']['melds_p1']
        assert melds == [{'type': 'peng', 'tiles': [tile, tile, tile],
                          'from': 'p0'}]
        assert s['_arrays']['hand_p1'].count(tile) == 0

    def test_chi_only_first_responder(self):
        a = MahjongAdapter(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        # Plant a run around a tile p0 discards
        hand = s['_arrays']['hand_p1']
        tile = 'm5'
        s['_arrays']['hand_p0'] = [tile] + s['_arrays']['hand_p0'][1:]
        rest = [t for t in hand if t not in ('m4', 'm5', 'm6')][:10]
        s['_arrays']['hand_p1'] = ['m4', 'm6', 'm5'] + rest
        s = _act(a, s, 'discard', tile=tile)
        # p1 can chi m4+m6+m5
        s = _act(a, s, 'claim_chi', tiles=['m4', 'm5', 'm6'])
        assert s['env']['turn'] == 'p1'
        assert s['_arrays']['melds_p1'][0]['type'] == 'chi'
        assert s['_arrays']['melds_p1'][0]['tiles'] == ['m4', 'm5', 'm6']
        assert 'm4' not in s['_arrays']['hand_p1']
        assert 'm6' not in s['_arrays']['hand_p1']

    def test_concealed_gang_draws_replacement(self):
        a = MahjongAdapter(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        hand = s['_arrays']['hand_p0']
        tile = hand[0]
        hand[1:5] = [tile, tile, tile]  # four of a kind
        s = _act(a, s, 'gang_concealed', tile=tile)
        assert s['env']['phase'] == 'gang_draw'
        s = _resolve_after(a, s)
        assert s['env']['phase'] == 'action'
        assert s['env']['turn'] == 'p0'
        assert s['_arrays']['melds_p0'][0]['type'] == 'concealed_gang'

    def test_added_gang_promotes_peng(self):
        a = MahjongAdapter(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        tile = 'm7'
        s['_arrays']['melds_p0'] = [{'type': 'peng', 'tiles': [tile, tile, tile],
                                     'from': 'p1'}]
        hand = s['_arrays']['hand_p0']
        if tile not in hand:
            hand[0] = tile
        s = _act(a, s, 'gang_added', tile=tile)
        assert s['env']['phase'] == 'gang_draw'
        s = _resolve_after(a, s)
        assert s['_arrays']['melds_p0'] == [
            {'type': 'added_gang', 'tiles': [tile, tile, tile, tile]}]
        assert tile not in s['_arrays']['hand_p0']


# ── Win detection (expression aliases) ────────────────────────────────

class TestWinHand:
    def test_standard_win_123_456_789_pairs(self):
        a = MahjongAdapter(player_count=2, seed=1)
        hand = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9',
                'p1', 'p2', 'p3', 'p3', 'p3']
        assert _eval_win(a, hand)

    def test_not_a_win(self):
        a = MahjongAdapter(player_count=2, seed=1)
        hand = ['m1', 'm2', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9',
                'p1', 'p1', 'p2', 'p3', 'p3', 'p4']
        assert not _eval_win(a, hand)

    def test_seven_pairs(self):
        a = MahjongAdapter(player_count=2, seed=1)
        hand = ['m1', 'm1', 'm2', 'm2', 'p3', 'p3', 's4', 's4',
                'z1', 'z1', 'z2', 'z2', 'z3', 'z3']
        assert _eval_win(a, hand)

    def test_thirteen_orphans(self):
        a = MahjongAdapter(player_count=2, seed=1)
        hand = ['m1', 'm9', 'p1', 'p9', 's1', 's9', 'z1', 'z2', 'z3',
                'z4', 'z5', 'z6', 'z7', 'm1']
        assert _eval_win(a, hand)

    def test_hongzhong_wild_fills_gap(self):
        a = MahjongAdapter(variant='hongzhong', player_count=2, seed=1)
        # Two red dragons stand in for the missing m3 and m6
        hand = ['m1', 'm2', 'm4', 'm5', 'p1', 'p1', 'p1', 'p2', 'p3',
                's1', 's2', 's3', 'z5', 'z5']
        assert _eval_win(a, hand)

    def test_hongzhong_wild_not_enough(self):
        a = MahjongAdapter(variant='hongzhong', player_count=2, seed=1)
        hand = ['m1', 'm2', 'm4', 'm5', 'p1', 'p1', 'p1', 'p2', 'p3',
                's1', 's2', 's3', 'z5']
        assert not _eval_win(a, hand)


# ── Full wins through the engine ──────────────────────────────────────

class TestWins:
    def test_tsumo_win_ends_game(self):
        a = MahjongAdapter(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        # Plant a ready hand for p0 (14 tiles: three runs + pair, with the
        # last drawn tile completing the pair).
        s['_arrays']['hand_p0'] = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6',
                                   'p1', 'p2', 'p3', 's1', 's2', 's3',
                                   'p3', 'p3']
        s['env']['last_drawn'] = 'z1'
        assert _win_legal(a, s)
        s = _act(a, s, 'win_self')
        assert a.is_terminal(s)
        assert s['env']['winner'] == 'p0'
        p0, p1 = (float(x) for x in s['env']['payoffs'])
        assert p0 == -p1 and p0 > 0

    def test_ron_win_via_claim(self):
        a = MahjongAdapter(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        # p0 discards m3; p1 (13 tiles) rons with m3 completing the pair.
        s['_arrays']['hand_p0'] = ['m3'] + s['_arrays']['hand_p0'][1:]
        s['_arrays']['hand_p1'] = ['m1', 'm2', 'm4', 'm5', 'm6',
                                   'p1', 'p2', 'p3', 's1', 's2', 's3',
                                   'z1', 'z1']
        s = _act(a, s, 'discard', tile='m3')
        assert s['env']['phase'] == 'claim'
        s = _act(a, s, 'claim_win', tile='m3')
        assert a.is_terminal(s)
        assert s['env']['winner'] == 'p1'

    def test_fan_pay_scale(self):
        a = MahjongAdapter(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s['_arrays']['hand_p0'] = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6',
                                   'm7', 'm8', 'm9', 'p1', 'p2', 'p3',
                                   'z1']
        s['env']['last_drawn'] = 'z1'
        s = _act(a, s, 'win_self')
        # 鸡胡(1) + 平胡(2, 无刻子) = 3 番 → 10 × 2^2 = 40
        assert s['env']['fan_pay'] == 40


# ── Variants ──────────────────────────────────────────────────────────

class TestVariants:
    def test_blood_continues_after_first_win(self):
        a = MahjongAdapter(variant='blood', player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        # p0 tsumos
        s['_arrays']['hand_p0'] = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6',
                                   'p1', 'p2', 'p3', 's1', 's2', 's3',
                                   'p3', 'p3']
        s['env']['last_drawn'] = 'z1'
        s = _act(a, s, 'win_self')
        assert not a.is_terminal(s), 'blood continues after one win'
        assert s['env']['done'] == ['p0']
        assert s['env']['turn'] == 'p1'
        assert s['env']['phase'] == 'draw'
        # p1 draws, then tsumos → two done → over
        s = _resolve_after(a, s)
        s['_arrays']['hand_p1'] = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6',
                                   'p1', 'p2', 'p3', 's1', 's2', 's3',
                                   'z2', 'z2']
        s['env']['last_drawn'] = 'z2'
        s = _act(a, s, 'win_self')
        assert a.is_terminal(s)
        assert s['env']['done'] == ['p0', 'p1']

    def test_guangdong_ends_after_one_win(self):
        a = MahjongAdapter(variant='guangdong', player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s['_arrays']['hand_p0'] = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6',
                                   'p1', 'p2', 'p3', 's1', 's2', 's3',
                                   'p3', 'p3']
        s['env']['last_drawn'] = 'z1'
        s = _act(a, s, 'win_self')
        assert a.is_terminal(s)

    def test_wall_empty_draw(self):
        a = MahjongAdapter(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        s = _act(a, s, 'discard', tile=s['_arrays']['hand_p0'][0])
        # Drain the wall before the draw resolves
        s['env']['wall_count'] = 0
        s = _act(a, s, 'claim_pass')
        assert a.is_terminal(s)
        assert s['env']['last_action'] == 'wall_empty'
        assert a.get_utility(s, 'p0') == 0.0

    def test_visibility_hides_opponent_hand(self):
        a = MahjongAdapter(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        obs = a.project_observation(s, 'p0')
        assert all('id' in c for c in obs['hand_view_p0'])
        assert all('id' not in c for c in obs['hand_view_p1'])
        obs1 = a.project_observation(s, 'p1')
        assert all('id' in c for c in obs1['hand_view_p1'])

    def test_adapter_observation(self):
        a = MahjongAdapter(player_count=2, seed=1)
        s = _resolve(a, a.create_initial_state())
        obs = a.get_observation(s, 'p0')
        assert len(obs['hand']) == 14
        assert obs['phase'] == 'action'
        assert obs['my_turn'] is True
        assert 'discard' in {a['type'] for a in obs['legal']}
