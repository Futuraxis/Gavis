"""Generate rules/mahjong.json — one JSON serving all variants.

Variants (guangdong/hongzhong/blood) × player counts (2/4) are injected
by MahjongAdapter into ``constants`` (variant, player_count, player_ids,
deal_target) at engine construction.  Everything else is static and
self-contained: zero builtins, pure expression aliases (v5.1).
"""

from __future__ import annotations

import json
import sys

# ── Tile data ─────────────────────────────────────────────────────────

SUITS = ['m', 'p', 's', 'z']
RANKS = {'m': 9, 'p': 9, 's': 9, 'z': 7}

TILE_IDS = []
for s in SUITS:
    for r in range(1, RANKS[s] + 1):
        TILE_IDS.extend([f'{s}{r}'] * 4)

SUIT_OF = {f'{s}{r}': s for s in SUITS for r in range(1, RANKS[s] + 1)}

CHI_RUNS = [
    [f'{s}{r}', f'{s}{r + 1}', f'{s}{r + 2}']
    for s in ('m', 'p', 's') for r in range(1, 8)
]

THIRTEEN_ORPHANS = ['m1', 'm9', 'p1', 'p9', 's1', 's9',
                    'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7']

FAN_PAY = [10, 20, 40, 80, 160, 320, 640, 1280]  # pay_base × 2^(n-1)

PLAYERS4 = ['p0', 'p1', 'p2', 'p3']


# ── Expression helpers ────────────────────────────────────────────────

def V(path):
    return {'var': path}


def C(value):
    return {'const': value}


def GET(obj, field):
    return {'get': [obj, field]}


def AT(container, idx):
    return {'at': [container, idx]}


def EQ(a, b):
    return {'eq': [a, b]}


def GTE(a, b):
    return {'gte': [a, b]}


def LT(a, b):
    return {'lt': [a, b]}


def AND(*args):
    return {'and': list(args)}


def OR(*args):
    return {'or': list(args)}


def NOT(a):
    return {'not': a}


def ADD(a, b):
    return {'add': [a, b]}


def SUB(a, b):
    return {'sub': [a, b]}


def MUL(a, b):
    return {'mul': [a, b]}


def DIV(a, b):
    return {'div': [a, b]}


def IF(cond, then_, else_=None):
    return {'if': {'cond': cond, 'then': then_, 'else': else_}}


def CALL(name, *args):
    return {'call': [name, *args]}


def COUNT(expr):
    return {'count': expr}


def SUM(expr):
    return {'sum': expr}


def MIN2(a, b):
    """min(a, b) — list min; concat flattens singleton lists.

    Each singleton uses its own ``as`` var so ``$node`` (the outer
    group entity) is never shadowed by the range item.
    """
    def _sing(item, v):
        return MAP(RANGE(C(0), C(1)), item, as_var=v)

    return {'min': CONCAT(_sing(a, '$ma'), _sing(b, '$mb'))}


def FILTER(lst, where, as_var='$node'):
    return {'filter': {'list': lst, 'as': as_var, 'where': where}}


def MAP(lst, expr, as_var='$node'):
    return {'map': {'list': lst, 'as': as_var, 'expr': expr}}


def ANY(lst, where, as_var='$node'):
    return {'any': {'list': lst, 'as': as_var, 'where': where}}


def ALL(lst, where, as_var='$node'):
    return {'all': {'list': lst, 'as': as_var, 'where': where}}


def RANGE(frm, to):
    return {'range': {'from': frm, 'to': to}}


def CONCAT(*items):
    return {'concat': list(items)}


def SINGLE(item):
    """[item] — a singleton list (concat str-joins when any item is scalar)."""
    return MAP(RANGE(C(0), C(1)), item)


def FLAT2(lst_of_lists):
    """Flatten a list of melds (lists of tiles) → tiles.

    ``concat`` flattens one level of its *items*; a single list argument
    would pass through unchanged, so each meld is addressed by ``at``
    (missing melds in partial combos become None and are dropped).
    """
    return CONCAT(AT(lst_of_lists, C(0)), AT(lst_of_lists, C(1)),
                  AT(lst_of_lists, C(2)), AT(lst_of_lists, C(3)))


def REPEAT(item, n):
    """[item] × n."""
    return MAP(RANGE(C(0), C(n)), item)


def SWITCH(cases, input_):
    return {'switch': [{'case': cv, 'then': te} for cv, te in cases],
            'input': input_}


def GROUP(lst):
    return {'group': {'list': lst}}


def DISTINCT(lst):
    return {'distinct': lst}


# ── Rule idioms ───────────────────────────────────────────────────────

def _idx_of(p_expr):
    """0-based index of a player id within constants.player_ids
    (= count of ids strictly less than it)."""
    return COUNT(FILTER(V('$constants.player_ids'), LT(V('$node'), p_expr)))


def _next_turn_expr():
    """Cyclic next player: player_ids[(idx+1) % n]."""
    i = _idx_of(V('$p'))
    n = COUNT(V('$constants.player_ids'))
    j = ADD(i, C(1))
    return AT(V('$constants.player_ids'), SUB(j, MUL(DIV(j, n), n)))


def _hand_count(hand_expr, tile_expr):
    """Count of ``tile_expr`` in ``hand_expr``.

    The filter binds ``$h`` (not ``$node``) so a ``tile_expr`` that
    references the surrounding ``$node`` (e.g. a group key) still
    resolves to the outer entity.
    """
    return COUNT(FILTER(hand_expr, EQ(V('$h'), tile_expr), as_var='$h'))


def _min_sel_have(group_expr, hand_expr):
    """Σ_g min(sel_g, have_g) over the selected-tile groups."""
    return SUM(MAP(
        {'group': {'list': group_expr}},
        MIN2(GET(V('$node'), 'count'),
             _hand_count(hand_expr, GET(V('$node'), 'key'))),
    ))


def _cover_ok(selected_expr, hand_expr):
    """Selected tiles coverable by hand + wild: min_sum + wild >= 14."""
    return GTE(ADD(_min_sel_have(selected_expr, hand_expr),
                   _hand_count(hand_expr, V('$constants.wild_tile'))), C(14))


def _cover_prefix(hand_expr):
    """Monotone prefix prune for partial meld sets: the tiles selected so
    far (3k) must be coverable by hand + wild: min_sum + wild >= 3k.
    Adding a meld raises min_sum by ≤3 and the bound by exactly 3, so a
    failing prefix stays failing for every extension (sound prune)."""
    return GTE(ADD(_min_sel_have(FLAT2(V('$m')), hand_expr),
                   _hand_count(hand_expr, V('$constants.wild_tile'))),
               MUL(C(3), COUNT(V('$m'))))


def _meld_pool(hand_expr):
    """Runs/pungs the hand can supply once wild tiles fill gaps.

    A run is a candidate when its three tiles are coverable:
    Σ min(1, have(t)) + wild ≥ 3; a pung needs count(t) + wild ≥ 3.
    The exact global coverage is verified by the choose ``where``.
    """
    wild = _hand_count(hand_expr, V('$constants.wild_tile'))
    chi = FILTER(V('$constants.chi_runs'),
                 GTE(ADD(SUM(MAP(V('$run'),
                                 MIN2(C(1), _hand_count(hand_expr, V('$node2'))),
                                 as_var='$node2')), wild), C(3)),
                 as_var='$run')
    pung = MAP(FILTER(GROUP(hand_expr),
                      GTE(ADD(GET(V('$node'), 'count'), wild), C(3))),
               GET(V('$node'), 'key'))
    return CONCAT(chi, pung)


def _pair_pool(hand_expr):
    wild = _hand_count(hand_expr, V('$constants.wild_tile'))
    return MAP(FILTER(GROUP(hand_expr),
                      GTE(ADD(GET(V('$node'), 'count'), wild), C(2))),
               GET(V('$node'), 'key'))


def _qidui(hand_expr):
    return ALL(GROUP(hand_expr), EQ(GET(V('$node'), 'count'), C(2)))


def _standard_win(hand_expr):
    """exists 4 melds (+ prefix prune) and 1 pair covering the hand."""
    return {'choose': {
        'items': _meld_pool(hand_expr), 'k': 4, 'as': '$m',
        'prefix': _cover_prefix(hand_expr),
        'where': {'choose': {
            'items': _pair_pool(hand_expr), 'k': 1, 'as': '$p',
            'where': _cover_ok(
                CONCAT(FLAT2(V('$m')), REPEAT(AT(V('$p'), C(0)), 2)),
                hand_expr),
        }},
    }}


# ── Sections ──────────────────────────────────────────────────────────

def _ground_env_fields():
    return {
        'phase': {'type': 'string', 'initial': 'deal'},
        'turn': {'type': 'player_id', 'initial': 'p0'},
        'actor': {'type': 'player_id?', 'initial': None},
        'last_discard': {'type': 'tile?', 'initial': None},
        'last_discarder': {'type': 'player_id?', 'initial': None},
        'last_drawn': {'type': 'tile?', 'initial': None},
        'dealer_idx': {'type': 'int', 'initial': 0},
        'claim_queue': {'type': 'list', 'initial': []},
        'claim_index': {'type': 'int', 'initial': 0},
        'dealt_count': {'type': 'int', 'initial': 0},
        'wall_count': {'type': 'int', 'initial': 136},
        'winners': {'type': 'list', 'initial': []},
        'done': {'type': 'list', 'initial': []},
        'fan_pay': {'type': 'int', 'initial': 0},
        'win_hand': {'type': 'list', 'initial': []},
        'payoffs': {'type': 'list', 'initial': []},
        'game_over': {'type': 'bool', 'initial': False},
        'winner': {'type': 'player_id?', 'initial': None},
        'last_action': {'type': 'string?', 'initial': None},
    }


def _ground_state():
    arrays = {}
    for p in PLAYERS4:
        arrays[f'hand_{p}'] = {'type': 'array', 'mutable': True, 'element': 'tile'}
        arrays[f'melds_{p}'] = {'type': 'array', 'mutable': True, 'element': 'meld'}
        arrays[f'discard_{p}'] = {'type': 'array', 'mutable': True, 'element': 'tile'}
    arrays['drawn'] = {'type': 'array', 'mutable': True, 'element': 'tile'}
    arrays['env'] = {'type': 'env', 'fields': _ground_env_fields()}
    return arrays


def _derived_views():
    views = {
        'tile': {'from': {'type': 'literal', 'list': {'var': 'tile_ids'}},
                 'fields': {'id': V('$self.value')}},
        'player': {'from': {'type': 'literal', 'list': {'var': '$players'}},
                   'fields': {'id': V('$self.value.id'), 'idx': V('$i')}},
    }
    for p in PLAYERS4:
        views[f'hand_view_{p}'] = {
            'from': {'type': 'enum', 'array': f'hand_{p}'},
            'fields': {'id': V('$self.value')}}
        views[f'meld_view_{p}'] = {
            'from': {'type': 'enum', 'array': f'melds_{p}'},
            'fields': {'id': V('$self.value')}}
        views[f'discard_view_{p}'] = {
            'from': {'type': 'enum', 'array': f'discard_{p}'},
            'fields': {'id': V('$self.value')}}
    return views


def _queries():
    return {'undrawn_tiles': {
        'view': 'tile',
        # The view holds 4 physical copies per tile kind but ``drawn``
        # records kinds — a kind stays available while fewer than 4
        # copies have been drawn (uniform over remaining kinds).
        'filter': {'lt': [{'count': FILTER(V('$drawn'), EQ(V('$t'),
                                                           GET(V('$node'), 'id')),
                                           as_var='$t')},
                          {'const': 4}]},
    }}


CLAIM_ACTOR = AT(V('$env.claim_queue'), V('$env.claim_index'))


def _actions():
    return [
        {'id': 'discard', 'type': 'move', 'phases': ['action'],
         'actor': V('$env.turn'),
         'params': {'tile': {'domain': {'array': {'template': 'hand_{$env.turn}'}}}},
         'legal': C(True),
         'effectRef': 'do_discard',
         'canonicalKey': {'template': 'discard:{tile}'}},
        {'id': 'win_self', 'type': 'move', 'phases': ['action'],
         'actor': V('$env.turn'), 'params': {},
         'legal': CALL('is_win_hand',
                       CONCAT(CALL('hand_of', V('$env.turn')),
                              SINGLE(V('$env.last_drawn')))),
         'effectRef': 'do_win_self',
         'canonicalKey': {'const': 'win_self'}},
        {'id': 'gang_concealed', 'type': 'move', 'phases': ['action'],
         'actor': V('$env.turn'),
         'params': {'tile': {'domain': {'array': {'template': 'hand_{$env.turn}'}}}},
         'legal': EQ(COUNT(FILTER(CALL('hand_of', V('$env.turn')),
                                  EQ(V('$node'), V('tile')))), C(4)),
         'effectRef': 'do_gang_concealed',
         'canonicalKey': {'template': 'gang_concealed:{tile}'}},
        {'id': 'gang_added', 'type': 'move', 'phases': ['action'],
         'actor': V('$env.turn'),
         'params': {'tile': {'domain': {'array': {'template': 'hand_{$env.turn}'}}}},
         'legal': AND(
             GTE(COUNT(FILTER(CALL('hand_of', V('$env.turn')),
                              EQ(V('$node'), V('tile')))), C(1)),
             ANY(CALL('melds_of', V('$env.turn')),
                 AND(EQ(GET(V('$node'), 'type'), C('peng')),
                     EQ(AT(GET(V('$node'), 'tiles'), C(0)), V('tile'))))),
         'effectRef': 'do_gang_added',
         'canonicalKey': {'template': 'gang_added:{tile}'}},
        {'id': 'claim_win', 'type': 'move', 'phases': ['claim'],
         'actor': CLAIM_ACTOR,
         'params': {'tile': {'domain': {'expr': SINGLE(V('$env.last_discard'))}}},
         'legal': CALL('is_win_hand',
                       CONCAT(CALL('hand_of', CLAIM_ACTOR), SINGLE(V('tile')))),
         'effectRef': 'do_claim_win',
         'canonicalKey': {'template': 'claim_win:{tile}'}},
        {'id': 'claim_peng', 'type': 'move', 'phases': ['claim'],
         'actor': CLAIM_ACTOR,
         'params': {'tile': {'domain': {'expr': SINGLE(V('$env.last_discard'))}}},
         'legal': GTE(COUNT(FILTER(CALL('hand_of', CLAIM_ACTOR),
                                   EQ(V('$node'), V('tile')))), C(2)),
         'effectRef': 'do_claim_peng',
         'canonicalKey': {'template': 'claim_peng:{tile}'}},
        {'id': 'claim_gang', 'type': 'move', 'phases': ['claim'],
         'actor': CLAIM_ACTOR,
         'params': {'tile': {'domain': {'expr': SINGLE(V('$env.last_discard'))}}},
         'legal': GTE(COUNT(FILTER(CALL('hand_of', CLAIM_ACTOR),
                                   EQ(V('$node'), V('tile')))), C(3)),
         'effectRef': 'do_claim_gang',
         'canonicalKey': {'template': 'claim_gang:{tile}'}},
        {'id': 'claim_chi', 'type': 'move', 'phases': ['claim'],
         'actor': CLAIM_ACTOR,
         'params': {'tiles': {'domain': {'expr': FILTER(
             V('$constants.chi_runs'),
             {'contains': [V('$node'), V('$env.last_discard')]})}}},
         'legal': EQ(V('$env.claim_index'), C(0)),
         'effectRef': 'do_claim_chi',
         'canonicalKey': {'template': 'claim_chi:{tiles}'}},
        {'id': 'claim_pass', 'type': 'move', 'phases': ['claim'],
         'actor': CLAIM_ACTOR, 'params': {},
         'legal': C(True),
         'effectRef': 'do_claim_pass',
         'canonicalKey': {'const': 'claim_pass'}},
    ]


def _set_env(key, value):
    return {'op': 'setEnv', 'key': key, 'value': value}


def _append(arr_name, value):
    return {'op': 'append', 'array': arr_name, 'value': value}


def _remove(arr_name, value, count=1):
    return {'op': 'remove', 'array': arr_name, 'value': value,
            'count': {'const': count}}


def _branch(cond, then_ops, else_ops=None):
    return {'op': 'branch', 'if': cond, 'then': then_ops,
            'else': else_ops if else_ops is not None else []}


def _inc(key, by):
    return {'op': 'inc', 'key': key, 'by': by}


def _call_effect(ref, args=None):
    return {'op': 'callEffect', 'effectRef': ref, 'args': args or {}}


def _end_game(action_label=None):
    """Ops that end the round: set game_over + phase + last_action."""
    ops = [_set_env('game_over', C(True)), _set_env('phase', C('game_over'))]
    if action_label is not None:
        ops.append(_set_env('last_action', C(action_label)))
    return ops


def _effectors():
    wall_empty = EQ(V('$env.wall_count'), C(0))
    turn_hand = {'template': 'hand_{$env.turn}'}
    actor = CLAIM_ACTOR
    actor_hand_tmpl = {'template': 'hand_{$env.actor}'}
    actor_melds_tmpl = {'template': 'melds_{$env.actor}'}
    to_gang_draw = [
        _branch(wall_empty, _end_game('wall_empty'),
                [_set_env('phase', C('gang_draw'))]),
    ]

    return {
        'to_draw': {
            'description': 'Advance to the draw phase (or end on an empty wall)',
            'ops': [
                _branch(wall_empty, _end_game('wall_empty'),
                        [_set_env('phase', C('draw'))]),
            ],
        },
        'do_draw': {
            'description': 'Deal / draw / gang-draw: one tile from the wall',
            'ops': [
                _branch(EQ(V('$env.phase'), C('deal')), [
                    _append(turn_hand, V('outcome')),
                    _append('drawn', V('outcome')),
                    _inc('wall_count', -1),
                    _inc('dealt_count', 1),
                    _set_env('last_drawn', V('outcome')),
                    _set_env('turn', CALL('next_turn', V('$env.turn'))),
                    _branch(GTE(V('$env.dealt_count'), V('$constants.deal_target')), [
                        _set_env('phase', C('action')),
                        _set_env('turn', C('p0')),
                        _set_env('dealt_count', C(0)),
                        _set_env('last_action', C('deal_done')),
                        # zero-score baseline — a wall-empty end settles 0
                        _set_env('payoffs', MAP(V('$players'), C(0))),
                    ], []),
                ], [
                    _append(turn_hand, V('outcome')),
                    _append('drawn', V('outcome')),
                    _inc('wall_count', -1),
                    _set_env('last_drawn', V('outcome')),
                    _set_env('phase', C('action')),
                ]),
            ],
        },
        'do_discard': {
            'description': 'Discard a tile, open the claim queue',
            'ops': [
                _remove(turn_hand, V('tile')),
                _append({'template': 'discard_{$env.turn}'}, V('tile')),
                _set_env('last_discard', V('tile')),
                _set_env('last_discarder', V('$env.turn')),
                _set_env('dealer_idx', GET(
                    AT(FILTER({'query': {'view': 'player'}},
                              EQ(GET(V('$node'), 'id'), V('$env.turn'))), C(0)),
                    'idx')),
                _set_env('claim_queue', MAP(
                    {'sort': {
                        'list': FILTER({'query': {'view': 'player'}},
                                       AND(NOT({'contains': [V('$env.done'),
                                                             GET(V('$node'), 'id')]}),
                                           {'not': {'eq': [GET(V('$node'), 'id'),
                                                           V('$env.last_discarder')]}})),
                        'by': {'expr': '(node.idx - $env.dealer_idx + 4) % 4'}}},
                    GET(V('$node'), 'id'))),
                _set_env('claim_index', C(0)),
                _set_env('last_action', C('discard')),
                _set_env('phase', C('claim')),
            ],
        },
        'do_claim_pass': {
            'description': 'Pass the claim; open the draw once all pass',
            'ops': [
                _inc('claim_index', 1),
                _branch(GTE(V('$env.claim_index'), COUNT(V('$env.claim_queue'))), [
                    _set_env('last_action', C('pass_all')),
                    _set_env('turn', AT(V('$env.claim_queue'), C(0))),
                    _set_env('claim_queue', C([])),
                    _set_env('claim_index', C(0)),
                    _call_effect('to_draw'),
                ], [
                    _set_env('last_action', C('pass')),
                ]),
            ],
        },
        'do_claim_peng': {
            'description': 'Pung: take the discard, meld a triplet',
            'ops': [
                _set_env('actor', actor),
                _remove(actor_hand_tmpl, V('tile'), 2),
                _append(actor_melds_tmpl,
                        {'type': 'peng', 'tiles': REPEAT(V('tile'), 3),
                         'from': V('$env.last_discarder')}),
                _set_env('turn', V('$env.actor')),
                _set_env('claim_queue', C([])),
                _set_env('claim_index', C(0)),
                _set_env('last_action', C('peng')),
                _call_effect('to_draw'),
            ],
        },
        'do_claim_gang': {
            'description': 'Exposed gang: take the discard, meld quads',
            'ops': [
                _set_env('actor', actor),
                _remove(actor_hand_tmpl, V('tile'), 3),
                _append(actor_melds_tmpl,
                        {'type': 'gang', 'tiles': REPEAT(V('tile'), 4),
                         'from': V('$env.last_discarder')}),
                _set_env('turn', V('$env.actor')),
                _set_env('claim_queue', C([])),
                _set_env('claim_index', C(0)),
                _set_env('last_action', C('gang')),
                *to_gang_draw,
            ],
        },
        'do_claim_chi': {
            'description': 'Chi (first responder only): meld a run',
            'ops': [
                _set_env('actor', actor),
                {'op': 'forEach', 'list': V('tiles'),
                 'do': [_remove(actor_hand_tmpl, V('$item'))]},
                _append(actor_melds_tmpl, {'type': 'chi', 'tiles': V('tiles')}),
                _set_env('turn', V('$env.actor')),
                _set_env('claim_queue', C([])),
                _set_env('claim_index', C(0)),
                _set_env('last_action', C('chi')),
                _call_effect('to_draw'),
            ],
        },
        'do_claim_win': {
            'description': 'Win off a discard (ron)',
            'ops': [
                _call_effect('do_win', {'pid': CLAIM_ACTOR,
                                        'tile': V('$env.last_discard'),
                                        'self_win': C(False)}),
            ],
        },
        'do_win_self': {
            'description': 'Win on the self-drawn tile (tsumo)',
            'ops': [
                _call_effect('do_win', {'pid': V('$env.turn'),
                                        'tile': V('$env.last_drawn'),
                                        'self_win': C(True)}),
            ],
        },
        'do_gang_concealed': {
            'description': 'Concealed gang from four held tiles',
            'ops': [
                {'op': 'forEach', 'list': RANGE(C(0), C(4)),
                 'do': [_remove(turn_hand, V('tile'))]},
                _append({'template': 'melds_{$env.turn}'},
                        {'type': 'concealed_gang', 'tiles': REPEAT(V('tile'), 4)}),
                _set_env('last_action', C('gang_concealed')),
                *to_gang_draw,
            ],
        },
        'do_gang_added': {
            'description': 'Added gang: promote a pung to a quad',
            'ops': [
                _remove(turn_hand, V('tile')),
                _set_env('gang_tiles', GET(
                    AT(FILTER(CALL('melds_of', V('$env.turn')),
                              AND(EQ(GET(V('$node'), 'type'), C('peng')),
                                  EQ(AT(GET(V('$node'), 'tiles'), C(0)), V('tile')))),
                        C(0)), 'tiles')),
                {'op': 'setArray', 'array': {'template': 'melds_{$env.turn}'},
                 'value': FILTER(CALL('melds_of', V('$env.turn')),
                                 NOT(AND(EQ(GET(V('$node'), 'type'), C('peng')),
                                         EQ(AT(GET(V('$node'), 'tiles'), C(0)),
                                            V('tile')))))},
                _append({'template': 'melds_{$env.turn}'},
                        {'type': 'added_gang',
                         'tiles': CONCAT(V('$env.gang_tiles'), SINGLE(V('tile')))}),
                _set_env('last_action', C('gang_added')),
                *to_gang_draw,
            ],
        },
        'do_win': {
            'description': 'Common win settlement: fans, winners, continue or end',
            'ops': [
                _set_env('win_hand', CONCAT(CALL('hand_of', V('pid')),
                                            SINGLE(V('tile')))),
                _set_env('fan_pay', AT(V('$constants.fan_pay'),
                                       SUB(CALL('fan_sum', V('$env.win_hand')), C(1)))),
                _branch(NOT({'contains': [V('$env.winners'), V('pid')]}),
                        [_append('winners', V('pid'))], []),
                _branch(NOT({'contains': [V('$env.done'), V('pid')]}),
                        [_append('done', V('pid'))], []),
                _set_env('last_action', IF(V('self_win'), C('win_self'),
                                           C('win_discard'))),
                _branch(EQ(V('$constants.variant'), C('blood')), [
                    _branch(OR(GTE(COUNT(V('$env.done')), C(2)),
                               EQ(V('$env.wall_count'), C(0))),
                            _end_game('blood_over'), [
                        _set_env('turn', IF(
                            {'contains': [V('$env.done'), CALL('next_turn', V('pid'))]},
                            CALL('next_turn', CALL('next_turn', V('pid'))),
                            CALL('next_turn', V('pid')))),
                        _call_effect('to_draw'),
                    ]),
                ], [
                    *_end_game(),
                    _set_env('winner', V('pid')),
                ]),
                _set_env('payoffs', MAP(
                    V('$players'),
                    CALL('payoff_for', GET(V('$node'), 'id'), V('pid'),
                         V('$env.fan_pay')))),
            ],
        },
    }


def _chance():
    return [{
        'id': 'draw',
        'phases': ['deal', 'draw', 'gang_draw'],
        'params': {'tile': {'view': 'tile', 'domain': {'ref': 'undrawn_tiles'}}},
        'probability': {'uniform': {'over': 'tile'}},
        'effectRef': 'do_draw',
        'canonicalKey': {'template': 'draw:{outcome}'},
    }]


def _phases():
    return [
        {'id': 'deal', 'actions': [], 'description': 'Chance: deal 13N+1 tiles'},
        {'id': 'action', 'actions': ['discard', 'win_self', 'gang_concealed',
                                     'gang_added'],
         'description': 'Player: discard / win / gang'},
        {'id': 'claim', 'actions': ['claim_win', 'claim_peng', 'claim_gang',
                                    'claim_chi', 'claim_pass'],
         'description': 'Player: respond to a discard'},
        {'id': 'draw', 'actions': [], 'description': 'Chance: draw one tile'},
        {'id': 'gang_draw', 'actions': [], 'description': 'Chance: gang replacement'},
        {'id': 'game_over', 'actions': [], 'description': 'Round finished'},
    ]


def _aliases():
    hand = V('$hand')
    suits = MAP(hand, AT(V('$node'), C(0)))
    suits_noz = MAP(FILTER(hand, NOT(EQ(AT(V('$node'), C(0)), C('z')))),
                    AT(V('$node'), C(0)))
    fans = {
        'fan_jihu': {'description': '鸡胡', 'params': ['hand'], 'expr': C(1)},
        'fan_pinghu': {'description': '平胡 (approx: no triplets)',
                       'params': ['hand'],
                       'expr': EQ(COUNT(FILTER(GROUP(hand),
                                               GTE(GET(V('$node'), 'count'),
                                                   C(3)))), C(0))},
        'fan_pengpenghu': {'description': '碰碰胡 (approx: four triplets)',
                           'params': ['hand'],
                           'expr': EQ(COUNT(FILTER(GROUP(hand),
                                                   GTE(GET(V('$node'), 'count'),
                                                       C(3)))), C(4))},
        'fan_qingyise': {'description': '清一色', 'params': ['hand'],
                         'expr': ALL(hand, EQ(AT(V('$node'), C(0)),
                                              AT(AT(hand, C(0)), C(0))))},
        'fan_hunyise': {'description': '混一色 (approx: honors + one suit)',
                        'params': ['hand'],
                        'expr': AND(ANY(hand, EQ(AT(V('$node'), C(0)), C('z'))),
                                    EQ(COUNT(DISTINCT(suits_noz)), C(1)))},
        'fan_qidui': {'description': '七对', 'params': ['hand'],
                      'expr': CALL('is_qidui', hand)},
        'fan_shisanyao': {'description': '十三幺', 'params': ['hand'],
                          'expr': ALL(V('$constants.thirteen_orphans'),
                                      {'contains': [hand, V('$node')]})},
        'fan_hongzhongke': {'description': '红中刻 (hongzhong only)',
                            'params': ['hand'],
                            'expr': IF(EQ(V('$constants.variant'), C('hongzhong')),
                                       GTE(COUNT(FILTER(hand, EQ(V('$node'),
                                                                 V('$constants.wild_tile')))),
                                           C(3)), C(0))},
        'fan_jueshang': {'description': '缺一门 (blood only, approx)',
                         'params': ['hand'],
                         'expr': IF(EQ(V('$constants.variant'), C('blood')),
                                    EQ(COUNT(DISTINCT(suits)), C(2)), C(0))},
    }
    fan_order = ['fan_jihu', 'fan_pinghu', 'fan_pengpenghu', 'fan_qingyise',
                 'fan_hunyise', 'fan_qidui', 'fan_shisanyao', 'fan_hongzhongke',
                 'fan_jueshang']
    fan_value = {'fan_jihu': 1, 'fan_pinghu': 2, 'fan_pengpenghu': 3,
                 'fan_qingyise': 5, 'fan_hunyise': 2, 'fan_qidui': 4,
                 'fan_shisanyao': 8, 'fan_hongzhongke': 1, 'fan_jueshang': 1}
    fan_sum_expr = C(0)
    for name in fan_order:
        fan_sum_expr = ADD(fan_sum_expr,
                           MUL(CALL(name, hand), C(fan_value[name])))

    return {
        'next_turn': {
            'description': 'Cyclic next player within constants.player_ids',
            'params': ['p'], 'expr': _next_turn_expr(),
        },
        'hand_of': {
            'description': "A player's hand array",
            'params': ['p'],
            'expr': SWITCH([(p, V(f'$hand_{p}')) for p in PLAYERS4], V('$p')),
        },
        'melds_of': {
            'description': "A player's meld list",
            'params': ['p'],
            'expr': SWITCH([(p, V(f'$melds_{p}')) for p in PLAYERS4], V('$p')),
        },
        'is_qidui': {
            'description': 'Seven pairs (every group of size 2)',
            'params': ['hand'], 'expr': _qidui(hand),
        },
        'is_win_hand': {
            'description': 'Winning hand: 7 pairs, thirteen orphans, '
                           'or standard form with coverage',
            'params': ['hand'],
            'expr': OR(CALL('is_qidui', hand),
                       ALL(V('$constants.thirteen_orphans'),
                           {'contains': [hand, V('$node')]}),
                       _standard_win(hand)),
        },
        'fan_sum': {
            'description': 'Sum of all fan flags for a winning hand',
            'params': ['hand'], 'expr': fan_sum_expr,
        },
        'payoff_for': {
            'description': 'Net score for one player when ``winner`` wins ``fan_pay``',
            'params': ['pid', 'winner', 'fan_pay'],
            'expr': IF(
                EQ(V('pid'), V('winner')),
                MUL(V('fan_pay'),
                    SUB(V('$constants.player_count'), COUNT(V('$env.done')))),
                IF({'contains': [V('$env.done'), V('pid')]}, C(0),
                   SUB(C(0), V('fan_pay')))),
        },
        **fans,
    }


def _visibility():
    rules = []
    for p in PLAYERS4:
        rules.append({
            'view': f'hand_view_{p}',
            'filter': NOT(EQ(V('$viewer'), p)),
            'fields': {'id': 'hidden'},
        })
    return {'default': 'partial', 'rules': rules}


def _terminal():
    return [{'id': 'round_done', 'condition': EQ(V('$env.game_over'), C(True))}]


def _utility():
    return [
        {'player': p, 'value': AT(V('$env.payoffs'), C(i)),
         'when': EQ(V('$env.game_over'), C(True))}
        for i, p in enumerate(PLAYERS4)
    ]


def build() -> dict:
    return {
        'meta': {
            'gameId': 'mahjong',
            'version': '5.1.0',
            'description': ('Mahjong — guangdong jihu / hongzhong wild / blood '
                            'variants × 2-4 players. Variant and player_count are '
                            'injected into constants by MahjongAdapter. '
                            'Pure-expression aliases (zero builtins).'),
        },
        'players': PLAYERS4,
        'constants': {
            'variant': 'guangdong',
            'player_count': 2,
            'deal_target': 27,          # 13×N + 1 (injected per player_count)
            'player_ids': ['p0', 'p1'],
            'tile_ids': TILE_IDS,
            'suit_of': SUIT_OF,
            'chi_runs': CHI_RUNS,
            'thirteen_orphans': THIRTEEN_ORPHANS,
            'wild_tile': 'z5',
            'fan_pay': FAN_PAY,
        },
        'groundState': _ground_state(),
        'derivedViews': _derived_views(),
        'queries': _queries(),
        'actions': _actions(),
        'effectors': _effectors(),
        'chance': _chance(),
        'phases': _phases(),
        'visibility': _visibility(),
        'terminal': _terminal(),
        'utility': _utility(),
        'functions': _aliases(),
    }


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'rules/mahjong.json'
    rules = build()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    print(f'written {path} ({len(json.dumps(rules))} bytes)')
