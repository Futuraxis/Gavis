#!/usr/bin/env python3
"""Undercover (谁是卧底) rules generator — generates ``rules/undercover.json`` (v5.2).

Generates the standard Who-is-the-Undercover rule set with declarative variants:

    python _gen_undercover.py [--players 8] [--max-players 12]
                              [--scenario fruit|food] [--out rules/undercover.json]

Composition (any player count 4..max_players):
  1 卧底 (undercover) + 1 白板 (blank) + N 平民 (civilian)

Scenario variants (``variants.options`` word_of patch):
  - ``fruit`` (default): 平民「苹果」/ 卧底「香蕉」
  - ``food``:            平民「汉堡」/ 卧底「肉夹馍」

Round flow: deal → describe(每人一句话描述) → vote(投其他存活玩家) →
            resolve(得票最多者出局；平票无人出局) → win check → 下一轮。

Win conditions:
  - 卧底 or 白板 被投出          → 平民胜 (winner=civilian)
  - 存活 ≤ 3 且白板存活          → 白板胜 (winner=blank)
  - 存活 ≤ 2 且卧底存活          → 卧底胜 (winner=undercover)
  - 存活 ≤ 2（无卧底）           → 平民胜
  - 轮次上限 (players+8)         → 平局 (winner=None)

Design notes
------------
- 部分可观测：``my_role`` / ``my_word`` 按 viewer 过滤单行（v5.2 visibility）；
  死后身份/词语公开（``dead_roles`` / ``dead_words`` 保留 alive==0 的行）。
- 结算用 ``chance``/``resolve`` 阶段 + ``effectMap``（explicit 概率 1.0）。
- ``speak`` 使用 v5.1 的 ``text`` 自由文本参数预制能力。
- mutable 数组发牌/死亡名单用 ``append``（数组索引即玩家索引）。
- 平票无人出局：``most_voted`` 返回唯一最高票目标，平票 → None。
- variants 声明式：scenario(词对) 经 ``options[scenario].constants.word_of``
  补丁选择；``role_pool`` / ``player_ids`` 为 ``$player_count`` 上下文的公式
  （引擎纯数据解析，无注入 API）；未知 variant → ValueError。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROLES = ("undercover", "blank", "civilian")
ROLE_LABEL = {"undercover": "卧底", "blank": "白板", "civilian": "平民"}

SCENARIOS: dict[str, dict[str, str]] = {
    "fruit": {"civilian": "苹果", "undercover": "香蕉", "blank": "白板"},
    "food": {"civilian": "汉堡", "undercover": "肉夹馍", "blank": "白板"},
}


def _count_eq(list_expr: dict, item_var: str, node_id_expr: dict) -> dict:
    """count(list where item == node.id)"""
    return {
        "count": {
            "filter": {
                "list": list_expr,
                "as": item_var,
                "where": {"eq": [{"var": item_var}, node_id_expr]},
            }
        }
    }


def _alive_cond() -> dict:
    return {"eq": [{"at": [{"var": "$alive"}, {"get": [{"var": "$node"}, "_index"]}]}, {"const": 1}]}


def _role_cond(role: str) -> dict:
    return {
        "eq": [
            {"at": [{"var": "$roles"}, {"get": [{"var": "$node"}, "_index"]}]},
            {"const": role},
        ]
    }


def _alive_role_count(role: str) -> dict:
    """Number of alive players of ``role`` (query over the player view)."""
    return {
        "count": {
            "query": {
                "view": "player",
                "filter": {"and": [_alive_cond(), _role_cond(role)]},
            }
        }
    }


class UndercoverRules:
    """Builds the undercover rules dict covering player counts 4..max_players."""

    def __init__(self, players: int = 8, max_players: int = 12, scenario: str = "fruit"):
        if not 4 <= players <= max_players:
            raise ValueError(f"players={players} must satisfy 4 <= players <= max_players={max_players}")
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; declared: {sorted(SCENARIOS)}")
        self.players = players  # 默认(声明)人数 —— variants.player_count
        self.max_players = max_players  # 顶层 players 覆盖的最大座位数
        self.ids = [f"p{i}" for i in range(max_players)]
        self.last_deal = max_players - 1
        self.scenario = scenario
        self.word_of = SCENARIOS[scenario]

    # ── Effectors ──────────────────────────────────────────────────

    def _deal_effector(self, i: int) -> dict:
        """Deal role + word to seat i; advance to the next deal or to describe."""
        branch = {
            "op": "branch",
            "if": {"eq": [{"count": {"var": "$roles"}}, {"var": "$constants.player_count"}]},
            "then": [
                {"op": "setEnv", "key": "phase", "value": {"const": "describe"}},
                {"op": "setEnv", "key": "round", "value": {"const": 1}},
                {"op": "setEnv", "key": "speechIdx", "value": {"const": 0}},
                {"op": "setEnv", "key": "voteIdx", "value": {"const": 0}},
                {"op": "setEnv", "key": "turn", "value": self._living({"const": 0})},
            ],
        }
        if i < self.last_deal:
            # 还没到最大座位数：若实际人数不足，前面的座位会走该分支继续发牌；
            # 到达实际人数后该分支恒为 then（进入 describe）。
            branch["else"] = [
                {"op": "setEnv", "key": "phase", "value": {"const": f"deal_{i + 1}"}},
                {"op": "setEnv", "key": "turn", "value": {"const": self.ids[i + 1]}},
            ]
        ops = [
            {"op": "append", "array": "roles", "value": {"var": "outcome"}},
            {"op": "append", "array": "words", "value": {"at": [{"var": "$constants.word_of"}, {"var": "outcome"}]}},
            {"op": "append", "array": "alive", "value": {"const": 1}},
            branch,
        ]
        return {"description": f"Deal role/word to {self.ids[i]}", "ops": ops}

    def _living(self, idx: dict) -> dict:
        """env.turn value: the idx-th alive player (describe/vote rotation)."""
        return {
            "get": [
                {"at": [{"query": {"view": "player", "filter": _alive_cond()}}, idx]},
                "id",
            ]
        }

    def _kill_player(self, pid_expr: dict) -> dict:
        return {
            "op": "setIndex",
            "array": "alive",
            "at": {"call": ["player_index", pid_expr]},
            "value": {"const": 0},
        }

    def _win_ops(self) -> list[dict]:
        """胜负判定链（在 do_resolve 里按顺序执行；phase==game_over 短路）。"""
        role_of = lambda pid: {"at": [{"var": "$roles"}, {"call": ["player_index", pid]}]}  # noqa: E731
        return [
            # 卧底或白板被投出 → 平民胜
            {
                "op": "branch",
                "if": {
                    "and": [
                        {"neq": [{"var": "$env.eliminated"}, {"const": None}]},
                        {
                            "or": [
                                {"eq": [role_of({"var": "$env.eliminated"}), {"const": "undercover"}]},
                                {"eq": [role_of({"var": "$env.eliminated"}), {"const": "blank"}]},
                            ]
                        },
                    ]
                },
                "then": [
                    {"op": "setEnv", "key": "winner", "value": {"const": "civilian"}},
                    {"op": "setEnv", "key": "phase", "value": {"const": "game_over"}},
                ],
            },
            # 存活 ≤ 3 且白板存活 → 白板胜
            {
                "op": "branch",
                "if": {
                    "and": [
                        {"neq": [{"var": "$env.phase"}, {"const": "game_over"}]},
                        {"lte": [{"call": ["alive_count"]}, {"const": 3}]},
                        {"gt": [{"call": ["blank_alive"]}, {"const": 0}]},
                    ]
                },
                "then": [
                    {"op": "setEnv", "key": "winner", "value": {"const": "blank"}},
                    {"op": "setEnv", "key": "phase", "value": {"const": "game_over"}},
                ],
            },
            # 存活 ≤ 2 → 卧底存活则卧底胜，否则平民胜
            {
                "op": "branch",
                "if": {
                    "and": [
                        {"neq": [{"var": "$env.phase"}, {"const": "game_over"}]},
                        {"lte": [{"call": ["alive_count"]}, {"const": 2}]},
                    ]
                },
                "then": [
                    {
                        "op": "branch",
                        "if": {"gt": [{"call": ["undercover_alive"]}, {"const": 0}]},
                        "then": [
                            {"op": "setEnv", "key": "winner", "value": {"const": "undercover"}},
                            {"op": "setEnv", "key": "phase", "value": {"const": "game_over"}},
                        ],
                        "else": [
                            {"op": "setEnv", "key": "winner", "value": {"const": "civilian"}},
                            {"op": "setEnv", "key": "phase", "value": {"const": "game_over"}},
                        ],
                    }
                ],
            },
        ]

    def _next_round_ops(self) -> list[dict]:
        return [
            {"op": "inc", "key": "round", "by": 1},
            {"op": "setEnv", "key": "speechIdx", "value": {"const": 0}},
            {"op": "setEnv", "key": "voteIdx", "value": {"const": 0}},
            {"op": "setEnv", "key": "phase", "value": {"const": "describe"}},
            {"op": "setEnv", "key": "turn", "value": self._living({"const": 0})},
        ]

    def _effectors(self) -> dict:
        e: dict = {}
        for i in range(self.max_players):
            e[f"do_deal_{i}"] = self._deal_effector(i)

        e["do_speak"] = {
            "description": "Record the description, advance the speaking turn",
            "ops": [
                {
                    "op": "append",
                    "array": "speechLog",
                    "value": {
                        "speaker": {"var": "$env.turn"},
                        "text": {"var": "$text"},
                        "round": {"var": "$env.round"},
                    },
                },
                {"op": "inc", "key": "speechIdx", "by": 1},
                {
                    "op": "branch",
                    "if": {"gte": [{"var": "$env.speechIdx"}, {"call": ["alive_count"]}]},
                    "then": [
                        {"op": "setEnv", "key": "phase", "value": {"const": "vote"}},
                        {"op": "setEnv", "key": "voteIdx", "value": {"const": 0}},
                        {"op": "setEnv", "key": "turn", "value": self._living({"const": 0})},
                    ],
                    "else": [
                        {"op": "setEnv", "key": "turn", "value": self._living({"var": "$env.speechIdx"})},
                    ],
                },
            ],
        }
        e["do_vote"] = {
            "description": "Record the vote, advance the voting turn",
            "ops": [
                {
                    "op": "append",
                    "array": "voteLog",
                    "value": {
                        "voter": {"var": "$env.turn"},
                        "target": {"get": [{"var": "$target"}, "id"]},
                        "round": {"var": "$env.round"},
                    },
                },
                {"op": "inc", "key": "voteIdx", "by": 1},
                {
                    "op": "branch",
                    "if": {"gte": [{"var": "$env.voteIdx"}, {"call": ["alive_count"]}]},
                    "then": [{"op": "setEnv", "key": "phase", "value": {"const": "resolve"}}],
                    "else": [
                        {"op": "setEnv", "key": "turn", "value": self._living({"var": "$env.voteIdx"})},
                    ],
                },
            ],
        }
        e["do_resolve"] = {
            "description": "Eliminate the most-voted player (ties skip), check winner, next round",
            "ops": [
                {
                    "op": "branch",
                    "if": {"neq": [{"call": ["most_voted"]}, {"const": None}]},
                    "then": [
                        {"op": "setEnv", "key": "eliminated", "value": {"call": ["most_voted"]}},
                        {"op": "append", "array": "deathsArr", "value": {"call": ["most_voted"]}},
                        self._kill_player({"call": ["most_voted"]}),
                    ],
                },
                *self._win_ops(),
                {
                    "op": "branch",
                    "if": {"neq": [{"var": "$env.phase"}, {"const": "game_over"}]},
                    "then": self._next_round_ops(),
                },
            ],
        }
        return e

    # ── Actions / chance / phases ──────────────────────────────────

    def _actions(self) -> list[dict]:
        return [
            {
                "id": "speak",
                "type": "speech",
                "phases": ["describe"],
                "actor": {"var": "$env.turn"},
                "params": {"text": {"type": "text"}},
                "legal": {"const": True},
                "effectRef": "do_speak",
                "canonicalKey": {"template": "speak"},
            },
            {
                "id": "vote",
                "type": "vote",
                "phases": ["vote"],
                "actor": {"var": "$env.turn"},
                "params": {"target": {"view": "player", "domain": {"ref": "alive_others"}}},
                "legal": {"const": True},
                "effectRef": "do_vote",
                "canonicalKey": {"template": "vote:{$target.id}"},
            },
        ]

    def _chance(self) -> list[dict]:
        chance = [
            {
                "id": f"deal_{i}",
                "phases": [f"deal_{i}"],
                "params": {"role": {"view": "role", "domain": {"ref": "unassigned_roles"}}},
                "probability": {"uniform": {"over": "role"}},
                "effectRef": f"do_deal_{i}",
                "canonicalKey": {"template": "deal:{outcome}"},
            }
            for i in range(self.max_players)
        ]
        chance.append(
            {
                "id": "resolve",
                "phases": ["resolve"],
                "params": {"resolve": {"view": "resolve", "domain": {"ref": "resolves"}}},
                "probability": {"explicit": [{"outcome": "resolve", "prob": 1.0}]},
                "effectMap": {"resolve": "do_resolve"},
                "canonicalKey": {"template": "resolve:{outcome}"},
            }
        )
        return chance

    def _phases(self) -> list[dict]:
        phases = [
            {"id": f"deal_{i}", "actions": [], "description": f"Deal role/word to {self.ids[i]}"}
            for i in range(self.max_players)
        ]
        phases.extend(
            [
                {"id": "describe", "actions": ["speak"], "description": "存活玩家轮流一句话描述自己的词"},
                {"id": "vote", "actions": ["vote"], "description": "存活玩家轮流投票指认卧底"},
                {"id": "resolve", "actions": [], "description": "投票结算：得票最多者出局"},
                {"id": "game_over", "actions": [], "description": "Game over"},
            ]
        )
        return phases

    def _views_queries(self) -> tuple[dict, dict]:
        views = {
            "player": {
                "from": {"type": "literal", "list": {"var": "player_ids"}},
                "fields": {"id": {"var": "$self.value"}},
            },
            "role": {
                "from": {"type": "literal", "list": {"var": "role_pool"}},
                "fields": {"id": {"var": "$self.value"}},
            },
            "resolve": {
                "from": {"type": "literal", "list": {"var": "resolve_outcomes"}},
                "fields": {"id": {"var": "$self.value"}},
            },
            # v5.2: player-facing views over ground arrays — partial
            # observability declared in ``visibility``; no adapter projection.
            "my_role": {
                "from": {"type": "enum", "array": "roles"},
                "fields": {"role": {"var": "$self.value"}},
            },
            "my_word": {
                "from": {"type": "enum", "array": "words"},
                "fields": {"word": {"var": "$self.value"}},
            },
            "alive": {
                "from": {"type": "enum", "array": "alive"},
                "fields": {"alive": {"var": "$self.value"}},
            },
            "speech_log": {
                "from": {"type": "enum", "array": "speechLog"},
                "fields": {"entry": {"var": "$self.value"}},
            },
            "vote_log": {
                "from": {"type": "enum", "array": "voteLog"},
                "fields": {"entry": {"var": "$self.value"}},
            },
            "deaths_arr": {
                "from": {"type": "enum", "array": "deathsArr"},
                "fields": {"entry": {"var": "$self.value"}},
            },
            # 死后身份/词语公开：visibility 里按 alive==0 保留行（drop）。
            "dead_roles": {
                "from": {"type": "enum", "array": "roles"},
                "fields": {"role": {"var": "$self.value"}},
            },
            "dead_words": {
                "from": {"type": "enum", "array": "words"},
                "fields": {"word": {"var": "$self.value"}},
            },
        }
        alive_filter = {"eq": [{"at": [{"var": "$alive"}, {"get": [{"var": "$node"}, "_index"]}]}, {"const": 1}]}
        queries = {
            "unassigned_roles": {
                "view": "role",
                "filter": {
                    "lt": [
                        _count_eq({"var": "$roles"}, "$r", {"get": [{"var": "$node"}, "id"]}),
                        _count_eq({"var": "$constants.role_pool"}, "$p", {"get": [{"var": "$node"}, "id"]}),
                    ]
                },
            },
            "alive_players": {"view": "player", "filter": alive_filter},
            # 投票目标：除当前投票者外的存活玩家（不能投自己，无弃权）。
            "alive_others": {
                "view": "player",
                "filter": {"and": [alive_filter, {"neq": [{"get": [{"var": "$node"}, "id"]}, {"var": "$env.turn"}]}]},
            },
            "resolves": {"view": "resolve"},
        }
        return views, queries

    def _visibility(self) -> dict:
        """Declarative partial observability (v5.2): 只看自己的词/身份，死者公开。"""
        viewer_role = lambda viewer: {"call": ["player_index", {"var": viewer}]}  # noqa: E731
        return {
            "default": "partial",
            "rules": [
                {
                    "view": "my_role",
                    "drop": True,
                    "filter": {"eq": [{"get": [{"var": "$node"}, "_index"]}, viewer_role("$viewer")]},
                },
                {
                    "view": "my_word",
                    "drop": True,
                    "filter": {"eq": [{"get": [{"var": "$node"}, "_index"]}, viewer_role("$viewer")]},
                },
                {
                    "view": "dead_roles",
                    "drop": True,
                    "filter": {
                        "eq": [{"at": [{"var": "$alive"}, {"get": [{"var": "$node"}, "_index"]}]}, {"const": 0}]
                    },
                },
                {
                    "view": "dead_words",
                    "drop": True,
                    "filter": {
                        "eq": [{"at": [{"var": "$alive"}, {"get": [{"var": "$node"}, "_index"]}]}, {"const": 0}]
                    },
                },
            ],
        }

    def _functions(self) -> dict:
        sorted_votes = {"call": ["this_round_votes"]}
        return {
            # 座位索引 = 首个 player_ids[i] == p 的 i（不能用"比 p 小的 id 数"：
            # 两位 id 的字典序 ≠ 座位序，p10 < p2；werewolf 只有 p0..p8 不暴露）。
            "player_index": {
                "description": "Seat index of player p within player_ids",
                "params": ["p"],
                "expr": {
                    "at": [
                        {
                            "filter": {
                                "list": {
                                    "range": {"from": {"const": 0}, "to": {"count": {"var": "$constants.player_ids"}}}
                                },
                                "as": "$i",
                                "where": {
                                    "eq": [{"at": [{"var": "$constants.player_ids"}, {"var": "$i"}]}, {"var": "$p"}]
                                },
                            }
                        },
                        {"const": 0},
                    ]
                },
            },
            "alive_count": {
                "description": "Number of alive players",
                "params": [],
                "expr": {
                    "count": {
                        "filter": {
                            "list": {"var": "$alive"},
                            "as": "$v",
                            "where": {"eq": [{"var": "$v"}, {"const": 1}]},
                        }
                    }
                },
            },
            "living": {
                "description": "Id of the idx-th alive player (seat order)",
                "params": ["idx"],
                "expr": {
                    "get": [
                        {"at": [{"query": {"view": "player", "filter": _alive_cond()}}, {"var": "$idx"}]},
                        "id",
                    ]
                },
            },
            "civilian_alive": {"params": [], "expr": _alive_role_count("civilian")},
            "undercover_alive": {"params": [], "expr": _alive_role_count("undercover")},
            "blank_alive": {"params": [], "expr": _alive_role_count("blank")},
            "this_round_votes": {
                "description": "Votes of the current round, grouped by target, sorted by count desc",
                "params": [],
                "expr": {
                    "sort": {
                        "list": {
                            "group": {
                                "list": {
                                    "filter": {
                                        "list": {"var": "$voteLog"},
                                        "as": "$v",
                                        "where": {"eq": [{"get": [{"var": "$v"}, "round"]}, {"var": "$env.round"}]},
                                    }
                                },
                                "by": {"get": [{"var": "$item"}, "target"]},
                            }
                        },
                        "by": {"get": [{"var": "$node"}, "count"]},
                        "reverse": True,
                    }
                },
            },
            "most_voted": {
                "description": "Unique most-voted target this round; tie → None (平票无人出局)",
                "params": [],
                "expr": {
                    "if": {
                        "cond": {
                            "eq": [
                                {
                                    "count": {
                                        "filter": {
                                            "list": sorted_votes,
                                            "as": "$g",
                                            "where": {
                                                "eq": [
                                                    {"get": [{"var": "$g"}, "count"]},
                                                    {"get": [{"at": [sorted_votes, {"const": 0}]}, "count"]},
                                                ]
                                            },
                                        }
                                    }
                                },
                                {"const": 1},
                            ]
                        },
                        "then": {"get": [{"at": [sorted_votes, {"const": 0}]}, "key"]},
                        "else": {"const": None},
                    }
                },
            },
        }

    def build(self) -> dict:
        views, queries = self._views_queries()
        env_fields = {
            "phase": {"type": "string", "initial": "deal_0"},
            "turn": {"type": "player_id", "initial": self.ids[0]},
            "round": {"type": "int", "initial": 1},
            "speechIdx": {"type": "int", "initial": 0},
            "voteIdx": {"type": "int", "initial": 0},
            "eliminated": {"type": "string", "initial": None},
            "winner": {"type": "string", "initial": None},
        }
        ground = {
            "roles": {"type": "array", "length": {"expr": "player_count"}, "mutable": True},
            "words": {"type": "array", "length": {"expr": "player_count"}, "mutable": True},
            "alive": {"type": "array", "length": {"expr": "player_count"}, "mutable": True},
            "speechLog": {"type": "array", "mutable": True},
            "voteLog": {"type": "array", "mutable": True},
            "deathsArr": {"type": "array", "mutable": True},
            "env": {"type": "env", "fields": env_fields},
        }
        utility = []
        for pid in self.ids:
            role_at = {"at": [{"var": "$roles"}, {"call": ["player_index", {"const": pid}]}]}
            for win_role in ("civilian", "undercover", "blank"):
                # 胜者阵营 +1，其余阵营 -1（三阵营两两互斥）。
                utility.append(
                    {
                        "player": pid,
                        "value": 1,
                        "when": {
                            "and": [
                                {"eq": [{"get": ["$env", "winner"]}, win_role]},
                                {"eq": [role_at, {"const": win_role}]},
                            ]
                        },
                    }
                )
                utility.append(
                    {
                        "player": pid,
                        "value": -1,
                        "when": {
                            "and": [
                                {"eq": [{"get": ["$env", "winner"]}, win_role]},
                                {"neq": [role_at, {"const": win_role}]},
                            ]
                        },
                    }
                )
        return {
            "meta": {
                "name": "undercover",
                "version": "5.2",
                "description": (
                    f"谁是卧底 {self.players}-player (default) 1卧底+1白板+平民 "
                    f"scenario={self.scenario}（词对: {self.word_of['civilian']}/{self.word_of['undercover']}）"
                ),
            },
            "players": [{"id": pid, "type": "player"} for pid in self.ids],
            # v5.2: 场景(词对)与人数纯数据声明——``options[<scenario>].constants``
            # 补丁 + ``$player_count`` 上下文公式；引擎不做任何注入。
            "variants": {
                "variant": self.scenario,
                "player_count": self.players,
                "options": {name: {"constants": {"word_of": words}} for name, words in SCENARIOS.items()},
                "player_ids": {
                    "map": {
                        "list": {"range": {"from": {"const": 0}, "to": {"var": "$player_count"}}},
                        "as": "$node",
                        "expr": {"template": "p{$node}"},
                    }
                },
                "role_pool": {
                    "concat": [
                        {"const": ["undercover", "blank"]},
                        {
                            "map": {
                                "list": {
                                    "range": {
                                        "from": {"const": 0},
                                        "to": {"sub": [{"var": "$player_count"}, {"const": 2}]},
                                    }
                                },
                                "as": "$n",
                                "expr": {"const": "civilian"},
                            }
                        },
                    ]
                },
                "trim_players": True,
                "trim_utility": True,
            },
            "groundState": ground,
            "derivedViews": views,
            "constants": {
                "word_of": SCENARIOS[self.scenario],
                "resolve_outcomes": ["resolve"],
            },
            "queries": queries,
            "functions": self._functions(),
            "actions": self._actions(),
            "chance": self._chance(),
            "effectors": self._effectors(),
            "phases": self._phases(),
            "visibility": self._visibility(),
            "terminal": [
                {"id": "winner_declared", "condition": {"neq": [{"get": ["$env", "winner"]}, {"const": None}]}},
                # 轮次上限（法官干预）：超限判平局（winner 保持 None）
                {
                    "id": "max_rounds",
                    "condition": {
                        "gte": [
                            {"get": ["$env", "round"]},
                            {"add": [{"var": "$constants.player_count"}, {"const": 8}]},
                        ]
                    },
                },
            ],
            "utility": utility,
        }


def gen_rules(players: int = 8, max_players: int = 12, scenario: str = "fruit") -> dict:
    """Generate the undercover rules dict (see UndercoverRules docstring)."""
    return UndercoverRules(players=players, max_players=max_players, scenario=scenario).build()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rules/undercover.json")
    parser.add_argument("--players", type=int, default=8, help="默认（声明）人数，4..max_players")
    parser.add_argument(
        "--max-players", type=int, default=12, help="JSON 覆盖的最大座位数（variants 可选的玩家数上限）"
    )
    parser.add_argument("--scenario", type=str, default="fruit", choices=sorted(SCENARIOS), help="默认词对场景")
    parser.add_argument("--out", type=str, default="rules/undercover.json")
    args = parser.parse_args()

    rules = gen_rules(args.players, args.max_players, args.scenario)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}  ({rules['meta']['description']})")


if __name__ == "__main__":
    main()
