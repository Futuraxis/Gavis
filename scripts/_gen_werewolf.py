#!/usr/bin/env python3
"""Werewolf rules generator — generates ``rules/werewolf.json`` (v5.1).

Generates the standard Werewolf rule set with configurable composition:

    python _gen_werewolf.py [--players 9] [--wolves 3] [--seers 1]
                            [--with-witch] [--with-hunter] [--with-guard]
                            [--win-mode side|total] [--first-night-protect]
                            [--no-self-save] [--no-hunter-lynch]

Roles: wolf / villager / seer(预言家) / witch(女巫) / hunter(猎人) / guard(守卫)

Standard presets:
  6 人: 2狼 1预 1女巫 2民 (屠边)
  9 人: 3狼 1预 1女巫 1猎人 3民 (屠边)
  12人: 4狼 1预 1女巫 1守卫 1猎人 4民 (屠边)

Variant switches:
  - ``win_mode``: "side" 屠边 (民全灭或神全灭→狼赢) / "total" 屠城
    (好人全灭→狼赢)
  - ``first_night_protect``: 首夜预言家不被狼刀杀死
  - ``witch_self_save``: 女巫能否自救 (默认不能)
  - ``hunter_shoots_when_lynched``: 猎人被放逐是否开枪 (默认开)

Design notes
------------
- 部分可观测：身份 / 夜晚行动由 ``WerewolfAdapter`` 按玩家过滤（规则层声明
  完整状态，adapter 只做投影）。
- 结算阶段（夜晚死亡、放逐、猎人开枪）用 ``chance``/``player`` 阶段表达；
  interpreter 用 ``effectMap`` 解析 explicit 条目的 effector（非 effectRef）。
- ``speak`` 动作使用 v5.1 的 ``text`` 自由文本参数预制能力。
- mutable 数组初始为空 → 发牌/死亡名单用 ``append``（数组索引即玩家索引）。
- 平票放逐：得票最高者被放逐，平票取最早被投票的目标（group 保序）。
- 夜晚顺序: 狼人 → 守卫 → 女巫 → 预言家 → 结算(死亡+猎人开枪)。
- 结算链: 狼刀(首夜保护/守卫/女巫救可免) + 女巫毒 + 猎人死亡开枪带人。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INTENTS = ["claim", "accuse", "defend", "question", "persuade"]

ROLES = ("wolf", "villager", "seer", "witch", "hunter", "guard")
ROLE_LABEL = {"wolf": "狼人", "villager": "村民", "seer": "预言家", "witch": "女巫", "hunter": "猎人", "guard": "守卫"}


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


def _alive_roles_any(*roles: str) -> dict:
    """Number of alive players whose role ∈ roles (gods = non-wolf non-villager)."""
    conds = []
    for r in roles:
        conds.append({"and": [_alive_cond(), _role_cond(r)]})
    return {
        "count": {
            "query": {
                "view": "player",
                "filter": conds[0] if len(conds) == 1 else {"or": conds},
            }
        }
    }


class WerewolfRules:
    """Builds the werewolf rules dict for one composition."""

    def __init__(
        self,
        players: int = 9,
        wolves: int = 3,
        seers: int = 1,
        with_witch: bool = True,
        with_hunter: bool = True,
        with_guard: bool = False,
        win_mode: str = "side",
        first_night_protect: bool = True,
        witch_self_save: bool = False,
        hunter_shoots_when_lynched: bool = True,
    ):
        role_list = ["wolf"] * wolves + ["villager"] * 1  # placeholder, fixed below
        self.players = players
        self.ids = [f"p{i}" for i in range(players)]
        extras = [
            r
            for r, on in (("seer", seers > 0), ("witch", with_witch), ("hunter", with_hunter), ("guard", with_guard))
            if on
        ]
        self.extras = extras
        villagers = players - wolves - len(extras)
        if wolves < 1 or villagers < 1:
            raise ValueError(f"players={players} wolves={wolves} extras={len(extras)} leaves {villagers} villagers")
        self.role_pool = ["wolf"] * wolves + ["villager"] * villagers + extras
        self.win_mode = win_mode
        self.first_night_protect = first_night_protect
        self.witch_self_save = witch_self_save
        self.hunter_shoots_when_lynched = hunter_shoots_when_lynched
        self.has = {r: r in extras or r in ("wolf", "villager") for r in ROLES}
        self.has["seer"] = seers > 0
        self.roles_in_night = [r for r in ("wolf", "guard", "witch", "seer") if self.has[r]]
        self.last_deal = players - 1
        self.max_rounds = max(6, players * 2)  # 法官干预：超限判平局（屠城长局兜底）

    # ── Effectors ──────────────────────────────────────────────────

    def _deal_effector(self, i: int) -> dict:
        ops = [
            {"op": "append", "array": "roles", "value": {"var": "outcome"}},
            {"op": "append", "array": "alive", "value": {"const": 1}},
        ]
        if i < self.last_deal:
            ops.append({"op": "setEnv", "key": "phase", "value": {"const": f"deal_{i + 1}"}})
            ops.append({"op": "setEnv", "key": "turn", "value": {"const": self.ids[i + 1]}})
        else:
            ops.append({"op": "setEnv", "key": "phase", "value": {"const": "night_wolf"}})
            ops.append({"op": "setEnv", "key": "round", "value": {"const": 1}})
            # 首个存活的狼（座位号最小的狼人）持刀
            ops.append({"op": "setEnv", "key": "turn", "value": self._role_holder("wolf")})
        return {"description": f"Deal role to {self.ids[i]}", "ops": ops}

    def _role_holder(self, role: str, alive: bool = True) -> dict:
        """env.turn value: the (first) player holding ``role``.

        ``alive=True`` filters to living players (night role phases);
        ``alive=False`` returns the first holder regardless of state --
        hunter shooting phases act for the already-dead hunter.
        """
        cond = {"and": [_alive_cond(), _role_cond(role)]} if alive else _role_cond(role)
        return {"get": [{"at": [{"query": {"view": "player", "filter": cond}}, {"const": 0}]}, "id"]}

    def _living(self, idx: dict) -> dict:
        """env.turn value: the idx-th alive player (day rotation)."""
        return {
            "get": [
                {"at": [{"query": {"view": "player", "filter": _alive_cond()}}, idx]},
                "id",
            ]
        }

    def _night_chain(self, after: str) -> list[dict]:
        """Branch chain: enter the next live night role's phase in order.

        Nested ``if/else`` gives elif semantics: the first role whose entry
        condition holds wins and later roles never overwrite ``phase``
        (the old sequential branches let the seer branch clobber
        ``night_witch``).  Visited order: guard → witch → seer.
        """
        entry = f"night_{after}"
        roles = [r for r in ("guard", "witch", "seer") if self.has[r] and r != after]
        # Fallback: still at the chain's entry phase → the night is over.
        ops: list[dict] = [
            {
                "op": "branch",
                "if": {"eq": [{"var": "$env.phase"}, {"const": entry}]},
                "then": [{"op": "setEnv", "key": "phase", "value": {"const": "night_end"}}],
            }
        ]
        for role in reversed(roles):
            if role == "witch":
                # 女巫入场：存活 且 至少有一瓶药能用。
                # 药可用 ≠ 入场充分：被刀的女巫在自救关闭时不能救自己，
                # 若毒也用过则没有任何合法动作 → 该夜直接跳过女巫阶段。
                heal_ok = {
                    "and": [
                        {"eq": [{"var": "$env.witchSaveUsed"}, {"const": 0}]},
                        {"neq": [{"var": "$env.nightKill"}, {"const": None}]},
                        {
                            "or": [
                                {"eq": [{"var": "$constants.witch_self_save"}, {"const": 1}]},
                                {"neq": [{"var": "$env.nightKill"}, self._role_holder("witch")]},
                            ]
                        },
                    ]
                }
                cond = {
                    "and": [
                        {"gt": [{"call": ["witch_alive"]}, {"const": 0}]},
                        {"or": [heal_ok, {"eq": [{"var": "$env.witchPoisonUsed"}, {"const": 0}]}]},
                    ]
                }
            else:
                cond = {"gt": [{"call": [f"{role}_alive"]}, {"const": 0}]}
            ops = [
                {
                    "op": "branch",
                    "if": cond,
                    "then": [
                        {"op": "setEnv", "key": "phase", "value": {"const": f"night_{role}"}},
                        {"op": "setEnv", "key": "turn", "value": self._role_holder(role)},
                    ],
                    "else": ops,
                }
            ]
        return ops

    def _win_checks(self) -> list[dict]:
        """胜负判定。狼赢条件按 win_mode：side = 民或神全灭；total = 好人全灭。"""
        if self.win_mode == "side":
            wolf_win = {
                "op": "branch",
                "if": {
                    "or": [
                        {"eq": [{"call": ["alive_villagers"]}, {"const": 0}]},
                        {"eq": [{"call": ["alive_gods"]}, {"const": 0}]},
                    ]
                },
                "then": [
                    {"op": "setEnv", "key": "winner", "value": {"const": "wolf"}},
                    {"op": "setEnv", "key": "phase", "value": {"const": "game_over"}},
                ],
            }
        else:  # total 屠城：好人(非狼)全灭
            wolf_win = {
                "op": "branch",
                "if": {"eq": [{"call": ["alive_good"]}, {"const": 0}]},
                "then": [
                    {"op": "setEnv", "key": "winner", "value": {"const": "wolf"}},
                    {"op": "setEnv", "key": "phase", "value": {"const": "game_over"}},
                ],
            }
        good_win = {
            "op": "branch",
            "if": {"eq": [{"call": ["alive_wolves"]}, {"const": 0}]},
            "then": [
                {"op": "setEnv", "key": "winner", "value": {"const": "good"}},
                {"op": "setEnv", "key": "phase", "value": {"const": "game_over"}},
            ],
        }
        return [good_win, wolf_win]

    def _reset_night(self) -> list[dict]:
        return [
            {"op": "setEnv", "key": "nightKill", "value": {"const": None}},
            {"op": "setEnv", "key": "poisonTarget", "value": {"const": None}},
            {"op": "setEnv", "key": "guardTarget", "value": {"const": None}},
            {"op": "setEnv", "key": "witchSavedTarget", "value": {"const": None}},
            {"op": "setEnv", "key": "hunterShoot", "value": {"const": None}},
            {"op": "setEnv", "key": "lynched", "value": {"const": None}},
            {"op": "setEnv", "key": "speechIdx", "value": {"const": 0}},
            {"op": "setEnv", "key": "voteIdx", "value": {"const": 0}},
            {"op": "inc", "key": "round", "by": 1},
            {"op": "setEnv", "key": "phase", "value": {"const": "night_wolf"}},
            {"op": "setEnv", "key": "turn", "value": self._role_holder("wolf")},
        ]

    def _kill_player(self, pid_expr: dict) -> dict:
        return {
            "op": "setIndex",
            "array": "alive",
            "at": {"call": ["player_index", pid_expr]},
            "value": {"const": 0},
        }

    def _effectors(self) -> dict:
        e: dict = {}
        for i in range(self.players):
            e[f"do_deal_{i}"] = self._deal_effector(i)

        # 狼人杀：记录目标，进入守卫/女巫/预言家
        chain = self._night_chain("wolf")
        e["do_kill"] = {
            "description": "Wolves kill one player",
            "ops": [
                {"op": "setEnv", "key": "nightKill", "value": {"get": [{"var": "$target"}, "id"]}},
                *chain,
            ],
        }

        # 守卫：记录守人目标 → 下一夜晚角色
        if self.has["guard"]:
            e["do_guard"] = {
                "description": "Guard protects one player (no consecutive nights)",
                "ops": [
                    {"op": "setEnv", "key": "guardTarget", "value": {"get": [{"var": "$target"}, "id"]}},
                    *self._night_chain("guard"),
                ],
            }

        # 女巫：救（目标=当夜狼刀）或毒 → 下一夜晚角色
        if self.has["witch"]:
            e["do_heal"] = {
                "description": "Witch saves the night-killed player",
                "ops": [
                    {"op": "setEnv", "key": "witchSavedTarget", "value": {"var": "$env.nightKill"}},
                    {"op": "setEnv", "key": "witchSaveUsed", "value": {"const": 1}},
                    *self._night_chain("witch"),
                ],
            }
            e["do_poison"] = {
                "description": "Witch poisons one player",
                "ops": [
                    {"op": "setEnv", "key": "poisonTarget", "value": {"get": [{"var": "$target"}, "id"]}},
                    {"op": "setEnv", "key": "witchPoisonUsed", "value": {"const": 1}},
                    *self._night_chain("witch"),
                ],
            }

        # 预言家
        if self.has["seer"]:
            e["do_check"] = {
                "description": "Seer checks one player's role",
                "ops": [
                    {
                        "op": "setEnv",
                        "key": "seerResult",
                        "value": {
                            "at": [
                                {"var": "$roles"},
                                {"call": ["player_index", {"get": [{"var": "$target"}, "id"]}]},
                            ]
                        },
                    },
                    {"op": "setEnv", "key": "phase", "value": {"const": "night_end"}},
                ],
            }

        # 夜晚结算：死亡链 + 胜负 + 进入白天
        # 1) 狼刀（首夜保护 / 守卫 / 女巫救可免）→ deathsArr
        night_ops: list[dict] = []
        wolf_killed = {"var": "$env.nightKill"}
        if self.first_night_protect:
            night_ops.append(
                {
                    "op": "branch",
                    "if": {
                        "and": [
                            {"eq": [{"var": "$env.round"}, {"const": 1}]},
                            {
                                "eq": [
                                    {"at": [{"var": "$roles"}, {"call": ["player_index", wolf_killed]}]},
                                    {"const": "seer"},
                                ]
                            },
                        ]
                    },
                    "then": [{"op": "setEnv", "key": "nightKill", "value": {"const": None}}],
                }
            )
        night_ops.append(
            {
                "op": "branch",
                "if": {
                    "and": [
                        {"neq": [{"var": "$env.nightKill"}, {"const": None}]},
                        {"neq": [{"var": "$env.nightKill"}, {"var": "$env.guardTarget"}]},
                        {"neq": [{"var": "$env.nightKill"}, {"var": "$env.witchSavedTarget"}]},
                    ]
                },
                "then": [
                    {"op": "append", "array": "deathsArr", "value": {"var": "$env.nightKill"}},
                    self._kill_player({"var": "$env.nightKill"}),
                ],
            }
        )
        # 2) 女巫毒
        if self.has["witch"]:
            night_ops.append(
                {
                    "op": "branch",
                    "if": {"neq": [{"var": "$env.poisonTarget"}, {"const": None}]},
                    "then": [
                        {"op": "append", "array": "deathsArr", "value": {"var": "$env.poisonTarget"}},
                        self._kill_player({"var": "$env.poisonTarget"}),
                    ],
                }
            )
        # 3) 守卫目标轮换（守过的不能连守）
        if self.has["guard"]:
            night_ops.append({"op": "setEnv", "key": "guardLastTarget", "value": {"var": "$env.guardTarget"}})
        # 4) 猎人被夜杀 → 进入开枪阶段
        if self.has["hunter"]:
            night_ops.append(
                {
                    "op": "branch",
                    "if": {
                        "gt": [
                            {
                                "count": {
                                    "filter": {
                                        "list": {"var": "$deathsArr"},
                                        "as": "$d",
                                        "where": {
                                            "eq": [
                                                {"at": [{"var": "$roles"}, {"call": ["player_index", {"var": "$d"}]}]},
                                                {"const": "hunter"},
                                            ]
                                        },
                                    }
                                }
                            },
                            {"const": 0},
                        ]
                    },
                    "then": [
                        {"op": "setEnv", "key": "phase", "value": {"const": "night_hunter"}},
                        # 猎人此时已死亡：turn 必须是已死的猎人本人
                        {"op": "setEnv", "key": "turn", "value": self._role_holder("hunter", alive=False)},
                    ],
                }
            )
        # 5) 非猎人分支：胜负 → 白天（公布 deathsArr）
        night_ops.append(
            {
                "op": "branch",
                "if": {"neq": [{"var": "$env.phase"}, {"const": "night_hunter"}]},
                "then": [
                    *self._win_checks(),
                    {
                        "op": "branch",
                        "if": {"neq": [{"var": "$env.phase"}, {"const": "game_over"}]},
                        "then": [
                            {"op": "setEnv", "key": "nightKill", "value": {"const": None}},
                            {"op": "setEnv", "key": "poisonTarget", "value": {"const": None}},
                            {"op": "setEnv", "key": "guardTarget", "value": {"const": None}},
                            {"op": "setEnv", "key": "witchSavedTarget", "value": {"const": None}},
                            {"op": "setEnv", "key": "speechIdx", "value": {"const": 0}},
                            {"op": "setEnv", "key": "phase", "value": {"const": "day_speech"}},
                            {"op": "setEnv", "key": "turn", "value": self._living({"const": 0})},
                        ],
                    },
                ],
            }
        )
        e["do_night_end"] = {"description": "Settle the night: deaths, hunter, winner", "ops": night_ops}

        # 猎人开枪（夜晚死亡触发）
        if self.has["hunter"]:
            e["do_hunter_shoot"] = {
                "description": "Hunter shoots one player (or passes)",
                "ops": [
                    {
                        "op": "branch",
                        "if": {"neq": [{"get": [{"var": "$target"}, "id"]}, {"const": "pass"}]},
                        "then": [
                            {"op": "setEnv", "key": "hunterShoot", "value": {"get": [{"var": "$target"}, "id"]}},
                            {"op": "append", "array": "deathsArr", "value": {"get": [{"var": "$target"}, "id"]}},
                            self._kill_player({"get": [{"var": "$target"}, "id"]}),
                        ],
                    },
                    *self._win_checks(),
                    {
                        "op": "branch",
                        "if": {"neq": [{"var": "$env.phase"}, {"const": "game_over"}]},
                        "then": [
                            {"op": "setEnv", "key": "nightKill", "value": {"const": None}},
                            {"op": "setEnv", "key": "poisonTarget", "value": {"const": None}},
                            {"op": "setEnv", "key": "guardTarget", "value": {"const": None}},
                            {"op": "setEnv", "key": "witchSavedTarget", "value": {"const": None}},
                            {"op": "setEnv", "key": "speechIdx", "value": {"const": 0}},
                            {"op": "setEnv", "key": "phase", "value": {"const": "day_speech"}},
                            {"op": "setEnv", "key": "turn", "value": self._living({"const": 0})},
                        ],
                    },
                ],
            }

        # 发言 / 投票
        e["do_speak"] = {
            "description": "Record the speech, advance the speaking turn",
            "ops": [
                {
                    "op": "append",
                    "array": "speechLog",
                    "value": {
                        "speaker": {"var": "$env.turn"},
                        "intent": {"get": [{"var": "$intent"}, "id"]},
                        "text": {"var": "$text"},
                        "round": {"var": "$env.round"},
                    },
                },
                {"op": "inc", "key": "speechIdx", "by": 1},
                {
                    "op": "branch",
                    "if": {"gte": [{"var": "$env.speechIdx"}, {"call": ["alive_count"]}]},
                    "then": [
                        {"op": "setEnv", "key": "phase", "value": {"const": "day_vote"}},
                        {"op": "setEnv", "key": "voteIdx", "value": {"const": 0}},
                        {"op": "setEnv", "key": "turn", "value": self._living({"const": 0})},
                    ],
                    "else": [
                        # 规则层推进发言轮转：turn 指向下一位存活玩家
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
                    "then": [{"op": "setEnv", "key": "phase", "value": {"const": "vote_resolve"}}],
                    "else": [
                        {"op": "setEnv", "key": "turn", "value": self._living({"var": "$env.voteIdx"})},
                    ],
                },
            ],
        }

        # 放逐结算
        vote_ops = [
            {
                "op": "branch",
                "if": {"neq": [{"call": ["most_voted"]}, {"const": None}]},
                "then": [
                    {"op": "setEnv", "key": "lynched", "value": {"call": ["most_voted"]}},
                    {"op": "append", "array": "deathsArr", "value": {"call": ["most_voted"]}},
                    self._kill_player({"call": ["most_voted"]}),
                ],
            },
        ]
        if self.has["hunter"] and self.hunter_shoots_when_lynched:
            vote_ops.append(
                {
                    "op": "branch",
                    "if": {
                        "and": [
                            {"neq": [{"var": "$env.lynched"}, {"const": None}]},
                            {
                                "eq": [
                                    {"at": [{"var": "$roles"}, {"call": ["player_index", {"var": "$env.lynched"}]}]},
                                    {"const": "hunter"},
                                ]
                            },
                        ]
                    },
                    "then": [
                        {"op": "setEnv", "key": "phase", "value": {"const": "vote_hunter"}},
                        # 被放逐的猎人已死亡：turn 必须是猎人本人
                        {"op": "setEnv", "key": "turn", "value": self._role_holder("hunter", alive=False)},
                    ],
                }
            )
        vote_ops.append(
            {
                "op": "branch",
                "if": {"neq": [{"var": "$env.phase"}, {"const": "vote_hunter"}]},
                "then": [
                    *self._win_checks(),
                    {
                        "op": "branch",
                        "if": {"neq": [{"var": "$env.phase"}, {"const": "game_over"}]},
                        "then": self._reset_night(),
                    },
                ],
            }
        )
        e["do_vote_resolve"] = {"description": "Eliminate the most-voted player", "ops": vote_ops}

        if self.has["hunter"] and self.hunter_shoots_when_lynched:
            e["do_vote_hunter_shoot"] = {
                "description": "Hunter shot after being lynched",
                "ops": [
                    {
                        "op": "branch",
                        "if": {"neq": [{"get": [{"var": "$target"}, "id"]}, {"const": "pass"}]},
                        "then": [
                            {"op": "append", "array": "deathsArr", "value": {"get": [{"var": "$target"}, "id"]}},
                            self._kill_player({"get": [{"var": "$target"}, "id"]}),
                        ],
                    },
                    *self._win_checks(),
                    {
                        "op": "branch",
                        "if": {"neq": [{"var": "$env.phase"}, {"const": "game_over"}]},
                        "then": self._reset_night(),
                    },
                ],
            }
        return e

    # ── Actions / chance / phases ──────────────────────────────────

    def _actions(self) -> list[dict]:
        def target_param() -> dict:
            return {"target": {"view": "player", "domain": {"ref": "alive_players"}}}

        actions = [
            {
                "id": "kill",
                "type": "night",
                "phases": ["night_wolf"],
                "actor": {"var": "$env.turn"},
                "params": target_param(),
                "legal": {"const": True},
                "effectRef": "do_kill",
                "canonicalKey": {"template": "kill:{$target.id}"},
            },
        ]
        if self.has["guard"]:
            actions.append(
                {
                    "id": "guard",
                    "type": "night",
                    "phases": ["night_guard"],
                    "actor": {"var": "$env.turn"},
                    "params": target_param(),
                    "legal": {"neq": [{"get": [{"var": "$target"}, "id"]}, {"var": "$env.guardLastTarget"}]},
                    "effectRef": "do_guard",
                    "canonicalKey": {"template": "guard:{$target.id}"},
                }
            )
        if self.has["witch"]:
            actions.append(
                {
                    "id": "heal",
                    "type": "night",
                    "phases": ["night_witch"],
                    "actor": {"var": "$env.turn"},
                    "params": {"target": {"view": "player", "domain": {"expr": [{"var": "$env.nightKill"}]}}},
                    "legal": {
                        "and": [
                            {"eq": [{"var": "$env.witchSaveUsed"}, {"const": 0}]},
                            {"neq": [{"var": "$env.nightKill"}, {"const": None}]},
                            # 自救仅当 witch_self_save 开启（默认 false：刀中女巫不可自救）
                            {
                                "or": [
                                    {"eq": [{"var": "$constants.witch_self_save"}, {"const": 1}]},
                                    {"neq": [{"var": "$env.nightKill"}, {"var": "$env.turn"}]},
                                ]
                            },
                        ]
                    },
                    "effectRef": "do_heal",
                    "canonicalKey": {"template": "heal:{$target}"},
                }
            )
            actions.append(
                {
                    "id": "poison",
                    "type": "night",
                    "phases": ["night_witch"],
                    "actor": {"var": "$env.turn"},
                    "params": target_param(),
                    "legal": {"eq": [{"var": "$env.witchPoisonUsed"}, {"const": 0}]},
                    "effectRef": "do_poison",
                    "canonicalKey": {"template": "poison:{$target.id}"},
                }
            )
        if self.has["seer"]:
            actions.append(
                {
                    "id": "check",
                    "type": "night",
                    "phases": ["night_seer"],
                    "actor": {"var": "$env.turn"},
                    "params": target_param(),
                    "legal": {"const": True},
                    "effectRef": "do_check",
                    "canonicalKey": {"template": "check:{$target.id}"},
                }
            )
        if self.has["hunter"]:
            # 夜晚死亡开枪 / 放逐死亡开枪：两个 effector（结算去向不同）
            actions.append(
                {
                    "id": "shoot",
                    "type": "night",
                    "phases": ["night_hunter"],
                    "actor": {"var": "$env.turn"},
                    "params": {"target": {"view": "shoot_target", "domain": {"ref": "shoot_targets"}}},
                    "legal": {"const": True},
                    "effectRef": "do_hunter_shoot",
                    "canonicalKey": {"template": "shoot:{$target.id}"},
                }
            )
            if self.hunter_shoots_when_lynched:
                actions.append(
                    {
                        "id": "shoot_lynched",
                        "type": "night",
                        "phases": ["vote_hunter"],
                        "actor": {"var": "$env.turn"},
                        "params": {"target": {"view": "shoot_target", "domain": {"ref": "shoot_targets"}}},
                        "legal": {"const": True},
                        "effectRef": "do_vote_hunter_shoot",
                        "canonicalKey": {"template": "shoot:{$target.id}"},
                    }
                )
        actions.append(
            {
                "id": "speak",
                "type": "speech",
                "phases": ["day_speech"],
                "actor": {"var": "$env.turn"},
                "params": {"intent": {"view": "intent", "domain": {"ref": "intents"}}, "text": {"type": "text"}},
                "legal": {"const": True},
                "effectRef": "do_speak",
                "canonicalKey": {"template": "speak:{$intent.id}"},
            }
        )
        actions.append(
            {
                "id": "vote",
                "type": "vote",
                "phases": ["day_vote"],
                "actor": {"var": "$env.turn"},
                "params": target_param(),
                "legal": {"const": True},
                "effectRef": "do_vote",
                "canonicalKey": {"template": "vote:{$target.id}"},
            }
        )
        return actions

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
            for i in range(self.players)
        ]
        chance.append(
            {
                "id": "night_end",
                "phases": ["night_end"],
                "params": {"resolve": {"view": "resolve", "domain": {"ref": "resolves"}}},
                "probability": {"explicit": [{"outcome": "resolve", "prob": 1.0}]},
                "effectMap": {"resolve": "do_night_end"},
                "canonicalKey": {"template": "night:{outcome}"},
            }
        )
        chance.append(
            {
                "id": "vote_resolve",
                "phases": ["vote_resolve"],
                "params": {"resolve": {"view": "resolve", "domain": {"ref": "resolves"}}},
                "probability": {"explicit": [{"outcome": "resolve", "prob": 1.0}]},
                "effectMap": {"resolve": "do_vote_resolve"},
                "canonicalKey": {"template": "resolve:{outcome}"},
            }
        )
        return chance

    def _phases(self) -> list[dict]:
        phases = [
            {"id": f"deal_{i}", "actions": [], "description": f"Deal role to {self.ids[i]}"}
            for i in range(self.players)
        ] + [
            {"id": "night_wolf", "actions": ["kill"], "description": "狼人杀人"},
        ]
        if self.has["guard"]:
            phases.append({"id": "night_guard", "actions": ["guard"], "description": "守卫守人"})
        if self.has["witch"]:
            phases.append({"id": "night_witch", "actions": ["heal", "poison"], "description": "女巫救/毒"})
        if self.has["seer"]:
            phases.append({"id": "night_seer", "actions": ["check"], "description": "预言家验人"})
        phases.append({"id": "night_end", "actions": [], "description": "夜晚结算"})
        if self.has["hunter"]:
            phases.append({"id": "night_hunter", "actions": ["shoot"], "description": "猎人开枪"})
        phases.extend(
            [
                {"id": "day_speech", "actions": ["speak"], "description": "存活玩家轮流发言"},
                {"id": "day_vote", "actions": ["vote"], "description": "存活玩家轮流投票"},
                {"id": "vote_resolve", "actions": [], "description": "放逐结算"},
            ]
        )
        if self.has["hunter"] and self.hunter_shoots_when_lynched:
            phases.append({"id": "vote_hunter", "actions": ["shoot"], "description": "猎人被放逐开枪"})
        phases.append({"id": "game_over", "actions": [], "description": "Game over"})
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
            "intent": {
                "from": {"type": "literal", "list": {"var": "intents"}},
                "fields": {"id": {"var": "$self.value"}},
            },
            "resolve": {
                "from": {"type": "literal", "list": {"var": "resolve_outcomes"}},
                "fields": {"id": {"var": "$self.value"}},
            },
        }
        if self.has["hunter"]:
            views["shoot_target"] = {
                "from": {"type": "literal", "list": {"var": "shoot_targets"}},
                "fields": {"id": {"var": "$self.value"}},
            }
        # v5.2: player-facing views over ground arrays — partial observability
        # is declared in ``visibility`` (my_role filtered per viewer; the rest
        # public), so no adapter projection is needed.
        views.update(
            {
                "my_role": {
                    "from": {"type": "enum", "array": "roles"},
                    "fields": {"role": {"var": "$self.value"}},
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
                # v5.3: 死后身份公开 — dead_roles 只保留死亡玩家的角色行
                # （visibility 里按 alive==1 drop），消费者用 _index → pid 映射。
                "dead_roles": {
                    "from": {"type": "enum", "array": "roles"},
                    "fields": {"role": {"var": "$self.value"}},
                },
            }
        )
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
            "intents": {"view": "intent"},
            "resolves": {"view": "resolve"},
        }
        if self.has["hunter"]:
            queries["shoot_targets"] = {"view": "shoot_target"}
        return views, queries

    def _visibility(self) -> dict:
        """Declarative partial observability (v5.2).

        - ``default: partial``: every view goes through per-viewer filtering;
          views without a rule pass through fully (public).
        - ``my_role``: only the viewer's own row survives the filter.
        - ``env.seerResult``: only the seer role sees the check result.
        """
        viewer_role = lambda viewer: {"call": ["player_index", {"var": viewer}]}
        return {
            "default": "partial",
            "rules": [
                {
                    "view": "my_role",
                    # ``drop`` (v5.2): a failing filter removes the whole row,
                    # so each viewer sees exactly their own role row.
                    "drop": True,
                    "filter": {
                        "eq": [{"get": [{"var": "$node"}, "_index"]}, viewer_role("$viewer")],
                    },
                },
                {
                    # 死后身份公开：filter 命中 → 保留行；未命中且 drop=True
                    # → 整行删除。alive==0（死亡）命中保留，alive==1 被删。
                    "view": "dead_roles",
                    "drop": True,
                    "filter": {
                        "eq": [{"at": [{"var": "$alive"}, {"get": [{"var": "$node"}, "_index"]}]}, {"const": 0}],
                    },
                }
            ],
            "env": {
                "seerResult": {
                    "filter": {
                        "eq": [{"at": [{"var": "$roles"}, viewer_role("$viewer")]}, {"const": "seer"}],
                    }
                }
            },
        }

    def _functions(self) -> dict:
        fns = {
            "player_index": {
                "description": "Index of player p within player_ids (count of smaller ids)",
                "params": ["p"],
                "expr": {
                    "count": {
                        "filter": {
                            "list": {"var": "$constants.player_ids"},
                            "as": "$n",
                            "where": {"lt": [{"var": "$n"}, {"var": "$p"}]},
                        }
                    }
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
            "alive_wolves": {"params": [], "expr": _alive_role_count("wolf")},
            "alive_villagers": {"params": [], "expr": _alive_role_count("villager")},
            # 神 = seer/witch/hunter/guard（非狼非民）
            "alive_gods": {"params": [], "expr": _alive_roles_any("seer", "witch", "hunter", "guard")},
            "alive_good": {
                "description": "Alive non-wolves (used by total win mode)",
                "params": [],
                "expr": {
                    "count": {
                        "query": {
                            "view": "player",
                            "filter": {
                                "and": [
                                    _alive_cond(),
                                    {
                                        "neq": [
                                            {"at": [{"var": "$roles"}, {"get": [{"var": "$node"}, "_index"]}]},
                                            {"const": "wolf"},
                                        ]
                                    },
                                ]
                            },
                        }
                    }
                },
            },
            "most_voted": {
                "description": "Most-voted target THIS round (first-voted wins ties)",
                "params": [],
                "expr": {
                    "get": [
                        {
                            "at": [
                                {
                                    "sort": {
                                        "list": {
                                            "group": {
                                                "list": {
                                                    "filter": {
                                                        "list": {"var": "$voteLog"},
                                                        "as": "$v",
                                                        "where": {
                                                            "eq": [
                                                                {"get": [{"var": "$v"}, "round"]},
                                                                {"var": "$env.round"},
                                                            ]
                                                        },
                                                    }
                                                },
                                                "by": {"get": [{"var": "$item"}, "target"]},
                                            }
                                        },
                                        "by": {"get": [{"var": "$node"}, "count"]},
                                        "reverse": True,
                                    }
                                },
                                {"const": 0},
                            ]
                        },
                        "key",
                    ]
                },
            },
        }
        for role in ("seer", "witch", "hunter", "guard"):
            if self.has[role]:
                fns[f"{role}_alive"] = {"params": [], "expr": _alive_role_count(role)}
        return fns

    def build(self) -> dict:
        views, queries = self._views_queries()
        shoot_targets = [*self.ids, "pass"] if self.has["hunter"] else ["pass"]
        env_fields = {
            "phase": {"type": "string", "initial": "deal_0"},
            "turn": {"type": "player_id", "initial": self.ids[0]},
            "round": {"type": "int", "initial": 1},
            "speechIdx": {"type": "int", "initial": 0},
            "voteIdx": {"type": "int", "initial": 0},
            "nightKill": {"type": "string", "initial": None},
            "poisonTarget": {"type": "string", "initial": None},
            "guardTarget": {"type": "string", "initial": None},
            "guardLastTarget": {"type": "string", "initial": None},
            "witchSavedTarget": {"type": "string", "initial": None},
            "witchSaveUsed": {"type": "int", "initial": 0},
            "witchPoisonUsed": {"type": "int", "initial": 0},
            "seerResult": {"type": "string", "initial": None},
            "hunterShoot": {"type": "string", "initial": None},
            "lynched": {"type": "string", "initial": None},
            "winner": {"type": "string", "initial": None},
        }
        ground = {
            "roles": {"type": "array", "length": {"expr": str(self.players)}, "mutable": True},
            "alive": {"type": "array", "length": {"expr": str(self.players)}, "mutable": True},
            "speechLog": {"type": "array", "mutable": True},
            "voteLog": {"type": "array", "mutable": True},
            "deathsArr": {"type": "array", "mutable": True},
            "env": {"type": "env", "fields": env_fields},
        }
        role_pool = self.role_pool
        utility = []
        for pid in self.ids:
            role_at = {"at": [{"var": "$roles"}, {"call": ["player_index", {"const": pid}]}]}
            for win_role in ("wolf", "good"):
                utility.append(
                    {
                        "player": pid,
                        "value": 1 if win_role == "wolf" else -1,
                        "when": {
                            "and": [
                                {"eq": [{"get": ["$env", "winner"]}, win_role]},
                                {"eq": [role_at, {"const": "wolf"}]},
                            ]
                        },
                    }
                )
                utility.append(
                    {
                        "player": pid,
                        "value": -1 if win_role == "wolf" else 1,
                        "when": {
                            "and": [
                                {"eq": [{"get": ["$env", "winner"]}, win_role]},
                                {"neq": [role_at, {"const": "wolf"}]},
                            ]
                        },
                    }
                )
        return {
            "meta": {
                "name": "werewolf",
                "version": "5.2",
                "description": (
                    f"Werewolf {self.players}-player "
                    f"({'/'.join(f'{self.role_pool.count(r)}{ROLE_LABEL[r]}' for r in ROLES if r in self.role_pool)}) "
                    f"win={self.win_mode}"
                ),
            },
            "players": [{"id": pid, "type": "player"} for pid in self.ids],
            # v5.2: player ids are derived from the declared role pool — the
            # engine selects the (single) composition without any injection.
            "variants": {
                "variant": "default",
                "options": {"default": {}},
                "player_ids": {
                    "map": {
                        "list": {"range": {"from": {"const": 0}, "to": {"count": {"var": "$constants.role_pool"}}}},
                        "as": "$node",
                        "expr": {"template": "p{$node}"},
                    }
                },
                "trim_players": True,
                "trim_utility": True,
            },
            "groundState": ground,
            "derivedViews": views,
            "constants": {
                "role_pool": role_pool,
                "intents": INTENTS,
                "resolve_outcomes": ["resolve"],
                "shoot_targets": shoot_targets,
                "win_mode": self.win_mode,
                "first_night_protect": int(self.first_night_protect),
                "witch_self_save": int(self.witch_self_save),
                "hunter_shoots_when_lynched": int(self.hunter_shoots_when_lynched),
            },
            "queries": queries,
            "functions": self._functions(),
            "actions": self._actions(),
            "chance": self._chance(),
            "effectors": self._effectors(),
            "phases": self._phases(),
            "visibility": self._visibility(),
            "terminal": [
                {"id": "game_over", "condition": {"neq": [{"get": ["$env", "winner"]}, {"const": None}]}},
                # 轮次上限（法官干预）：超限判平局（winner 保持 None）
                {"id": "max_rounds", "condition": {"gte": [{"get": ["$env", "round"]}, {"const": self.max_rounds}]}},
            ],
            "utility": utility,
        }


def gen_rules(
    players: int = 9,
    wolves: int = 3,
    seers: int = 1,
    with_witch: bool = True,
    with_hunter: bool = True,
    with_guard: bool = False,
    win_mode: str = "side",
    first_night_protect: bool = True,
    witch_self_save: bool = False,
    hunter_shoots_when_lynched: bool = True,
) -> dict:
    """Generate the werewolf rules dict (see WerewolfRules docstring)."""
    return WerewolfRules(
        players=players,
        wolves=wolves,
        seers=seers,
        with_witch=with_witch,
        with_hunter=with_hunter,
        with_guard=with_guard,
        win_mode=win_mode,
        first_night_protect=first_night_protect,
        witch_self_save=witch_self_save,
        hunter_shoots_when_lynched=hunter_shoots_when_lynched,
    ).build()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rules/werewolf.json")
    parser.add_argument("--players", type=int, default=9)
    parser.add_argument("--wolves", type=int, default=3)
    parser.add_argument("--seers", type=int, default=1)
    parser.add_argument("--with-witch", action="store_true", default=True)
    parser.add_argument("--with-hunter", action="store_true", default=True)
    parser.add_argument("--with-guard", action="store_true", default=False)
    parser.add_argument("--win-mode", type=str, default="side", choices=["side", "total"])
    parser.add_argument("--no-first-night-protect", action="store_true")
    parser.add_argument("--witch-self-save", action="store_true")
    parser.add_argument("--no-hunter-lynch", action="store_true")
    parser.add_argument("--out", type=str, default="rules/werewolf.json")
    args = parser.parse_args()

    rules = gen_rules(
        args.players,
        args.wolves,
        args.seers,
        with_witch=args.with_witch,
        with_hunter=args.with_hunter,
        with_guard=args.with_guard,
        win_mode=args.win_mode,
        first_night_protect=not args.no_first_night_protect,
        witch_self_save=args.witch_self_save,
        hunter_shoots_when_lynched=not args.no_hunter_lynch,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}  ({rules['meta']['description']})")


if __name__ == "__main__":
    main()
