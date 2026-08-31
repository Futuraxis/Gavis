#!/usr/bin/env python3
"""Undercover (谁是卧底) rules generator — generates ``rules/undercover.json`` (v5.2).

Generates the standard Who-is-the-Undercover rule set with declarative variants:

    python _gen_undercover.py [--players 8] [--max-players 12]
                              [--theme fruit|food|animal|object|place|plant]
                              [--difficulty easy|normal|hard] [--out rules/undercover.json]

Composition (any player count 4..max_players):
  1 卧底 (undercover) + 1 白板 (blank) + N 平民 (civilian)

Theme × difficulty word-pair bank (``variants.options[<theme>_<diff>].constants.word_pairs``
patch, ``pick_word_pair`` chance 开局 uniform 抽一对):
  - 主题(6): fruit / food / animal / object / place / plant
  - 难度档(3,按词对相似度): easy(差异大) / normal(同类相近) / hard(极易混淆)
  - 默认 variant = ``fruit_normal``；平台按 (主题, 难度) 选 variant 名传引擎(不注入词对)

Round flow: deal → describe(每人一句话描述) → vote(投其他存活玩家 或 自爆猜词) →
            resolve(得票最多者出局；平票无人出局) → win check → 下一轮。

身份隐藏（更贴近实际玩法）：开局不开 ``my_role`` 视图——平民/卧底只看自己的
词（``my_word``），靠发言推断自己阵营；白板看到「白板」(无词) 自知是白板。

自爆（self_destruct，投票阶段替代投票）：点名一个存活玩家并猜其词语——
  - 平民自爆 / 卧底猜错 / 白板猜错 → 自爆者直接淘汰，游戏继续（中断本轮投票，
    跳过本轮 resolve，直接进入下一轮 describe）；
  - 卧底猜对平民词（target 是平民且 guess==其词）→ 卧底直接获胜；
  - 白板猜对目标词 → 白板直接获胜。
自爆失败只淘汰自爆者本身，不触发「卧底/白板被投出→平民胜」（那是投票专属）。

Win conditions:
  - 卧底 or 白板 被投出          → 平民胜 (winner=civilian)
  - 卧底自爆猜对平民词           → 卧底胜 (winner=undercover)
  - 白板自爆猜对                 → 白板胜 (winner=blank)
  - 存活 ≤ 3 且白板存活          → 白板胜 (winner=blank)
  - 存活 ≤ 2 且卧底存活          → 卧底胜 (winner=undercover)
  - 存活 ≤ 2（无卧底）           → 平民胜
  - 轮次上限 (players+8)         → 平局 (winner=None)

Design notes
------------
- 部分可观测：``my_word`` 按 viewer 过滤单行（v5.2 visibility）；不开
  ``my_role``（身份隐藏）；死后身份/词语公开（``dead_roles`` / ``dead_words``
  保留 alive==0 的行）。
- 结算用 ``chance``/``resolve`` 阶段 + ``effectMap``（explicit 概率 1.0）。
- ``speak`` / ``self_destruct.guess`` 使用 v5.1 的 ``text`` 自由文本参数预制能力。
- mutable 数组发牌/死亡名单用 ``append``（数组索引即玩家索引）。
- 平票无人出局：``most_voted`` 返回唯一最高票目标，平票 → None。
- variants 声明式：主题×难度(词对池) 经 ``options[<theme>_<diff>].constants.word_pairs``
  补丁选择，``pick_word_pair`` chance 开局 uniform 抽一对写入 env.word_of(全员隐藏)；
  ``role_pool`` / ``player_ids`` 为 ``$player_count`` 上下文的公式
  （引擎纯数据解析，无注入 API）；未知 variant → ValueError。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROLES = ("undercover", "blank", "civilian")
ROLE_LABEL = {"undercover": "卧底", "blank": "白板", "civilian": "平民"}

#: 空白(白板)玩家看到的占位词——无实词,靠听描述混入。
BLANK_WORD = "白板"

#: 难度档位(与平台 ``difficulty`` 契约对齐:easy/normal/hard)。
DIFFICULTIES = ("easy", "normal", "hard")

#: 词对相似度分级:
#:   easy   —— 平民词与卧底词差异大(类别/形态明显不同),易描述区分;
#:   normal —— 同类相近(同属一类的近邻),需仔细描述才能区分;
#:   hard   —— 极易混淆(近义/异名/极似),描述稍不慎即暴露或误判。
#:
#: 每个条目为 ``(平民词, 卧底词)``。每局开局由 ``pick_word_pair`` chance
#: 节点从 ``word_pairs`` 列表 uniform 抽一对,写入 ``env.word_of`` 供发牌查询——
#: 平台只按 (主题, 难度) 选 variant 名(如 ``fruit_hard``),词对在 options
#: 声明、chance 抽取,符合 v5.2「消费者只校验不注入」。
THEMES: dict[str, dict[str, list[tuple[str, str]]]] = {
    "fruit": {
        "easy": [("苹果", "香蕉"), ("西瓜", "葡萄"), ("橘子", "榴莲")],
        "normal": [("苹果", "梨"), ("桃子", "李子"), ("草莓", "蓝莓")],
        "hard": [("苹果", "沙果"), ("菠萝", "凤梨"), ("樱桃", "车厘子")],
    },
    "food": {
        "easy": [("汉堡", "面条"), ("饺子", "冰淇淋"), ("披萨", "寿司")],
        "normal": [("包子", "饺子"), ("馒头", "花卷"), ("烧饼", "油条")],
        "hard": [("肉夹馍", "驴肉火烧"), ("煎饼果子", "鸡蛋灌饼"), ("凉皮", "米皮")],
    },
    "animal": {
        "easy": [("猫", "大象"), ("狗", "企鹅"), ("兔子", "鳄鱼")],
        "normal": [("猫", "狗"), ("老虎", "狮子"), ("海豹", "海狮")],
        "hard": [("猎豹", "花豹"), ("骡子", "马"), ("驴", "骡")],
    },
    "object": {
        "easy": [("手机", "书"), ("钟表", "雨伞"), ("钥匙", "枕头")],
        "normal": [("钢笔", "铅笔"), ("剪刀", "镊子"), ("台灯", "手电筒")],
        "hard": [("抹布", "毛巾"), ("钳子", "扳手"), ("钉子", "螺丝")],
    },
    "place": {
        "easy": [("学校", "医院"), ("公园", "机场"), ("超市", "银行")],
        "normal": [("餐厅", "食堂"), ("咖啡馆", "茶馆"), ("图书馆", "书店")],
        "hard": [("宾馆", "酒店"), ("理发店", "美发厅"), ("药店", "诊所")],
    },
    "plant": {
        "easy": [("树", "草"), ("花", "蘑菇"), ("仙人掌", "竹子")],
        "normal": [("玫瑰", "百合"), ("菊花", "向日葵"), ("柳树", "杨树")],
        "hard": [("玫瑰", "月季"), ("梅花", "桃花"), ("牡丹", "芍药"), ("韭菜", "麦苗")],
    },
}


def all_variants() -> list[str]:
    """主题×难度 展开的全部 variant 名(如 ``fruit_easy``),供校验/枚举。"""
    return [f"{t}_{d}" for t in THEMES for d in DIFFICULTIES]


def _word_of_from_pair(pair: tuple[str, str]) -> dict[str, str]:
    """一对 (平民词, 卧底词) → 发牌查词用的 role→word 映射(白板占位)。"""
    return {"civilian": pair[0], "undercover": pair[1], "blank": BLANK_WORD}


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

    def __init__(self, players: int = 8, max_players: int = 12, theme: str = "fruit", difficulty: str = "normal"):
        if not 4 <= players <= max_players:
            raise ValueError(f"players={players} must satisfy 4 <= players <= max_players={max_players}")
        if theme not in THEMES:
            raise ValueError(f"unknown theme {theme!r}; declared: {sorted(THEMES)}")
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"unknown difficulty {difficulty!r}; declared: {list(DIFFICULTIES)}")
        self.players = players  # 默认(声明)人数 —— variants.player_count
        self.max_players = max_players  # 顶层 players 覆盖的最大座位数
        self.ids = [f"p{i}" for i in range(max_players)]
        self.last_deal = max_players - 1
        self.theme = theme
        self.difficulty = difficulty
        self.variant = f"{theme}_{difficulty}"
        # 展示用:该档首对词(meta description);实际词对开局 chance 抽取。
        self.word_of = _word_of_from_pair(THEMES[theme][difficulty][0])

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
            {"op": "append", "array": "words", "value": {"call": ["word_of_role", {"var": "outcome"}]}},
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

    def _survival_win_ops(self) -> list[dict]:
        """存活数胜负判定链（不含「被投出」分支）——自爆失败结算用。

        自爆失败时只淘汰自爆者本身（平民误用 / 卧底猜错 / 白板猜错），
        这不是「平民把卧底/白板投出」的胜利条件，所以跳过被投出分支，
        只跑存活数判定（≤3 且白板存活→白板胜；≤2→卧底存活则卧底胜否则
        平民胜）。
        """
        return [
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

    def _win_ops(self) -> list[dict]:
        """胜负判定链（do_resolve 用：含「卧底/白板被投出→平民胜」）。"""
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
            *self._survival_win_ops(),
        ]

    def _self_elim_ops(self) -> list[dict]:
        """自爆失败公共尾巴：淘汰自爆者 → 存活胜负判定 → 未结束则下一轮。

        自爆中断当前投票轮（不结算本轮已投的票），直接进入下一轮 describe。
        """
        self_pid = {"var": "$env.turn"}
        return [
            {"op": "setEnv", "key": "eliminated", "value": self_pid},
            {"op": "append", "array": "deathsArr", "value": self_pid},
            self._kill_player(self_pid),
            *self._survival_win_ops(),
            {
                "op": "branch",
                "if": {"neq": [{"var": "$env.phase"}, {"const": "game_over"}]},
                "then": self._next_round_ops(),
            },
        ]

    def _self_destruct_effector(self) -> dict:
        """自爆：猜测目标玩家的词。

        - 平民自爆 → 自爆者淘汰（误用惩罚），游戏继续；
        - 卧底自爆 → 猜对平民词（target 是平民且 guess==其词）→ 卧底胜；
          否则（猜错 / 猜的是白板词）→ 卧底淘汰，游戏继续；
        - 白板自爆 → 猜对目标词 → 白板胜；猜错 → 白板淘汰，游戏继续。

        玩家开局不知自己是平民还是卧底（my_role 隐藏），只能凭自己的词
        （my_word）+ 发言推断是否该自爆。
        """
        self_role = {"at": [{"var": "$roles"}, {"call": ["player_index", {"var": "$env.turn"}]}]}
        target_role = {"at": [{"var": "$roles"}, {"call": ["player_index", {"get": [{"var": "$target"}, "id"]}]}]}
        target_word = {"call": ["word_of_role", target_role]}
        correct = {"eq": [{"var": "$guess"}, target_word]}
        win_ops = lambda winner: [  # noqa: E731
            {"op": "setEnv", "key": "winner", "value": {"const": winner}},
            {"op": "setEnv", "key": "phase", "value": {"const": "game_over"}},
        ]
        return {
            "description": "自爆：猜目标词；平民/猜错淘汰，卧底猜对平民词胜，白板猜对胜",
            "ops": [
                # 公开记录自爆事件（进 speechLog，不进 voteLog——避免被
                # this_round_votes 当作一张票计入。自爆后跳过本轮 resolve，
                # 下轮 this_round_votes 按 round 过滤也不会误读这条）。
                {
                    "op": "append",
                    "array": "speechLog",
                    "value": {
                        "speaker": {"var": "$env.turn"},
                        "event": {"const": "self_destruct"},
                        "target": {"get": [{"var": "$target"}, "id"]},
                        "guess": {"var": "$guess"},
                        "round": {"var": "$env.round"},
                    },
                },
                {
                    "op": "branch",
                    "if": {"eq": [self_role, {"const": "civilian"}]},
                    "then": self._self_elim_ops(),
                    "else": [
                        {
                            "op": "branch",
                            "if": {"eq": [self_role, {"const": "undercover"}]},
                            "then": [
                                {
                                    "op": "branch",
                                    "if": {"and": [correct, {"eq": [target_role, {"const": "civilian"}]}]},
                                    "then": win_ops("undercover"),
                                    "else": self._self_elim_ops(),
                                }
                            ],
                            "else": [
                                {
                                    "op": "branch",
                                    "if": correct,
                                    "then": win_ops("blank"),
                                    "else": self._self_elim_ops(),
                                }
                            ],
                        }
                    ],
                }
            ],
        }

    def _next_round_ops(self) -> list[dict]:
        return [
            {"op": "inc", "key": "round", "by": 1},
            {"op": "setEnv", "key": "speechIdx", "value": {"const": 0}},
            {"op": "setEnv", "key": "voteIdx", "value": {"const": 0}},
            {"op": "setEnv", "key": "phase", "value": {"const": "describe"}},
            {"op": "setEnv", "key": "turn", "value": self._living({"const": 0})},
        ]

    def _effectors(self) -> dict:
        e: dict = {
            # 开局抽词对:从 word_pairs 池(viant patch 提供)uniform 抽一对,
            # 构造 role→word 映射写入 env.word_of,再进入 deal_0 发牌。
            # 词对在规则层声明+chance 抽取,平台只传 variant 名,不注入词对。
            "do_pick_word_pair": {
                "description": "开局从 word_pairs 池 uniform 抽一对词,写入 env.word_of",
                "ops": [
                    # outcome = 词对在 word_pairs 里的序号(int);回查
                    # $constants.word_pairs[outcome] 得 [civ, und] 数组,再取首/次项
                    # 写入 env(白板词固定「白板」常量)。$constants 前缀:effector
                    # ctx 走 $constants.X 解析(与 word_of_role 用 $env.X 同源)。
                    {"op": "setEnv", "key": "civ_word", "value": {"at": [{"at": [{"var": "$constants.word_pairs"}, {"var": "outcome"}]}, {"const": 0}]}},
                    {"op": "setEnv", "key": "und_word", "value": {"at": [{"at": [{"var": "$constants.word_pairs"}, {"var": "outcome"}]}, {"const": 1}]}},
                    {"op": "setEnv", "key": "phase", "value": {"const": "deal_0"}},
                    {"op": "setEnv", "key": "turn", "value": {"const": self.ids[0]}},
                ],
            }
        }
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
        e["do_self_destruct"] = self._self_destruct_effector()
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
            {
                # 自爆（替代投票）：点名一个存活玩家并猜其词语。
                # guess 走 text 预制能力（不参与合法枚举，solver 填实际词）；
                # target 与 vote 同域（alive_others，不能点自己）。
                "id": "self_destruct",
                "type": "self_destruct",
                "phases": ["vote"],
                "actor": {"var": "$env.turn"},
                "params": {
                    "target": {"view": "player", "domain": {"ref": "alive_others"}},
                    "guess": {"type": "text"},
                },
                "legal": {"const": True},
                "effectRef": "do_self_destruct",
                "canonicalKey": {"template": "self_destruct:{$target.id}"},
            },
        ]

    def _chance(self) -> list[dict]:
        chance = [
            {
                "id": "pick_word_pair",
                "phases": ["pick_word_pair"],
                "params": {"word_pair": {"view": "word_pair", "domain": {"ref": "all_word_pairs"}}},
                "probability": {"uniform": {"over": "word_pair"}},
                "effectRef": "do_pick_word_pair",
                "canonicalKey": {"template": "pick_word_pair:{outcome}"},
            },
            *[
                {
                    "id": f"deal_{i}",
                    "phases": [f"deal_{i}"],
                    "params": {"role": {"view": "role", "domain": {"ref": "unassigned_roles"}}},
                    "probability": {"uniform": {"over": "role"}},
                    "effectRef": f"do_deal_{i}",
                    "canonicalKey": {"template": "deal:{outcome}"},
                }
                for i in range(self.max_players)
            ],
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
            {"id": "pick_word_pair", "actions": [], "description": "开局从词对池抽一对词"},
            *[
                {"id": f"deal_{i}", "actions": [], "description": f"Deal role/word to {self.ids[i]}"}
                for i in range(self.max_players)
            ],
        ]
        phases.extend(
            [
                {"id": "describe", "actions": ["speak"], "description": "存活玩家轮流一句话描述自己的词"},
                {"id": "vote", "actions": ["vote", "self_destruct"], "description": "存活玩家轮流投票指认卧底，或自爆猜词"},
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
            # 开局抽词对:word_pairs 由 variant patch 提供(主题×难度档的词对列表),
            # chance uniform 抽一项。fields.id = $self._index(词对在列表里的序号,
            # int→hashable)——chance outcome.key 取节点 id 字段;若用 pair 数组作
            # id 会不可 hash(effect_map.get 崩),故用 index,effector 再回查 word_pairs。
            "word_pair": {
                "from": {"type": "literal", "list": {"var": "word_pairs"}},
                "fields": {"id": {"var": "$self._index"}},
            },
            # v5.2: player-facing views over ground arrays — partial
            # observability declared in ``visibility``; no adapter projection.
            # 身份隐藏：不开 my_role 视图——平民/卧底开局只看自己的词
            # （my_word），靠发言推断阵营；白板看到「白板」(无词) 自知是白板。
            # 死者身份经 dead_roles 公开（alive==0 的行）。
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
            # 全部词对(无过滤)——开局 chance uniform 抽一对。
            "all_word_pairs": {"view": "word_pair"},
        }
        return views, queries

    def _visibility(self) -> dict:
        """Declarative partial observability (v5.2): 只看自己的词，死者公开。

        身份隐藏：不开 my_role 视图（玩家不知自己是平民/卧底），只投影
        my_word（自己的词）+ dead_roles/dead_words（死者身份/词公开）。
        """
        viewer_role = lambda viewer: {"call": ["player_index", {"var": viewer}]}  # noqa: E731
        return {
            "default": "partial",
            "rules": [
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
            # env: civ_word/und_word 含本局平民/卧底词,对所有人隐藏(恒 false
            # filter → pop)。玩家只经 my_word 视图看自己的词;effector 在引擎
            # 内部执行,不受 visibility 限制,照常可读 env.civ_word/und_word。
            "env": {
                "civ_word": {"filter": {"eq": [{"const": 0}, {"const": 1}]}},
                "und_word": {"filter": {"eq": [{"const": 0}, {"const": 1}]}},
            },
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
            "word_of_role": {
                "description": "某身份的本局词(civilian→env.civ_word, undercover→env.und_word, blank→「白板」占位)",
                "params": ["role"],
                "expr": {
                    "if": {
                        "cond": {"eq": [{"var": "$role"}, {"const": "civilian"}]},
                        "then": {"var": "$env.civ_word"},
                        "else": {
                            "if": {
                                "cond": {"eq": [{"var": "$role"}, {"const": "undercover"}]},
                                "then": {"var": "$env.und_word"},
                                "else": {"const": BLANK_WORD},
                            }
                        },
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
            "phase": {"type": "string", "initial": "pick_word_pair"},
            "turn": {"type": "player_id", "initial": self.ids[0]},
            "round": {"type": "int", "initial": 1},
            "speechIdx": {"type": "int", "initial": 0},
            "voteIdx": {"type": "int", "initial": 0},
            "eliminated": {"type": "string", "initial": None},
            "winner": {"type": "string", "initial": None},
            # 本局抽到的平民词/卧底词;由 pick_word_pair chance 填入,
            # 全员隐藏(visibility.env 恒 false)——玩家只经 my_word 视图看自己
            # 的词,绝不能看到平民/卧底词对(隐藏信息红线)。白板词为固定占位
            # 「白板」(常量,非 env),白板靠词自知是白板。
            "civ_word": {"type": "string", "initial": None},
            "und_word": {"type": "string", "initial": None},
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
                    f"variant={self.variant}（默认档首对: {self.word_of['civilian']}/{self.word_of['undercover']}）"
                    " 主题×难度档词库+开局抽词对+身份隐藏(只看词)+自爆猜词"
                ),
            },
            "players": [{"id": pid, "type": "player"} for pid in self.ids],
            # v5.2: 主题×难度档与人数纯数据声明——``options[<theme_diff>].constants``
            # 补丁选该档词对池 + ``$player_count`` 上下文公式;引擎不做任何注入。
            # 词对池 ``word_pairs`` 由 ``pick_word_pair`` chance 开局 uniform 抽一对,
            # 写入 env.word_of(全员隐藏)供发牌/自爆查询。
            "variants": {
                "variant": self.variant,
                "player_count": self.players,
                "options": {
                    f"{t}_{d}": {"constants": {"word_pairs": [[c, u] for c, u in THEMES[t][d]]}}
                    for t in THEMES
                    for d in DIFFICULTIES
                },
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
                # 默认档(fruit_normal)词对池 fallback;variant patch 覆盖为选中档。
                "word_pairs": [[c, u] for c, u in THEMES[self.theme][self.difficulty]],
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


def gen_rules(players: int = 8, max_players: int = 12, theme: str = "fruit", difficulty: str = "normal") -> dict:
    """Generate the undercover rules dict (see UndercoverRules docstring)."""
    return UndercoverRules(players=players, max_players=max_players, theme=theme, difficulty=difficulty).build()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rules/undercover.json")
    parser.add_argument("--players", type=int, default=8, help="默认（声明）人数，4..max_players")
    parser.add_argument(
        "--max-players", type=int, default=12, help="JSON 覆盖的最大座位数（variants 可选的玩家数上限）"
    )
    parser.add_argument("--theme", type=str, default="fruit", choices=sorted(THEMES), help="默认词对主题")
    parser.add_argument(
        "--difficulty", type=str, default="normal", choices=list(DIFFICULTIES), help="默认难度档(词对相似度)"
    )
    parser.add_argument("--out", type=str, default="rules/undercover.json")
    args = parser.parse_args()

    rules = gen_rules(args.players, args.max_players, args.theme, args.difficulty)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}  ({rules['meta']['description']})")


if __name__ == "__main__":
    main()
