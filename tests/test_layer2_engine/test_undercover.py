"""Engine tests for 谁是卧底 (undercover) rules — ``rules/undercover.json`` (v5.2).

Covers:
- variants：场景(词对 fruit/food) 与人数 (4..12) 纯数据选择；未知 variant → ValueError
- 发牌：1卧底 + 1白板 + N平民，词与身份一一对应
- 身份隐藏：不开 my_role 视图（玩家不知自己是平民/卧底），只看 my_word；
  白板看到「白板」自知是白板；死后身份/词语公开
- 轮转：describe/vote 由规则层推进（speechLog[i].speaker == living[i]）
- 自爆（self_destruct，投票阶段替代投票）：平民/猜错淘汰、卧底猜对平民词胜、
  白板猜对胜；自爆失败中断本轮投票，跳过 resolve 进入下一轮
- 平票无人出局、不能投自己
- 胜负：卧底/白板被投出 → 平民胜；白板活到剩三人 → 白板胜；
  卧底活到剩两人 → 卧底胜
- 收益：胜方阵营 +1、其余 -1；平局 0
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from _gen_undercover import gen_rules
from layer1_translator.schema_validator import SchemaValidator
from layer2_engine.core.engine import GameEngine

RULES_PATH = Path(__file__).resolve().parent.parent.parent / "rules" / "undercover.json"


def _engine(seed: int = 7, player_count: int = 8, variant: str | None = None) -> GameEngine:
    """Engine over the shipped JSON (variants are resolved as declared data)."""
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=seed, player_count=player_count, variant=variant)


# 测试固定词对(匹配自爆测试里的 guess「苹果/香蕉/白板」);词对现由开局
# pick_word_pair chance 随机抽写入 env.word_of,手构状态需显式注入。
_WORD_OF = {"civilian": "苹果", "undercover": "香蕉", "blank": "白板"}


def _play_until(adapter: GameEngine, rng: random.Random, phase_target: str, max_steps: int = 300) -> dict | None:
    """Advance until a phase is reached; return the state (or None)."""
    state = adapter.create_initial_state()
    for _ in range(max_steps):
        if state["env"].get("phase") == phase_target:
            return state
        nt = adapter.get_node_type(state)
        if nt == "chance":
            outs = adapter.get_chance_outcomes(state)
            if not outs:
                return None
            state = adapter.apply_chance(state, rng.choices(outs, weights=[o.probability for o in outs], k=1)[0])
        elif nt == "player":
            legal = adapter.get_legal_actions(state)
            if not legal:
                return None
            a = rng.choice(legal)
            if a.template_id == "speak":
                a = replace(a, params={**a.params, "text": "测试描述"})
            state = adapter.apply_action(state, a)
            if adapter.is_terminal(state):
                return state
        else:
            return state
    return None


def _act(adapter: GameEngine, state: dict, template: str, **params) -> dict:
    """Apply the first legal action matching ``template`` with the given params."""
    for a in adapter.get_legal_actions(state):
        if a.template_id != template:
            continue
        matched = all(a.params.get(k) == v for k, v in params.items())
        if matched:
            return adapter.apply_action(state, a)
    raise AssertionError(f"not legal: {template} {params} at {state['env']['phase']}")


def _speak(adapter: GameEngine, state: dict, text: str = "发言") -> dict:
    for a in adapter.get_legal_actions(state):
        if a.template_id == "speak":
            return adapter.apply_action(state, replace(a, params={**a.params, "text": text}))
    raise AssertionError(f"speak not legal at {state['env']['phase']}")


def _living(adapter: GameEngine, state: dict) -> list[str]:
    alive = state["_arrays"]["alive"]
    return [pid for i, pid in enumerate(adapter._constants["player_ids"]) if i < len(alive) and alive[i] == 1]  # noqa: SLF001


def _resolve(adapter: GameEngine, state: dict) -> dict:
    """Apply the resolve chance node (phase must be ``resolve``)."""
    assert state["env"]["phase"] == "resolve"
    outs = adapter.get_chance_outcomes(state)
    assert len(outs) == 1 and outs[0].probability == 1.0
    return adapter.apply_chance(state, outs[0])


def _craft(
    adapter: GameEngine,
    roles: list[str],
    alive: list[int],
    votes: list[dict],
    round_: int = 1,
    eliminated: str | None = None,
) -> dict:
    """Hand-built resolve state (votes are data; the engine only counts them)."""
    st = adapter.create_initial_state()
    st["_arrays"]["roles"] = list(roles)
    st["_arrays"]["words"] = [_WORD_OF[r] for r in roles]
    st["_arrays"]["alive"] = list(alive)
    st["_arrays"]["speechLog"] = []
    st["_arrays"]["voteLog"] = list(votes)
    st["_arrays"]["deathsArr"] = []
    st["env"].update(
        {
            "phase": "resolve",
            "round": round_,
            "speechIdx": 0,
            "voteIdx": 0,
            "eliminated": eliminated,
            "winner": None,
            "civ_word": _WORD_OF["civilian"],
            "und_word": _WORD_OF["undercover"],
            "turn": adapter._constants["player_ids"][0],
        }
    )
    return st


def _craft_vote(
    adapter: GameEngine,
    roles: list[str],
    alive: list[int],
    voter_idx: int = 0,
    round_: int = 1,
) -> dict:
    """Hand-built vote-phase state（voter = 第 voter_idx 个存活玩家）。"""
    st = adapter.create_initial_state()
    st["_arrays"]["roles"] = list(roles)
    st["_arrays"]["words"] = [_WORD_OF[r] for r in roles]
    st["_arrays"]["alive"] = list(alive)
    st["_arrays"]["speechLog"] = []
    st["_arrays"]["voteLog"] = []
    st["_arrays"]["deathsArr"] = []
    pids = adapter._constants["player_ids"]
    living = [pids[i] for i, a in enumerate(alive) if a == 1]
    st["env"].update(
        {
            "phase": "vote",
            "round": round_,
            "speechIdx": 0,
            "voteIdx": 0,
            "eliminated": None,
            "winner": None,
            "civ_word": _WORD_OF["civilian"],
            "und_word": _WORD_OF["undercover"],
            "turn": living[voter_idx],
        }
    )
    return st


def _self_destruct(adapter: GameEngine, state: dict, target_id: str, guess: str) -> dict:
    """Apply self_destruct against ``target_id`` with the given guessed word."""
    for a in adapter.get_legal_actions(state):
        if a.template_id != "self_destruct":
            continue
        t = a.params.get("target")
        t_id = t.get("id") if isinstance(t, dict) else t
        if t_id == target_id:
            return adapter.apply_action(state, replace(a, params={**a.params, "guess": guess}))
    raise AssertionError(f"self_destruct target {target_id} not legal at {state['env']['phase']}")


# ── L1 / variants ──────────────────────────────────────────────────


def test_rules_pass_l1_schema() -> None:
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        rules = json.load(f)
    result = SchemaValidator.validate(rules)
    assert result.valid, result.errors


def test_gen_rules_builds_engine() -> None:
    rules = gen_rules(players=8, max_players=12)
    adapter = GameEngine(rules, seed=1)
    assert adapter.get_node_type(adapter.create_initial_state()) == "chance"  # deal_0


def test_variants_pick_player_count_and_scenario() -> None:
    for count in (4, 6, 8, 10, 12):
        adapter = _engine(seed=7, player_count=count)
        pids = adapter._constants["player_ids"]
        pool = adapter._constants["role_pool"]
        assert len(pids) == count
        assert len(pool) == count
        assert pool.count("undercover") == 1
        assert pool.count("blank") == 1
        assert pool.count("civilian") == count - 2
        assert pids[0] == "p0" and pids[-1] == f"p{count - 1}"
    # 主题×难度档词对池:options[<theme>_<diff>].constants.word_pairs 补丁生效
    food_hard = _engine(seed=7, variant="food_hard")
    pairs = food_hard._constants["word_pairs"]
    assert isinstance(pairs, list) and len(pairs) >= 3
    assert ["肉夹馍", "驴肉火烧"] in pairs  # hard 档含高混淆词对
    fruit_normal = _engine(seed=7, variant="fruit_normal")
    fn_pairs = fruit_normal._constants["word_pairs"]
    assert ["苹果", "梨"] in fn_pairs  # normal 档含同类相近词对
    # 旧 variant 名(fruit/food 单主题)已废弃 → 现需 theme_diff
    with pytest.raises(ValueError, match="unknown variant"):
        _engine(seed=7, variant="food")


def test_unknown_variant_raises() -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        _engine(seed=7, variant="no_such_scenario")


# ── 发牌 ───────────────────────────────────────────────────────────


def test_deal_assigns_roles_and_words() -> None:
    adapter = _engine(seed=7, player_count=6)
    rng = random.Random(7)
    st = _play_until(adapter, rng, "describe")
    assert st is not None
    roles = st["_arrays"]["roles"]
    words = st["_arrays"]["words"]
    assert len(roles) == 6 and len(words) == 6
    assert roles.count("undercover") == 1
    assert roles.count("blank") == 1
    assert roles.count("civilian") == 4
    expected = {"civilian": st["env"]["civ_word"], "undercover": st["env"]["und_word"], "blank": "白板"}
    for role, word in zip(roles, words):
        assert word == expected[role], f"word {word} != {expected[role]} for {role}"
    assert st["env"]["phase"] == "describe"
    assert st["env"]["turn"] == "p0"


# ── 部分可观测 ────────────────────────────────────────────────────


def test_observation_hides_role_shows_word() -> None:
    """身份隐藏：不开 my_role 视图；每个玩家只看自己的词（my_word）。

    平民/卧底看不到自己的身份标签（要靠发言推断阵营）；白板看到「白板」
    （无词）自知是白板——靠词而非靠 my_role。死后身份/词语仍经 dead_roles/
    dead_words 公开。
    """
    adapter = _engine(seed=7, player_count=6)
    rng = random.Random(7)
    st = _play_until(adapter, rng, "describe")
    assert st is not None
    pids = adapter._constants["player_ids"]
    roles = st["_arrays"]["roles"]
    for pid in pids:
        obs = adapter.project_observation(st, pid)
        idx = pids.index(pid)
        # 身份隐藏：my_role 视图不存在（玩家不知自己的身份标签）
        assert "my_role" not in obs
        # 但自己的词可见（单行视图）——平民/卧底看真词，白板看「白板」
        assert [e.get("word") for e in obs["my_word"]] == [st["_arrays"]["words"][idx]]
        assert [e.get("_index") for e in obs["my_word"]] == [idx]
    # 白板的词就是「白板」→ 白板靠词自知是白板
    blank_idx = roles.index("blank")
    blank_pid = pids[blank_idx]
    assert st["_arrays"]["words"][blank_idx] == "白板"
    assert adapter.project_observation(st, blank_pid)["my_word"][0]["word"] == "白板"
    # 公共信息（发言/存活/收场日志）完整可见
    obs0 = adapter.project_observation(st, "p0")
    assert isinstance(obs0["speech_log"], list)
    assert isinstance(obs0["alive"], list)
    assert isinstance(obs0["dead_roles"], list)
    assert isinstance(obs0["dead_words"], list)


def test_dead_roles_and_words_public_after_elimination() -> None:
    adapter = _engine(seed=7, player_count=6)
    roles = ["civilian", "undercover", "blank", "civilian", "civilian", "civilian"]
    alive = [1, 1, 1, 0, 0, 0]  # p3-p5 已死
    st = adapter.create_initial_state()
    st["_arrays"]["roles"] = roles
    st["_arrays"]["words"] = [_WORD_OF[r] for r in roles]
    st["_arrays"]["alive"] = alive
    st["env"].update({"phase": "describe", "turn": "p0", "round": 1, "speechIdx": 0, "voteIdx": 0})
    for pid in adapter._constants["player_ids"]:
        obs = adapter.project_observation(st, pid)
        dead_idx = [i for i, a in enumerate(alive) if a == 0]
        assert sorted(e.get("_index") for e in obs["dead_roles"]) == dead_idx
        assert sorted(e.get("_index") for e in obs["dead_words"]) == dead_idx
        assert [e.get("role") for e in obs["dead_roles"]] == [roles[i] for i in dead_idx]


# ── 轮转 ──────────────────────────────────────────────────────────


def test_speech_rotation() -> None:
    adapter = _engine(seed=203, player_count=6)
    rng = random.Random(203)
    st = _play_until(adapter, rng, "describe")
    assert st is not None and adapter.get_node_type(st) == "player"
    living = _living(adapter, st)
    assert st["env"]["turn"] == living[0]
    for i in range(len(living)):
        st = _speak(adapter, st, f"第{i}个描述")
        log = st["_arrays"]["speechLog"]
        assert log[-1]["speaker"] == living[i], f"speaker {log[-1]['speaker']} != {living[i]}"
        assert log[-1]["round"] == st["env"]["round"]
        if i < len(living) - 1:
            assert st["env"]["turn"] == living[i + 1]
    # 最后一言后进入投票，turn 回到首位
    assert st["env"]["phase"] == "vote"
    assert st["env"]["voteIdx"] == 0
    assert st["env"]["turn"] == living[0]


def test_vote_rotation_and_no_self_vote() -> None:
    adapter = _engine(seed=203, player_count=6)
    rng = random.Random(203)
    st = _play_until(adapter, rng, "describe")
    assert st is not None
    living = _living(adapter, st)
    for _ in range(len(living)):
        st = _speak(adapter, st)
    assert st["env"]["phase"] == "vote"
    for i in range(len(living)):
        voter = st["env"]["turn"]
        legal = adapter.get_legal_actions(state=st)
        vote_acts = [a for a in legal if a.template_id == "vote"]
        assert vote_acts, f"no vote actions at {st['env']['phase']}"
        # 不能投自己，且目标都是存活玩家
        for a in vote_acts:
            assert a.params["target"]["id"] in living
            assert a.params["target"]["id"] != voter
        st = _apply_first_vote(adapter, st, vote_acts)
        log = st["_arrays"]["voteLog"]
        assert log[-1]["voter"] == living[i]
    assert st["env"]["phase"] == "resolve"


def _apply_first_vote(adapter: GameEngine, state: dict, vote_acts) -> dict:
    target = vote_acts[0].params["target"]["id"]
    for a in adapter.get_legal_actions(state):
        if a.template_id == "vote" and a.params["target"]["id"] == target:
            return adapter.apply_action(state, a)
    raise AssertionError("vote target disappeared")


# ── 平票 ──────────────────────────────────────────────────────────


def test_tie_vote_no_elimination() -> None:
    adapter = _engine(seed=7, player_count=6)
    votes = [
        {"voter": "p0", "target": "p3", "round": 1},
        {"voter": "p1", "target": "p3", "round": 1},
        {"voter": "p2", "target": "p4", "round": 1},
        {"voter": "p3", "target": "p4", "round": 1},
        {"voter": "p4", "target": "p5", "round": 1},
        {"voter": "p5", "target": "p5", "round": 1},  # 数据构造：平票 2/2/2
    ]
    st = _craft(adapter, ["civilian"] * 6, [1] * 6, votes)
    st = _resolve(adapter, st)
    assert st["_arrays"]["deathsArr"] == []
    assert st["_arrays"]["alive"] == [1] * 6
    assert st["env"].get("winner") is None
    # 进入下一轮 describe
    assert st["env"]["phase"] == "describe"
    assert st["env"]["round"] == 2
    assert st["env"]["turn"] == "p0"


# ── 胜负判定 ──────────────────────────────────────────────────────


def test_win_when_undercover_voted_out() -> None:
    adapter = _engine(seed=7, player_count=6)
    roles = ["undercover", "blank", "civilian", "civilian", "civilian", "civilian"]
    votes = [
        {"voter": "p1", "target": "p0", "round": 1},
        {"voter": "p2", "target": "p0", "round": 1},
        {"voter": "p3", "target": "p5", "round": 1},
    ]
    st = _craft(adapter, roles, [1] * 6, votes)
    st = _resolve(adapter, st)
    assert st["env"]["winner"] == "civilian"
    assert st["env"]["phase"] == "game_over"
    assert st["_arrays"]["deathsArr"] == ["p0"]
    assert st["env"]["eliminated"] == "p0"


def test_win_when_blank_voted_out() -> None:
    adapter = _engine(seed=7, player_count=6)
    roles = ["undercover", "blank", "civilian", "civilian", "civilian", "civilian"]
    votes = [
        {"voter": "p0", "target": "p1", "round": 1},
        {"voter": "p2", "target": "p1", "round": 1},
        {"voter": "p3", "target": "p5", "round": 1},
    ]
    st = _craft(adapter, roles, [1] * 6, votes)
    st = _resolve(adapter, st)
    assert st["env"]["winner"] == "civilian"
    assert st["env"]["eliminated"] == "p1"


def test_win_blank_at_three_alive() -> None:
    """白板存活到只剩 3 人 → 白板胜（即使本回合被投出的是平民）。"""
    adapter = _engine(seed=7, player_count=6)
    roles = ["undercover", "blank", "civilian", "civilian", "civilian", "civilian"]
    alive = [1, 1, 1, 0, 0, 0]
    votes = [
        {"voter": "p0", "target": "p2", "round": 1},
        {"voter": "p1", "target": "p2", "round": 1},
        {"voter": "p2", "target": "p0", "round": 1},
    ]
    st = _craft(adapter, roles, alive, votes)
    st = _resolve(adapter, st)
    assert st["env"]["winner"] == "blank"
    assert st["env"]["phase"] == "game_over"
    assert st["env"]["eliminated"] == "p2"


def test_win_undercover_at_two_alive() -> None:
    """卧底存活到只剩 2 人（白板已死）→ 卧底胜；平票也不影响。"""
    adapter = _engine(seed=7, player_count=6)
    roles = ["undercover", "blank", "civilian", "civilian", "civilian", "civilian"]
    alive = [1, 0, 1, 0, 0, 0]
    votes = [
        {"voter": "p0", "target": "p2", "round": 1},
        {"voter": "p2", "target": "p0", "round": 1},
    ]
    st = _craft(adapter, roles, alive, votes)
    st = _resolve(adapter, st)
    assert st["env"]["winner"] == "undercover"
    assert st["env"]["phase"] == "game_over"
    assert st["_arrays"]["deathsArr"] == []


def test_win_civilian_at_two_alive_without_undercover() -> None:
    """只剩 2 人且无卧底（防御分支）→ 平民胜。"""
    adapter = _engine(seed=7, player_count=6)
    roles = ["civilian", "blank", "civilian", "civilian", "civilian", "civilian"]
    alive = [1, 0, 1, 0, 0, 0]
    votes = []
    st = _craft(adapter, roles, alive, votes)
    st = _resolve(adapter, st)
    assert st["env"]["winner"] == "civilian"


def test_eliminating_civilian_continues_game() -> None:
    """平民被投出不会直接结束（人数仍 > 3）。"""
    adapter = _engine(seed=7, player_count=6)
    roles = ["undercover", "blank", "civilian", "civilian", "civilian", "civilian"]
    votes = [
        {"voter": "p0", "target": "p2", "round": 1},
        {"voter": "p1", "target": "p2", "round": 1},
        {"voter": "p3", "target": "p5", "round": 1},
    ]
    st = _craft(adapter, roles, [1] * 6, votes)
    st = _resolve(adapter, st)
    assert st["env"].get("winner") is None
    assert st["env"]["phase"] == "describe"
    assert st["env"]["round"] == 2
    assert st["_arrays"]["alive"] == [1, 1, 0, 1, 1, 1]


# ── 自爆（self_destruct）─────────────────────────────────────────────


def test_self_destruct_legal_alongside_vote() -> None:
    """投票阶段：当前玩家既可 vote 也可 self_destruct（同一阶段两动作共存）。"""
    adapter = _engine(seed=203, player_count=6)
    rng = random.Random(203)
    st = _play_until(adapter, rng, "vote")
    assert st is not None and st["env"]["phase"] == "vote"
    legal = adapter.get_legal_actions(state=st)
    assert any(a.template_id == "vote" for a in legal)
    assert any(a.template_id == "self_destruct" for a in legal)
    # self_destruct 不能点自己（target 域 = alive_others，与 vote 同）
    voter = st["env"]["turn"]
    for a in legal:
        if a.template_id != "self_destruct":
            continue
        t = a.params.get("target")
        t_id = t.get("id") if isinstance(t, dict) else t
        assert t_id != voter


def test_undercover_self_destruct_correct_wins() -> None:
    """卧底自爆、猜对平民词（target 是平民且 guess==其词）→ 卧底直接获胜。"""
    adapter = _engine(seed=7, player_count=6)
    roles = ["undercover", "blank", "civilian", "civilian", "civilian", "civilian"]
    st = _craft_vote(adapter, roles, [1] * 6, voter_idx=0)  # voter = p0(卧底)
    st = _self_destruct(adapter, st, target_id="p2", guess="苹果")  # 苹果=平民词
    assert st["env"]["winner"] == "undercover"
    assert st["env"]["phase"] == "game_over"
    # 赢家不被淘汰（自爆成功=获胜，非出局）；eliminated 保持 None
    assert st["env"].get("eliminated") is None


def test_undercover_self_destruct_wrong_eliminated_continues() -> None:
    """卧底自爆猜错 → 卧底淘汰、游戏继续（不触发平民胜，自爆非投票）。"""
    adapter = _engine(seed=7, player_count=6)
    roles = ["undercover", "blank", "civilian", "civilian", "civilian", "civilian"]
    st = _craft_vote(adapter, roles, [1] * 6, voter_idx=0)
    st = _self_destruct(adapter, st, target_id="p2", guess="香蕉")  # 香蕉=卧底词，对平民是错的
    assert st["env"].get("winner") is None  # 不是平民胜
    assert st["env"]["phase"] == "describe"  # 中断本轮投票 → 下一轮
    assert st["env"]["round"] == 2
    assert st["_arrays"]["alive"] == [0, 1, 1, 1, 1, 1]  # p0(卧底) 出局
    assert st["env"]["eliminated"] == "p0"


def test_undercover_self_destruct_blank_word_does_not_win() -> None:
    """卧底猜对白板的词（白板）不算赢——必须猜对平民词。"""
    adapter = _engine(seed=7, player_count=6)
    roles = ["undercover", "blank", "civilian", "civilian", "civilian", "civilian"]
    st = _craft_vote(adapter, roles, [1] * 6, voter_idx=0)
    st = _self_destruct(adapter, st, target_id="p1", guess="白板")  # p1=白板，词白板，猜对但非平民
    assert st["env"].get("winner") is None  # 不算卧底胜
    assert st["env"]["phase"] == "describe"
    assert st["_arrays"]["alive"] == [0, 1, 1, 1, 1, 1]  # 卧底淘汰


def test_civilian_self_destruct_always_eliminated() -> None:
    """平民自爆（即便猜对）→ 直接淘汰，游戏继续。"""
    adapter = _engine(seed=7, player_count=6)
    roles = ["civilian", "blank", "undercover", "civilian", "civilian", "civilian"]
    st = _craft_vote(adapter, roles, [1] * 6, voter_idx=0)  # voter = p0(平民)
    st = _self_destruct(adapter, st, target_id="p2", guess="香蕉")  # 猜对卧底词，但平民→淘汰
    assert st["env"].get("winner") is None
    assert st["env"]["phase"] == "describe"
    assert st["_arrays"]["alive"] == [0, 1, 1, 1, 1, 1]


def test_blank_self_destruct_correct_wins() -> None:
    """白板自爆、猜对目标词 → 白板直接获胜。"""
    adapter = _engine(seed=7, player_count=6)
    roles = ["blank", "undercover", "civilian", "civilian", "civilian", "civilian"]
    st = _craft_vote(adapter, roles, [1] * 6, voter_idx=0)  # voter = p0(白板)
    st = _self_destruct(adapter, st, target_id="p2", guess="苹果")  # 猜对平民词
    assert st["env"]["winner"] == "blank"
    assert st["env"]["phase"] == "game_over"


def test_blank_self_destruct_wrong_eliminated() -> None:
    """白板自爆猜错 → 白板淘汰、游戏继续。"""
    adapter = _engine(seed=7, player_count=6)
    roles = ["blank", "undercover", "civilian", "civilian", "civilian", "civilian"]
    st = _craft_vote(adapter, roles, [1] * 6, voter_idx=0)
    st = _self_destruct(adapter, st, target_id="p2", guess="香蕉")  # 猜错
    assert st["env"].get("winner") is None
    assert st["env"]["phase"] == "describe"
    assert st["_arrays"]["alive"] == [0, 1, 1, 1, 1, 1]


def test_self_destruct_failure_triggers_survival_win() -> None:
    """自爆失败淘汰后，若剩余存活触发生存胜利条件仍会结算（白板剩三人）。"""
    adapter = _engine(seed=7, player_count=6)
    roles = ["civilian", "blank", "undercover", "civilian", "civilian", "civilian"]
    # 4 存活：p0(平民)、p1(白板)、p2(卧底)、p3(平民)；p0 自爆失败淘汰 → 剩3人且白板存活
    st = _craft_vote(adapter, roles, [1, 1, 1, 1, 0, 0], voter_idx=0)
    st = _self_destruct(adapter, st, target_id="p2", guess="苹果")  # 平民自爆→淘汰
    assert st["env"]["winner"] == "blank"  # 剩3人(p1,p2,p3) 且白板存活 → 白板胜
    assert st["env"]["phase"] == "game_over"
    assert st["_arrays"]["alive"] == [0, 1, 1, 1, 0, 0]


# ── 收益 ──────────────────────────────────────────────────────────


def test_utility_by_faction() -> None:
    adapter = _engine(seed=7, player_count=6)
    roles = ["undercover", "blank", "civilian", "civilian", "civilian", "civilian"]
    alive = [1, 1, 1, 0, 0, 0]
    votes = [
        {"voter": "p0", "target": "p2", "round": 1},
        {"voter": "p1", "target": "p2", "round": 1},
        {"voter": "p2", "target": "p0", "round": 1},
    ]
    st = _resolve(adapter, _craft(adapter, roles, alive, votes))
    winner = st["env"]["winner"]
    assert winner == "blank"
    pids = adapter._constants["player_ids"]
    for i, pid in enumerate(pids):
        u = adapter.get_utility(st, pid)
        assert u == (1.0 if roles[i] == winner else -1.0), f"{pid} role {roles[i]} utility {u}"


def test_draw_utility_is_zero() -> None:
    """轮次上限触发（winner=None）→ 所有人收益 0。"""
    adapter = _engine(seed=7, player_count=6)
    st = adapter.create_initial_state()
    st["env"]["round"] = 999  # 超过 players+8 上限
    assert adapter.is_terminal(st)
    for pid in adapter._constants["player_ids"]:
        assert adapter.get_utility(st, pid) == 0.0


# ── 随机自对弈 ────────────────────────────────────────────────────


def test_full_games_terminate() -> None:
    """随机自对弈若干局（多人数）全部正常终局、胜负合法。"""
    winners = Counter()
    for count in (4, 6, 8, 10):
        for s in range(8):
            adapter = _engine(seed=100 + s, player_count=count)
            rng = random.Random(100 + s)
            state = adapter.create_initial_state()
            steps = 0
            while True:
                steps += 1
                assert steps <= 2000, f"count={count} seed={s} did not terminate"
                nt = adapter.get_node_type(state)
                if nt == "chance":
                    outs = adapter.get_chance_outcomes(state)
                    if not outs:
                        break
                    state = adapter.apply_chance(
                        state, rng.choices(outs, weights=[o.probability for o in outs], k=1)[0]
                    )
                    continue
                if nt != "player":
                    break
                cur = adapter.get_current_player(state)
                assert cur is not None, f"stuck at {state['env']['phase']}"
                legal = adapter.get_legal_actions(state)
                assert legal, f"no legal actions at {state['env']['phase']}"
                a = rng.choice(legal)
                if a.template_id == "speak":
                    a = replace(a, params={**a.params, "text": f"{cur}的描述"})
                state = adapter.apply_action(state, a)
                if adapter.is_terminal(state):
                    break
            winner = state["env"].get("winner")
            assert winner in (None, "civilian", "undercover", "blank"), f"winner {winner!r}"
            winners[winner] += 1
    assert len(winners) >= 2  # 至少两类结局（随机对弈不应只有一种）
    print(f"  undercover random self-play: {dict(winners)}")
