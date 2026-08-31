import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

from scripts._gen_werewolf import gen_rules
from layer2_engine.core.engine import GameEngine

RULES_PATH = Path(__file__).resolve().parent.parent.parent / "rules" / "werewolf.json"


def _engine(seed: int = 7) -> GameEngine:
    """Bare engine — composition is declared data in the regenerated JSON (v5.2)."""
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=seed)


def play_until(adapter, rng, phase_target, max_steps=300):
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
                a = replace(a, params={**a.params, "text": "测试发言"})
            state = adapter.apply_action(state, a)
            if adapter.is_terminal(state):
                return state
        else:
            return state
    return None


# ── 部分可观测 ────────────────────────────────────────────────────


def test_observation_filters_roles_and_seer_result():
    """非预言家看不到别人的角色与验人结果；预言家能看到自己的验人结果。"""
    adapter = _engine(seed=7)
    rng = random.Random(7)
    # 找到预言家的索引
    st = play_until(adapter, rng, "night_seer")
    roles = st["_arrays"]["roles"]
    seer_idx = roles.index("seer")
    seer_id = f"p{seer_idx}"
    other_id = next(pid for i, pid in enumerate(adapter._constants["player_ids"]) if i != seer_idx)
    # 先让预言家验人
    legal = adapter.get_legal_actions(state=st)
    check = next(a for a in legal if a.template_id == "check")
    st2 = adapter.apply_action(st, check)
    assert st2["env"].get("seerResult") is not None
    # 预言家自己能看到验人结果（env 字段级可见性，v5.2）
    obs_seer = adapter.project_observation(st2, seer_id)
    assert obs_seer["env"].get("seerResult") == st2["env"]["seerResult"]
    # my_role 是按 viewer 过滤的单行视图
    assert [e.get("role") for e in obs_seer["my_role"]] == ["seer"]
    # 其他玩家看不到验人结果
    obs_other = adapter.project_observation(st2, other_id)
    assert "seerResult" not in obs_other["env"]
    assert [e.get("role") for e in obs_other["my_role"]] != ["seer"]
    # 任何人都只看到自己的角色行
    other_idx = adapter._constants["player_ids"].index(other_id)  # noqa: SLF001
    assert [e.get("_index") for e in obs_other["my_role"]] == [other_idx]
    # 发言记录公开
    assert isinstance(obs_seer["speech_log"], list)


def test_observation_speech_log_is_public():
    """发言日志对所有玩家公开。"""
    st = None
    for s in range(20):  # 有些局第一夜后狼直接获胜，重试找进白天的局
        adapter = _engine(seed=200 + s)
        rng = random.Random(200 + s)
        cand = play_until(adapter, rng, "day_speech")
        if cand is not None and adapter.get_node_type(cand) == "player":
            st = cand
            break
    assert st is not None, "no game reached day_speech"
    legal = adapter.get_legal_actions(state=st)
    assert any(a.template_id == "speak" for a in legal)
    speak = next(a for a in legal if a.template_id == "speak")
    speak = replace(speak, params={**speak.params, "text": "我是真预言家"})
    st2 = adapter.apply_action(st, speak)
    for pid in adapter._constants["player_ids"]:
        obs = adapter.project_observation(st2, pid)
        entries = [e.get("entry", {}) for e in obs["speech_log"]]
        assert any("我是真预言家" in s.get("text", "") for s in entries)


# ── 胜负与收益 ────────────────────────────────────────────────────


def test_utility_by_faction():
    """狼赢：狼玩家 +1 好人 -1；好人赢：相反。"""
    adapter = _engine(seed=3)
    rng = random.Random(3)
    st = play_until(adapter, rng, "game_over")
    assert st is not None
    winner = st["env"]["winner"]
    assert winner in ("wolf", "good")
    roles = st["_arrays"]["roles"]
    for i, pid in enumerate(adapter._constants["player_ids"]):
        u = adapter.get_utility(st, pid)
        if roles[i] == "wolf":
            assert u == (1 if winner == "wolf" else -1), f"{pid} wolf utility {u}"
        else:
            assert u == (-1 if winner == "wolf" else 1), f"{pid} good utility {u}"


def test_full_games_terminate():
    """随机自对弈 30 局全部正常终局、胜负合法（默认 3狼1预1女巫1猎人）。"""
    winners = Counter()
    for s in range(30):
        adapter = _engine(seed=100 + s)
        rng = random.Random(100 + s)
        state = adapter.create_initial_state()
        steps = 0
        while True:
            steps += 1
            assert steps <= 2000, f"game {s} did not terminate"
            nt = adapter.get_node_type(state)
            if nt == "chance":
                outs = adapter.get_chance_outcomes(state)
                if not outs:
                    break
                state = adapter.apply_chance(state, rng.choices(outs, weights=[o.probability for o in outs], k=1)[0])
                continue
            if nt != "player":
                break
            cur = adapter.get_current_player(state)
            assert cur is not None, f"game {s} stuck with no current player at {state['env']['phase']}"
            legal = adapter.get_legal_actions(state)
            assert legal, f"game {s} no legal actions at {state['env']['phase']}"
            a = rng.choice(legal)
            if a.template_id == "speak":
                a = replace(a, params={**a.params, "text": f"{cur}的发言"})
            state = adapter.apply_action(state, a)
            if adapter.is_terminal(state):
                break
        winner = state["env"].get("winner")
        assert winner in ("wolf", "good"), f"game {s} winner {winner}"
        winners[winner] += 1
    assert len(winners) >= 1
    print(f"  werewolf random self-play 30 games: {dict(winners)}")


# ── 规则层轮转（P1-5/6/7/8/16 修复回归）────────────────────────────


def _living_ids(adapter, state):
    alive = state["_arrays"]["alive"]
    return [pid for i, pid in enumerate(adapter._constants["player_ids"]) if i < len(alive) and alive[i] == 1]  # noqa: SLF001


def _act(adapter, state, template, **params):
    """Apply the first legal action matching ``template`` with the given
    param values (entity params are matched by their ``id``)."""
    for a in adapter.get_legal_actions(state):
        if a.template_id != template:
            continue
        matched = True
        for k, v in params.items():
            pv = a.params.get(k)
            if isinstance(pv, dict):
                pv = pv.get("id")
            if isinstance(v, dict):
                v = v.get("id")
            if pv != v:
                matched = False
                break
        if matched:
            return adapter.apply_action(state, a)
    raise AssertionError(f"not legal: {template} {params} at {state['env']['phase']}")


def _speak(adapter, state, text="发言"):
    for a in adapter.get_legal_actions(state):
        if a.template_id == "speak":
            return adapter.apply_action(state, replace(a, params={**a.params, "text": text}))
    raise AssertionError(f"speak not legal at {state['env']['phase']}")


def test_witch_phase_precedes_seer():
    """P1-16: 狼刀后先进入女巫夜（预言家存活也不跳过），女巫行动后到预言家夜。"""
    adapter = _engine(seed=7)
    rng = random.Random(7)
    st = play_until(adapter, rng, "night_wolf")
    assert st is not None
    roles = st["_arrays"]["roles"]
    witch_idx = roles.index("witch")
    seer_idx = roles.index("seer")
    # 刀一个村民（避开首夜预言家保护），女巫在场且有药
    target = next(
        pid
        for i, pid in enumerate(adapter._constants["player_ids"])
        if roles[i] == "villager" and st["_arrays"]["alive"][i] == 1
    )
    st = _act(adapter, st, "kill", target={"id": target})
    assert st["env"]["phase"] == "night_witch", f"phase {st['env']['phase']}"
    assert st["env"]["turn"] == f"p{witch_idx}"
    # 女巫救/毒后 → 预言家夜，turn=预言家
    heal = next(a for a in adapter.get_legal_actions(st) if a.template_id == "heal")
    st = adapter.apply_action(st, heal)
    assert st["env"]["phase"] == "night_seer", f"phase {st['env']['phase']}"
    assert st["env"]["turn"] == f"p{seer_idx}"


def test_dead_witch_skipped_even_with_potions():
    """P1-8: 女巫已死（药未用）时狼刀后跳过女巫夜，直接进预言家夜。"""
    adapter = _engine(seed=7)
    rng = random.Random(7)
    st = play_until(adapter, rng, "night_wolf")
    assert st is not None
    roles = st["_arrays"]["roles"]
    witch_idx = roles.index("witch")
    st["_arrays"]["alive"][witch_idx] = 0
    target = next(
        pid
        for i, pid in enumerate(adapter._constants["player_ids"])
        if roles[i] == "villager" and st["_arrays"]["alive"][i] == 1
    )
    st = _act(adapter, st, "kill", target={"id": target})
    assert st["env"]["phase"] == "night_seer", f"phase {st['env']['phase']}"


def test_wolf_phase_turn_is_wolf():
    """P1-5: night_wolf 入场 turn 必须是最小座位的存活狼人。"""
    adapter = _engine(seed=7)
    rng = random.Random(7)
    st = play_until(adapter, rng, "night_wolf")
    assert st is not None
    roles = st["_arrays"]["roles"]
    first_wolf = next(i for i, r in enumerate(roles) if r == "wolf")
    assert st["env"]["turn"] == f"p{first_wolf}"


def test_day_speech_rotation():
    """P1-7: 发言轮转由规则层推进 — speechLog[i].speaker 恒等于 living[i]。"""
    adapter = _engine(seed=203)
    rng = random.Random(203)
    st = play_until(adapter, rng, "day_speech")
    assert st is not None and adapter.get_node_type(st) == "player"
    living = _living_ids(adapter, st)
    assert st["env"]["turn"] == living[0]
    for i in range(len(living)):
        st = _speak(adapter, st, f"第{i}个发言")
        log = st["_arrays"]["speechLog"]
        assert log[-1]["speaker"] == living[i], f"speaker {log[-1]['speaker']} != {living[i]}"
        if i < len(living) - 1:
            assert st["env"]["turn"] == living[i + 1]
    # 最后一言后进入投票，turn 回到首位
    assert st["env"]["phase"] == "day_vote"
    assert st["env"]["voteIdx"] == 0
    assert st["env"]["turn"] == living[0]


def test_day_vote_rotation():
    """P1-7: 投票轮转同样由规则层推进 — voteLog[i].voter 恒等于 living[i]。"""
    adapter = _engine(seed=203)
    rng = random.Random(203)
    st = play_until(adapter, rng, "day_speech")
    assert st is not None
    living = _living_ids(adapter, st)
    for _ in range(len(living)):
        st = _speak(adapter, st)
    assert st["env"]["phase"] == "day_vote"
    for i in range(len(living)):
        voter = st["env"]["turn"]
        st = _act(adapter, st, "vote", target={"id": voter})
        log = st["_arrays"]["voteLog"]
        assert log[-1]["voter"] == living[i], f"voter {log[-1]['voter']} != {living[i]}"
    assert st["env"]["phase"] == "vote_resolve"


def test_witch_self_save_legal():
    """W5/P1-9: witch_self_save 开关真正生效 — 刀中女巫时 heal 仅在该开关下合法。"""
    for self_save, expect_heal in ((False, False), (True, True)):
        rules = gen_rules(witch_self_save=self_save, players=6, wolves=2, seers=1, with_hunter=False)
        engine = GameEngine(rules, seed=11)
        state = engine.create_initial_state()
        state["_arrays"]["roles"] = ["wolf", "wolf", "villager", "villager", "seer", "witch"]
        state["_arrays"]["alive"] = [1] * 6
        witch_id = "p5"
        state["env"].update(
            {"phase": "night_witch", "turn": witch_id, "nightKill": witch_id, "witchSaveUsed": 0, "witchPoisonUsed": 0}
        )
        acts = engine.get_legal_actions(state)
        ids = {a.template_id for a in acts}
        assert ("heal" in ids) is expect_heal, f"self_save={self_save}: heal={ids}"
        assert "poison" in ids
