import random
from collections import Counter
from dataclasses import replace

from layer2_engine.games.werewolf.werewolf_adapter import WerewolfAdapter


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
    adapter = WerewolfAdapter(seed=7)
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
    # 预言家自己能看到验人结果
    obs_seer = adapter.get_observation(st2, seer_id)
    assert obs_seer["seer_result"] == st2["env"]["seerResult"]
    assert obs_seer["my_role"] == "seer"
    # 其他玩家看不到
    obs_other = adapter.get_observation(st2, other_id)
    assert obs_other["seer_result"] is None
    assert obs_other["my_role"] != "seer"
    # 任何人都看不到别人的角色
    assert all(o["my_role"] is None or o["player"] == other_id for o in [obs_other])
    # 发言记录公开
    assert isinstance(obs_seer["speech_log"], list)


def test_observation_speech_log_is_public():
    """发言日志对所有玩家公开。"""
    st = None
    for s in range(20):  # 有些局第一夜后狼直接获胜，重试找进白天的局
        adapter = WerewolfAdapter(seed=200 + s)
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
        obs = adapter.get_observation(st2, pid)
        assert any("我是真预言家" in s.get("text", "") for s in obs["speech_log"])


# ── 胜负与收益 ────────────────────────────────────────────────────


def test_utility_by_faction():
    """狼赢：狼玩家 +1 好人 -1；好人赢：相反。"""
    adapter = WerewolfAdapter(seed=3)
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
        adapter = WerewolfAdapter(seed=100 + s, players=9, wolves=3)
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
