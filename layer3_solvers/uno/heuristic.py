"""UNO rollout 启发式策略 — 给 MCTS 的 rollout 注入轻量先验。

裸 MCTS 在 UNO 上的 rollout 是均匀随机——统计信号弱（4 人局随机走难以
在 ``rollout_depth`` 内到终局，非终局 ``leaf_value`` 恒 0）。本策略按
真实 UNO 常识排序出牌：

- 能赢即赢（手牌仅剩 1 张且能出）。
- 反击叠加（stack2/stack4）优先——转嫁罚牌给下家。
- 特殊牌（draw2/skip/reverse）次之——打击下家 + 高罚分（20）。
- 数字牌按面值降序——优先打出高罚分牌。
- wild4 中等优先、wild 保留（灵活兜底）。
- 无牌可出时 draw 优先于 pass（尽快推进局面）。

带 30% 随机回退保留探索性（仿 ``BoardHeuristicPolicy`` 的
``block_prob``）；接口与 ``MCTS.rollout_policy`` 一致：
``fn(state, actions) -> ActionInstance | None``（None → 调用方回退随机）。
纯字符串解析 card id，不查规则常量——rollout 每步调用，必须 <1ms。

注意：这是 **rollout 先验**，非完整策略——只在裸 MCTS 的随机 rollout
路径生效（hybrid 无 ``hiddenWorld`` / 无 CFR 表时）。PIMC（``_opponent_mcts``）
路径用 hybrid 自己的 ``_rollout_prior``，不受此影响。
"""

from __future__ import annotations

import random
from typing import Optional

from layer2_engine.core.state_graph import ActionInstance, State

#: 出牌类动作模板（主动出牌 / 抢牌）。
_PLAY_TIDS = frozenset({"play", "play_wild", "play7", "play_drawn", "jump_play"})
#: 罚牌/叠加类动作（反击叠加优先，被动吃罚低优先）。
_STACK_TIDS = frozenset({"stack2", "stack4", "take_penalty"})
#: 摸牌动作。
_DRAW_TID = "draw"
#: 过牌动作（摸牌后无可打 / 抢牌窗口放弃）。
_PASS_TIDS = frozenset({"pass", "jump_pass"})

#: 启发式命中率（其余回退随机，保持 rollout 探索性）。
_HIT_PROB = 0.7


def _split_card(card: str) -> tuple[str, str]:
    """card id → (color, symbol)。纯字符串解析，不查规则常量。

    - ``r5a`` → ('r', '5')；``rsa`` → ('r', 's'=skip)；``rda`` → ('r', 'd'=draw2)
    - ``r0`` → ('r', '0')（0 片单张无 instance 后缀）
    - ``wild4_1`` → ('wild', 'wild4')；``wild_1`` → ('wild', 'wild')
    """
    if card.startswith("wild4_"):
        return "wild", "wild4"
    if card.startswith("wild_"):
        return "wild", "wild"
    # {color}{symbol}{instance}：color/symbol 各 1 字符（symbol=数字或 s/r/d）
    return card[0], card[1]


def _play_score(action: ActionInstance) -> float:
    """出牌得分（高=优先）。反击 > 特殊牌 > 大点数数字 > wild4 > wild。

    无 ``card`` 参数的出牌动作（如纯叠加）按 template 评级，保证罚牌
    场景下 stack 反击优先于 take_penalty 被动吃下。
    """
    tid = action.template_id
    if tid in ("stack2", "stack4"):
        return 200.0  # 反击叠加——转嫁罚牌给下家，最强
    if tid == "take_penalty":
        return 10.0  # 被动吃罚——最后手段
    card = action.params.get("card")
    if not card:
        return 0.0
    _color, symbol = _split_card(card)
    if symbol == "d":  # draw2：罚 2 张 + 跳过下家
        return 125.0
    if symbol == "s":  # skip
        return 120.0
    if symbol == "r":  # reverse
        return 120.0
    if symbol == "wild4":  # 强打击但留作终结手段
        return 90.0
    if symbol == "wild":  # 灵活，保留倾向
        return 40.0
    if symbol.isdigit():
        return float(symbol)  # 数字牌：面值 0-9，大点数优先（减少潜在罚分）
    return 0.0


class UnoRolloutPolicy:
    """MCTS rollout 启发式：UNO 出牌优先级 + 探索性回退。"""

    def __init__(self, engine: object | None = None, seed: int = 0) -> None:
        #: engine 留作接口一致（工厂签名 ``(engine, seed)``）；当前策略
        #: 纯字符串解析 card id，不查引擎常量。
        self._engine = engine
        self._rng = random.Random(seed)

    def __call__(self, state: State, actions: list[ActionInstance]) -> Optional[ActionInstance]:
        plays: list[ActionInstance] = []
        draws: list[ActionInstance] = []
        passes: list[ActionInstance] = []
        for a in actions:
            tid = a.template_id
            if tid in _PLAY_TIDS or tid in _STACK_TIDS:
                plays.append(a)
            elif tid == _DRAW_TID:
                draws.append(a)
            elif tid in _PASS_TIDS:
                passes.append(a)
            else:
                plays.append(a)  # 未知出牌类归入 plays（保守优先出，避免卡死）

        # 无牌可出 → 推进摸牌（draw 优先于 pass，尽快给局面引入新牌）
        if not plays:
            if draws:
                return draws[0]
            if passes:
                return passes[0]
            return None  # 无可推进一步——回退随机

        env = state.get("env", {}) if isinstance(state, dict) else {}
        arrs = state.get("_arrays", {}) if isinstance(state, dict) else {}
        turn = env.get("turn")
        hand = list(arrs.get(f"hand_{turn}", [])) if turn else []

        # 能赢即赢：手牌仅剩 1 张且能出 → 出了即清空获胜（不探索必胜）
        if len(hand) <= 1:
            return plays[0]

        # 命中率外回退随机，保持 rollout 探索性（防止纯贪心让 MCTS 高估）
        if self._rng.random() >= _HIT_PROB:
            return None

        return max(plays, key=_play_score)
