"""BayesSolver — posterior-driven Werewolf player (SolverBase).

Decision layer over :class:`BeliefTracker`: every action follows from the
player's posterior over the other players' roles.

  - day_vote: 投后验狼概率最高者（对 Level-0 对手的解析最优）
  - night_wolf: 刀后验狼概率最低者（最不像狼的好人）
  - night_seer: 验当前熵最高者（信息增益最大）
  - night_witch: heal 被刀者（若药未用且非自己/按配置）；poison 后验最可疑狼
  - night_guard: 守神（预言家/女巫）优先
  - night_hunter / vote_hunter: 开枪后验最可疑狼（低置信时 pass）
  - day_speech: 按身份的模板发言（狼伪装 / 预言家报验 / 好人指控）

信念更新是增量的（只折叠新增的发言/投票），solver 实例跨决策持有
tracker。非法/无候选时 fallback 随机合法动作。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from layer2_engine.interfaces.solver_adapter import ActionInstance, SolverAdapter, State

from ..base import SolverBase, SolverConfig, SolverMetrics
from .belief import BeliefTracker

GOD_ROLES = ("seer", "witch", "guard", "hunter")


@dataclass
class BayesConfig(SolverConfig):
    guard_priority: tuple = GOD_ROLES  # 守卫优先保护的神
    poison_min_wolf: float = 0.45  # 毒药最低狼概率阈值
    shoot_min_wolf: float = 0.5  # 开枪最低狼概率阈值（低于则 pass）
    speech_llm: bool = False  # True 时发言委托 OllamaSolver（预留）


class BayesSolver(SolverBase):
    """Bayesian Werewolf player for one ``player_id``."""

    def __init__(self, adapter: SolverAdapter, config: SolverConfig | None = None, player_id: str | None = None):
        super().__init__(adapter, config or BayesConfig())
        self.player_id = player_id or self._default_player(adapter)
        self._tracker: BeliefTracker | None = None
        self._seen_speech = 0
        self._seen_votes = 0
        self._rng = random.Random(getattr(self.config, "seed", None))

    @staticmethod
    def _default_player(adapter: SolverAdapter) -> str:
        state = adapter.create_initial_state()
        return str(adapter.get_current_player(state) or "p0")

    @property
    def name(self) -> str:
        return f"Bayes@{self.player_id}"

    # ── SolverBase ──────────────────────────────────────────────────

    def select_action(self, state: State) -> ActionInstance | None:
        legal = self.adapter.get_legal_actions(state)
        if not legal:
            return None
        obs = self.adapter.get_observation(state, self.player_id)
        self._ensure_tracker(obs)
        self._fold_incremental(obs)

        phase = str(obs.get("phase"))
        action = None
        if phase == "day_vote":
            action = self._vote_action(legal)
        elif phase == "night_wolf":
            action = self._target_action(legal, self._least_wolfy, "kill")
        elif phase == "night_seer":
            action = self._target_action(legal, self._most_uncertain, "check")
        elif phase == "night_witch":
            action = self._witch_action(legal)
        elif phase == "night_guard":
            action = self._guard_action(legal)
        elif phase in ("night_hunter", "vote_hunter"):
            action = self._hunter_action(legal)
        elif phase == "day_speech":
            action = self._speech_action(legal, obs)
        return action if action is not None else self._fallback(legal)

    def reset(self) -> None:
        """重置信念状态——solver 跨对局复用时必须调用（M-08）。

        否则上一局的发言/投票/死亡证据会泄漏到下一局的后验里。
        """
        self._tracker = None
        self._seen_speech = 0
        self._seen_votes = 0

    def train(self, episodes: int = 100, **kwargs) -> SolverMetrics:
        """贝叶斯求解器无需训练。"""
        return SolverMetrics(episodes=0, win_rate=0.0, avg_return=0.0)

    # ── 信念管理 ────────────────────────────────────────────────────

    def _ensure_tracker(self, obs: dict) -> None:
        if self._tracker is None:
            self._tracker = BeliefTracker.from_adapter(
                self.adapter, self.player_id, seed=getattr(self.config, "seed", None)
            )
            self._seen_speech = 0
            self._seen_votes = 0

    def _fold_incremental(self, obs: dict) -> None:
        """只折叠新增的信号（观察是全量历史）。"""
        speech = list(obs.get("speech_log") or [])
        votes = list(obs.get("vote_log") or [])
        if self._tracker is not None and (len(speech) < self._seen_speech or len(votes) < self._seen_votes):
            # 观察历史比上次更短 → 新对局开始，重建信念防止跨局泄漏。
            self.reset()
            self._ensure_tracker(obs)
        if len(speech) > self._seen_speech or len(votes) > self._seen_votes:
            snapshot = dict(obs)
            snapshot["speech_log"] = speech[self._seen_speech :]
            snapshot["vote_log"] = votes[self._seen_votes :]
            self._seen_speech = len(speech)
            self._seen_votes = len(votes)
            self._tracker.update_from_observation(snapshot)

    def _alive_others(self, obs: dict) -> list[str]:
        alive = obs.get("alive") or []
        return [f"p{i}" for i, v in enumerate(alive) if v == 1 and f"p{i}" != self.player_id]

    # ── 决策 ────────────────────────────────────────────────────────

    def _target_action(self, legal: list[ActionInstance], rank_fn, template_id: str):
        """从 legal 中选 template_id 动作，target 按 rank_fn 排序取最优。"""
        cands = [
            (a, a.params.get("target", {}).get("id"))
            for a in legal
            if a.template_id == template_id and a.params.get("target")
        ]
        cands = [(a, t) for a, t in cands if t and t != self.player_id]
        if not cands:
            return None
        ranked = sorted(cands, key=lambda at: rank_fn(at[1]))
        return ranked[0][0] if ranked else None

    def _vote_action(self, legal: list[ActionInstance]):
        """投后验狼概率最高者（解析最优）。"""
        return self._target_action(legal, lambda t: -self._tracker.wolf_prob(t), "vote")

    def _least_wolfy(self, target: str) -> float:
        return self._tracker.wolf_prob(target)

    def _most_uncertain(self, target: str) -> float:
        return -self._tracker.entropy(target)

    def _witch_action(self, legal: list[ActionInstance]):
        """heal 被刀者；否则 poison 后验最可疑狼（置信达标）。"""
        for a in legal:
            if a.template_id == "heal":
                return a  # 目标由 domain 决定（被刀者）
        poison = self._target_action(legal, lambda t: -self._tracker.wolf_prob(t), "poison")
        if poison is not None:
            t = poison.params.get("target", {}).get("id")
            if t and self._tracker.wolf_prob(t) >= self.config.poison_min_wolf:
                return poison
        return None

    def _guard_action(self, legal: list[ActionInstance]):
        """守神优先：后验神概率最高者。"""
        cands = [
            (a, a.params.get("target", {}).get("id"))
            for a in legal
            if a.template_id == "guard" and a.params.get("target")
        ]
        cands = [(a, t) for a, t in cands if t and t != self.player_id]
        if not cands:
            return None

        def god_prob(t: str) -> float:
            return sum(self._tracker.prob(t, r) for r in self.config.guard_priority)

        return max(cands, key=lambda at: god_prob(at[1]))[0]

    def _hunter_action(self, legal: list[ActionInstance]):
        """开枪最可疑狼；置信不足 pass。"""
        shoot = self._target_action(legal, lambda t: -self._tracker.wolf_prob(t), "shoot")
        if shoot is None:
            shoot = self._target_action(legal, lambda t: -self._tracker.wolf_prob(t), "shoot_lynched")
        if shoot is not None:
            t = shoot.params.get("target", {}).get("id")
            if t and self._tracker.wolf_prob(t) >= self.config.shoot_min_wolf:
                return shoot
        for a in legal:
            if a.params.get("target", {}).get("id") == "pass":
                return a
        return None

    def _speech_action(self, legal: list[ActionInstance], obs: dict):
        """按身份的模板发言（LLM 委托预留）。"""
        speak = next((a for a in legal if a.template_id == "speak"), None)
        if speak is None:
            return None
        my_role = str(obs.get("my_role"))
        alive_others = self._alive_others(obs)
        if not alive_others:
            return self._with_speech(speak, "我没什么要说的。")
        if my_role == "wolf":
            # 狼：伪装村民，指控最不可疑者（好人）
            victim = min(alive_others, key=lambda t: self._tracker.wolf_prob(t))
            return self._with_speech(
                speak, f"我是普通村民。我怀疑{victim}，他的发言和投票都在带节奏，投票他。", intent="accuse"
            )
        if my_role == "seer":
            result = obs.get("seer_result")
            if result and result in self._alive_others(obs):
                return self._with_speech(
                    speak, f"我是预言家，昨晚验了{result}，{result}是狼！大家投{result}。", intent="accuse"
                )
            return self._with_speech(speak, "我是预言家，先观望一轮，有好消息再报。", intent="claim")
        # 好人：指控后验最可疑者
        sus = max(alive_others, key=lambda t: self._tracker.wolf_prob(t))
        if self._tracker.wolf_prob(sus) < 0.4:
            return self._with_speech(speak, "我先听一轮，暂时没有明确怀疑对象。", intent="question")
        return self._with_speech(speak, f"我怀疑{sus}是狼，投{sus}。", intent="accuse")

    @staticmethod
    def _with_speech(action: ActionInstance, text: str, intent: str = "claim"):
        from dataclasses import replace

        return replace(action, params={**action.params, "text": text, "intent": {"id": intent}})

    def _fallback(self, legal: list[ActionInstance]) -> ActionInstance | None:
        return legal[self._rng.randrange(len(legal))] if legal else None
