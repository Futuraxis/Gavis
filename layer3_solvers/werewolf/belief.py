"""BeliefTracker — Bayesian role inference for Werewolf.

The game starts with a common prior: the known role pool (e.g. 3 wolves /
3 villagers / seer / witch / hunter for 9 players) minus one's own role.
Public history — speeches (intent + text), votes, and *revealed deaths*
(dead roles are published) — updates each other player's posterior via
Bayes' rule with a heuristic likelihood model:

    P(tᵢ | h) ∝ P(h | tᵢ) · P(tᵢ)

The likelihood model is deliberately simple and tunable: accusing/voting
someone makes the target more wolf-likely (and the actor slightly more
so); a dead player's role is exact evidence.  A ``signal_fn`` hook allows
replacing the heuristic with an LLM scorer later (same interface).

Sampling: a full role assignment consistent with the role-count constraint
is drawn **without replacement** from the remaining pool, weighted by each
player's marginal posterior — the natural joint for turn-based play.

Pure probability — no engine dependency beyond reading the observation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# ── 启发式似然参数：P(信号 | 角色) 的乘性权重 ───────────────────────
# 数值 > 1 表示该信号在该角色下更常见。调参即调策略。
LIKELIHOOD = {
    "vote_target": {"wolf": 1.5, "good": 1.0},      # 被投票者更像狼
    "voter": {"wolf": 1.15, "good": 1.0},            # 投狼票的投票者略像狼
    "accuse_target": {"wolf": 1.6, "good": 1.0},     # 被指控者更像狼
    "accuser": {"wolf": 1.1, "good": 1.0},           # 指控者略像狼
    "claim_seer": {"wolf": 0.35, "good": 1.0},       # 声称预言家：好人更可信
    "claimed_wolf": {"wolf": 2.0, "good": 1.0},      # 被指为狼
}

ROLE_KEY = "role"  # 观察里的角色字段（'my_role'）


@dataclass
class BeliefTracker:
    """Per-player posterior role distributions + joint sampling."""

    players: list[str]
    role_pool: list[str]              # 完整角色池（含自己）
    my_role: str
    signal_fn=None                    # 可选：LLM 打分器（预留）
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        # 边际后验：{player: {role: weight}}，初始为条件先验。
        # 池 = 完整角色池移除"一个"自己的角色（同角色其余保留）；
        # 权重按池计数（3 个狼的先验是 1 个预言家的 3 倍）。
        pool = list(self.role_pool)
        if self.my_role in pool:
            pool.remove(self.my_role)
        self._pool = pool
        counts: dict[str, int] = {}
        for r in pool:
            counts[r] = counts.get(r, 0) + 1
        self._post: dict[str, dict[str, float]] = {}
        for p in self.players:
            if p == self._self_id():
                self._post[p] = {self.my_role: 1.0}
            else:
                self._post[p] = {r: float(c) for r, c in counts.items()}
                self._normalize(p)
        self._known: dict[str, str] = {}  # 确定角色（自己 + 已公布死亡）

    def _self_id(self) -> str:
        return self.players[0] if self.players else ""

    # ── 先验 ────────────────────────────────────────────────────────

    @classmethod
    def from_adapter(cls, adapter, player_id: str, seed: int | None = None) -> "BeliefTracker":
        """Build from a WerewolfAdapter (players / role pool / own role)."""
        pids = list(adapter._constants.get("player_ids", []))
        pool = list(adapter._constants.get("role_pool", []))
        obs = adapter.get_observation(adapter.create_initial_state(), player_id)
        return cls(pids, pool, str(obs.get("my_role")), rng=random.Random(seed))

    # ── 观察与更新 ──────────────────────────────────────────────────

    def update_from_observation(self, obs: dict) -> None:
        """Fold one observation's public signals into the posteriors."""
        self._fold_deaths(obs)
        self._fold_speech(obs)
        self._fold_votes(obs)

    def _fold_deaths(self, obs: dict) -> None:
        """已公布死亡的角色是确定性证据（死后公布身份）。"""
        for pid, dead_role in (obs.get("dead_roles") or {}).items():
            if dead_role and pid in self._post:
                self._set_known(pid, dead_role)

    def _fold_speech(self, obs: dict) -> None:
        for s in obs.get("speech_log") or []:
            speaker = str(s.get("speaker"))
            if speaker not in self._post:
                continue
            intent = str(s.get("intent") or "")
            text = str(s.get("text") or "")
            if intent == "accuse":
                target = _find_target(text, self.players)
                if target:
                    self._apply(speaker, "accuser")
                    self._apply(target, "accuse_target")
            elif intent == "claim":
                if "预言家" in text or "seer" in text or "验" in text:
                    self._apply(speaker, "claim_seer")
            if "狼" in text:
                target = _find_target(text, self.players)
                if target and target != speaker:
                    self._apply(target, "claimed_wolf")

    def _fold_votes(self, obs: dict) -> None:
        for v in obs.get("vote_log") or []:
            voter = str(v.get("voter"))
            target = str(v.get("target"))
            if voter in self._post and target in self._post:
                self._apply(target, "vote_target")
                self._apply(voter, "voter")

    # ── 核心更新 ────────────────────────────────────────────────────

    def _apply(self, player: str, signal: str) -> None:
        """乘性贝叶斯更新：post[player][role] *= P(signal | role)。"""
        w = LIKELIHOOD.get(signal)
        if w is None or player not in self._post:
            return
        post = self._post[player]
        for role, weight in w.items():
            if role in post:
                post[role] *= weight
        self._normalize(player)

    def _set_known(self, player: str, role: str) -> None:
        """确定角色（死亡公布）：固定后验并从池中移除。

        池中该角色用尽时，其他人的该角色概率清零（计数约束）并归一化。
        """
        self._known[player] = role
        if player in self._post:
            self._post[player] = {role: 1.0}
        if role in self._pool:
            self._pool.remove(role)
            if role not in self._pool:
                # 该角色已全部确定归属 → 其他人后验清零
                for p in self._post:
                    if p != player and role in self._post[p]:
                        del self._post[p][role]
                        self._normalize(p)

    def _normalize(self, player: str) -> None:
        post = self._post[player]
        total = sum(post.values())
        if total > 0:
            for r in post:
                post[r] /= total

    # ── 查询 ────────────────────────────────────────────────────────

    def wolf_prob(self, player: str) -> float:
        """P(player 是狼 | 历史)。"""
        return self._post.get(player, {}).get("wolf", 0.0)

    def prob(self, player: str, role: str) -> float:
        return self._post.get(player, {}).get(role, 0.0)

    def most_suspicious(self, exclude: set[str] | None = None) -> str | None:
        """后验狼概率最高的玩家（排除集合外）。"""
        exclude = exclude or set()
        best, best_p = None, -1.0
        for p, post in self._post.items():
            if p in exclude or p == self._self_id():
                continue
            w = post.get("wolf", 0.0)
            if w > best_p:
                best, best_p = p, w
        return best

    def entropy(self, player: str) -> float:
        """角色后验熵（信息增益决策用）。"""
        post = self._post.get(player, {})
        return -sum(p * math.log(p) for p in post.values() if p > 0)

    # ── 联合采样 ────────────────────────────────────────────────────

    def sample_assignment(self, exclude: set[str] | None = None) -> dict[str, str]:
        """从联合后验采样一个完整角色分配（角色计数一致，无放回）。

        顺序采样：按狼概率降序确定狼（信息最强者先定），其余按权重从
        剩余池抽取——保证与 role_pool 计数一致。
        """
        exclude = exclude or set()
        pool = list(self._pool)
        assign: dict[str, str] = {}
        for p, r in self._known.items():
            if p not in exclude:
                assign[p] = r
                if r in pool:
                    pool.remove(r)
        remaining = [p for p in self.players
                     if p not in assign and p not in exclude and p != self._self_id()]
        # 先定狼：按狼概率降序
        ordered = sorted(remaining, key=lambda p: -self.wolf_prob(p))
        for p in ordered:
            if "wolf" in pool:
                # 以 wolf_prob 为权重决定是否给狼（否则留给后续角色）
                if self.rng.random() < self.wolf_prob(p) * (len(pool) / max(1, len(remaining))):
                    assign[p] = "wolf"
                    pool.remove("wolf")
                    continue
            # 其余角色按后验权重无放回
            weights = [self._post.get(p, {}).get(r, 0.0) for r in pool]
            total = sum(weights)
            if total <= 0:
                assign[p] = self.rng.choice(pool)
            else:
                assign[p] = self.rng.choices(pool, weights=weights, k=1)[0]
            pool.remove(assign[p])
        return assign


def _find_target(text: str, players: list[str]) -> str | None:
    """从发言文本中提取被指认的玩家（p0..pN-1）。"""
    for pid in players:
        if pid in text:
            return pid
    return None
