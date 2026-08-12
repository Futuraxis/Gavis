"""OllamaSolver — a local-LLM player (SolverBase) for werewolf & friends.

Each instance is bound to one ``player_id`` and talks to a local ollama
server (``qwen3:8b`` by default).  ``select_action`` builds a Chinese
prompt from the adapter's observation (role / alive / speech log / votes)
with a strict JSON output contract, parses the model's reply, and maps it
to an engine ``ActionInstance``:

  - speech phases:  {"intent": ..., "speech": "..."} → speak:{intent} + text
  - other phases:   {"target": "pX"} / {"target": "pass"} → the matching
    action template (kill/check/vote/poison/heal/guard/shoot/shoot_lynched)

Malformed / timed-out replies fall back to a random legal action, so the
game never stalls on a bad model output.  No external deps: the ollama
REST API is called via stdlib ``urllib``.

Usage::

    solver = OllamaSolver(adapter, OllamaConfig(model='qwen3:8b'), player_id='p3')
    action = solver.select_action(state)
"""

from __future__ import annotations

import json
import random
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from layer2_engine.interfaces.solver_adapter import ActionInstance, SolverAdapter, State

from ..base import SolverBase, SolverConfig, SolverMetrics

# 角色策略提示：每个身份一段简短行为指南（8B 模型的推理上限内的实用策略）
ROLE_GUIDE = {
    "wolf": (
        "你是狼人。白天要伪装成普通村民，不要暴露身份；"
        "可以伪装成预言家（悍跳）或指认其他玩家是狼；"
        "投票时优先投掉威胁大的好人（预言家、女巫）。"
    ),
    "seer": (
        "你是预言家。夜晚验人后，白天要巧妙公布验人结果引导好人；"
        "如果验出狼就指认它，注意别让狼人知道你是预言家。"
    ),
    "witch": (
        "你是女巫。解药只在关键时用（被刀的是预言家/女巫时优先救），"
        "毒药用来毒掉确认的狼人；不要随便浪费药。"
    ),
    "hunter": (
        "你是猎人。白天低调发言避免被狼人优先刀死；"
        "死亡开枪时带走你认为最可能是狼的玩家（不确定时选 pass 不开枪）。"
    ),
    "guard": (
        "你是守卫。夜晚保护你认为最可能被狼刀的人（预言家/女巫优先）；"
        "不能连续两晚守同一个人。"
    ),
    "villager": (
        "你是普通村民。白天听发言找狼：观察谁在说谎、谁在带节奏；"
        "投票给发言最可疑的玩家。"
    ),
}

TARGET_PHASES = ("night_wolf", "night_guard", "night_witch", "night_seer",
                 "night_hunter", "vote_hunter", "day_vote")
SPEECH_PHASES = ("day_speech",)


@dataclass
class OllamaConfig(SolverConfig):
    model: str = "qwen3:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    timeout: float = 120.0  # 冷启动加载模型可能很慢
    max_speech_log: int = 40  # 拼进 prompt 的最大发言条数
    fallback_seed: Optional[int] = None


class OllamaSolver(SolverBase):
    """LLM player for one ``player_id`` via a local ollama model."""

    def __init__(self, adapter: SolverAdapter, config: SolverConfig | None = None,
                 player_id: str | None = None):
        super().__init__(adapter, config or OllamaConfig())
        self.player_id = player_id or self._default_player(adapter)
        self._rng = random.Random(getattr(self.config, "fallback_seed", None))

    @staticmethod
    def _default_player(adapter: SolverAdapter) -> str:
        state = adapter.create_initial_state()
        return str(adapter.get_current_player(state) or "p0")

    @property
    def name(self) -> str:
        return f"Ollama({getattr(self.config, 'model', '?')}@{self.player_id})"

    # ── SolverBase ──────────────────────────────────────────────────

    def select_action(self, state: State) -> ActionInstance | None:
        legal = self.adapter.get_legal_actions(state)
        if not legal:
            return None
        obs = self.adapter.get_observation(state, self.player_id)
        try:
            reply = self._ask_model(self._build_prompt(obs, legal))
        except Exception:
            return self._fallback(legal)
        action = self._parse_reply(reply, legal)
        return action if action is not None else self._fallback(legal)

    def train(self, episodes: int = 100, **kwargs) -> SolverMetrics:
        """LLM 求解器无需训练；返回占位指标。"""
        return SolverMetrics(episodes=0, win_rate=0.0, avg_return=0.0)

    # ── Prompt ─────────────────────────────────────────────────────

    def _build_prompt(self, obs: dict, legal: list[ActionInstance]) -> str:
        cfg = self.config
        phase = str(obs.get("phase"))
        # 存活玩家 id 列表（obs['alive'] 是 [0/1] 数组，玩家命名 p0..pN-1）
        alive = [f"p{i}" for i, v in enumerate(obs.get("alive") or []) if v == 1]

        guide = ROLE_GUIDE.get(str(obs.get("my_role")), "你是普通玩家。")
        lines = [
            f"你是《狼人杀》玩家 {self.player_id}，身份是{obs.get('my_role')}。{guide}",
            "",
            f"当前：第 {obs.get('round')} 轮，阶段 {phase}",
            f"存活玩家：{alive}",
            f"昨夜/近日死亡：{obs.get('deaths_arr') or obs.get('deaths') or '无'}",
        ]
        if obs.get("seer_result"):
            lines.append(f"你的验人结果：{obs['seer_result']}")
        if obs.get("witch_save_used") is not None:
            lines.append(f"你已用解药：{bool(obs.get('witch_save_used'))}，"
                         f"已用毒药：{bool(obs.get('witch_poison_used'))}")
        speech = list(obs.get("speech_log") or [])[-cfg.max_speech_log:]
        if speech:
            lines.append("")
            lines.append("最近发言记录：")
            for s in speech:
                lines.append(f"  {s.get('speaker')}(第{s.get('round')}轮): "
                             f"[{s.get('intent')}] {s.get('text')}")
        votes = list(obs.get("vote_log") or [])
        if votes:
            lines.append("")
            lines.append(f"投票记录（共 {len(votes)} 条）：{votes[-20:]}")
        lines.append("")
        if phase in SPEECH_PHASES:
            lines.append(
                '请只输出一个 JSON 对象（不要任何其他文字），格式：'
                '{"intent": "claim|accuse|defend|question|persuade", "speech": "你的发言（一句话，中文）"}'
            )
        else:
            targets = sorted({a.params.get("target", {}).get("id", "")
                              for a in legal if a.params.get("target")})
            lines.append(f"可选目标：{targets}")
            lines.append(
                '请只输出一个 JSON 对象（不要任何其他文字），格式：{"target": "pX"}'
                '（放弃/不开枪时输出 {"target": "pass"}）'
            )
        return "\n".join(lines)

    # ── Model call ─────────────────────────────────────────────────

    def _ask_model(self, prompt: str) -> str:
        cfg = self.config
        body = json.dumps({
            "model": cfg.model,
            "stream": False,
            "options": {"temperature": cfg.temperature},
            "messages": [
                {"role": "system", "content": "你是狼人杀玩家，严格按照要求的 JSON 格式输出。"},
                {"role": "user", "content": prompt},
            ],
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{cfg.base_url}/api/chat", body, {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
            data = json.load(resp)
        return str(data.get("message", {}).get("content", ""))

    # ── Reply parsing ──────────────────────────────────────────────

    def _parse_reply(self, reply: str, legal: list[ActionInstance]) -> ActionInstance | None:
        reply = reply.strip()
        if not reply:
            return None
        try:
            # 去掉 ```json ... ``` 围栏
            m = re.search(r"\{.*\}", reply, re.S)
            if m is None:
                return None
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        # 用模板 id 匹配合法动作（speak 匹配意图槽，其余匹配 target）
        target = data.get("target")
        intent = data.get("intent")
        speech = data.get("speech")
        for a in legal:
            if a.template_id == "speak":
                if intent is not None and a.params.get("intent", {}).get("id") == str(intent):
                    return self._with_speech(a, speech)
                continue
            if target is not None and a.params.get("target", {}).get("id") == str(target):
                return a
        return None

    @staticmethod
    def _with_speech(action: ActionInstance, speech) -> ActionInstance:
        from dataclasses import replace

        return replace(action, params={**action.params, "text": str(speech or "")})

    def _fallback(self, legal: list[ActionInstance]) -> ActionInstance | None:
        return legal[self._rng.randrange(len(legal))] if legal else None
