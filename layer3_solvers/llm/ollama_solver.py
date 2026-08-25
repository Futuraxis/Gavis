"""OllamaSolver — a local-LLM player (SolverBase) for werewolf & friends.

Each instance is bound to one ``player_id`` and talks to a local ollama
server (``qwen3:8b`` by default).  ``select_action`` builds a Chinese
prompt from the engine's observation (role / alive / speech log / votes)
with a strict JSON output contract, parses the model's reply, and maps it
to an engine ``ActionInstance``:

  - speech phases:  {"intent": ..., "speech": "..."} → speak:{intent} + text
  - other phases:   {"target": "pX"} / {"target": "pass"} → the matching
    action template (kill/check/vote/poison/heal/guard/shoot/shoot_lynched)

Malformed / timed-out replies fall back to a random legal action, so the
game never stalls on a bad model output.  No external deps: the ollama
REST API is called via stdlib ``urllib``.

Usage::

    solver = OllamaSolver(engine, OllamaConfig(model='qwen3:8b'), player_id='p3')
    action = solver.select_action(state)
"""

from __future__ import annotations

import json
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import ActionInstance, State

from ..base import SolverBase, SolverConfig, SolverMetrics
from ..werewolf.belief import belief_obs

# 角色策略提示：每个身份一段简短行为指南（8B 模型的推理上限内的实用策略）。
# 注意：guard 不在 ROLE_GUIDE 中 — 当前生成的 rules/werewolf.json 没有
# night_guard 阶段/protect 动作（with_guard=True 时守卫只是村民），
# 保留守卫提示属于死配置，等规则侧补上守卫再恢复。
ROLE_GUIDE = {
    "wolf": (
        "你是狼人。白天要伪装成普通村民，不要暴露身份；"
        "可以伪装成预言家（悍跳）或指认其他玩家是狼；"
        "投票时优先投掉威胁大的好人（预言家、女巫）。"
    ),
    "seer": (
        "你是预言家。夜晚验人后，白天要巧妙公布验人结果引导好人；如果验出狼就指认它，注意别让狼人知道你是预言家。"
    ),
    "witch": ("你是女巫。解药只在关键时用（被刀的是预言家/女巫时优先救），毒药用来毒掉确认的狼人；不要随便浪费药。"),
    "hunter": (
        "你是猎人。白天低调发言避免被狼人优先刀死；死亡开枪时带走你认为最可能是狼的玩家（不确定时选 pass 不开枪）。"
    ),
    "villager": ("你是普通村民。白天听发言找狼：观察谁在说谎、谁在带节奏；投票给发言最可疑的玩家。"),
}

SPEECH_PHASES = ("day_speech",)
# 意图枚举兜底（规则常量缺失时）：正常路径在 _build_prompt 里从
# 合法 speak 动作动态提取，避免与 rules intents 漂移（审查 P3-6）。
_FALLBACK_INTENTS = ("claim", "accuse", "defend", "question", "persuade")

# 发言清洗：剔除控制字符（含 \x00-\x1f、\x7f）——审计 3.6 prompt 注入修复
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


@dataclass
class OllamaConfig(SolverConfig):
    model: str = "qwen3:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    timeout: float = 120.0  # 冷启动加载模型可能很慢
    max_speech_log: int = 40  # 拼进 prompt 的最大发言条数
    max_speech_len: int = 200  # 模型输出发言的长度上限（注入防护）
    fallback_seed: Optional[int] = None


class OllamaSolver(SolverBase):
    """LLM player for one ``player_id`` via a local ollama model."""

    def __init__(self, engine: GameEngine, config: SolverConfig | None = None, player_id: str | None = None):
        super().__init__(engine, config or OllamaConfig())
        self.player_id = player_id or self._default_player(engine)
        # 单一 seed 旋钮：优先 SolverConfig.seed，fallback_seed 保留兼容
        # （审查 P3-5：双 seed 曾导致复现性混乱）。
        cfg = self.config
        seed = getattr(cfg, "seed", None)
        if seed is None:
            seed = getattr(cfg, "fallback_seed", None)
        self._rng = random.Random(seed)

    @staticmethod
    def _default_player(engine: GameEngine) -> str:
        """首个玩家的便宜推导（不展开初始状态 — 8 个 AI 席位曾重复 8 次发牌）。"""
        rules = getattr(engine, "rules", None)
        players = rules.get("players") if isinstance(rules, dict) else None
        if players:
            first = players[0]
            return str(first.get("id", first)) if isinstance(first, dict) else str(first)
        return "p0"

    @property
    def name(self) -> str:
        return f"Ollama({getattr(self.config, 'model', '?')}@{self.player_id})"

    # ── SolverBase ──────────────────────────────────────────────────

    def select_action(self, state: State) -> ActionInstance | None:
        legal = self.engine.get_legal_actions(state)
        if not legal:
            return None
        obs = self.engine.get_observation(state, self.player_id)
        flat = belief_obs(obs, self.player_id)
        try:
            reply = self._ask_model(self._build_prompt(flat, legal))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            # 网络/响应质量问题才随机回退；编程错误（如规则形状不匹配）
            # 不再被吞掉，上浮以便修复（审查 P1-15 曾被裸 except 掩盖）。
            return self._fallback(legal)
        action = self._parse_reply(reply, legal)
        return action if action is not None else self._fallback(legal)

    def train(self, episodes: int = 100, **kwargs) -> SolverMetrics:
        """LLM 求解器无需训练；返回占位指标。"""
        return SolverMetrics(episodes=0, win_rate=0.0, avg_return=0.0)

    # ── Prompt ─────────────────────────────────────────────────────

    def _build_prompt(self, obs: dict, legal: list[ActionInstance]) -> str:
        cfg = self.config
        if "env" in obs:
            # v5.2 view-shaped engine observation → 扁平（prompt 契约）
            obs = belief_obs(obs, self.player_id)
        phase = str(obs.get("phase"))
        # 存活玩家 id 列表（obs['alive'] 是 [0/1] 数组，玩家命名 p0..pN-1）
        alive = [f"p{i}" for i, v in enumerate(obs.get("alive") or []) if v == 1]

        def _yes_no(v) -> str:
            return "已用" if bool(v) else "未用"

        guide = ROLE_GUIDE.get(str(obs.get("my_role")), "你是普通玩家。")
        deaths = obs.get("deaths_arr") or obs.get("deaths") or []
        lines = [
            f"你是《狼人杀》玩家 {self.player_id}，身份是{obs.get('my_role')}。{guide}",
            "",
            f"当前：第 {obs.get('round')} 轮，阶段 {phase}",
            f"存活玩家：{alive}",
            # deathsArr 是死者 id 列表 — 直接列名而非 Python repr
            f"昨夜/近日死亡：{'、'.join(str(d) for d in deaths) if deaths else '无'}",
        ]
        if obs.get("seer_result"):
            lines.append(f"你的验人结果：{obs['seer_result']}")
        if obs.get("witch_save_used") is not None:
            lines.append(
                f"你已用解药：{_yes_no(obs.get('witch_save_used'))}，已用毒药：{_yes_no(obs.get('witch_poison_used'))}"
            )
        speech = list(obs.get("speech_log") or [])[-cfg.max_speech_log :]
        if speech:
            lines.append("")
            lines.append("最近发言记录：")
            for s in speech:
                lines.append(f"  {s.get('speaker')}(第{s.get('round')}轮): [{s.get('intent')}] {s.get('text')}")
        votes = list(obs.get("vote_log") or [])
        if votes:
            lines.append("")
            lines.append(f"投票记录（共 {len(votes)} 条）：")
            for v in votes[-20:]:
                lines.append(f"  {v.get('voter')} → {v.get('target')}（第{v.get('round')}轮）")
        lines.append("")
        if phase in SPEECH_PHASES:
            # 意图枚举从合法动作动态提取（审查 P3-6）：与 rules intents
            # 同源，改规则不会忘改 prompt 而错配。
            intents = sorted(
                {
                    str(a.params.get("intent", {}).get("id", ""))
                    for a in legal
                    if a.template_id == "speak" and isinstance(a.params.get("intent"), dict)
                }
            ) or list(_FALLBACK_INTENTS)
            lines.append(
                "请只输出一个 JSON 对象（不要任何其他文字），格式："
                f'{{"intent": "{"|".join(intents)}", "speech": "你的发言（一句话，中文）"}}'
            )
        elif phase == "night_witch":
            targets = sorted({t for a in legal if (t := self._target_of(a.params)) is not None})
            lines.append(f"可选目标：{targets}")
            lines.append(
                '请只输出一个 JSON 对象（不要任何其他文字），格式：{"potion": "heal"|"poison", "target": "pX"}'
                "（heal 的目标必须是当夜被狼刀者，且每瓶药只能使用一次）"
            )
        else:
            targets = sorted({t for a in legal if (t := self._target_of(a.params)) is not None})
            lines.append(f"可选目标：{targets}")
            lines.append(
                '请只输出一个 JSON 对象（不要任何其他文字），格式：{"target": "pX"}'
                '（放弃/不开枪时输出 {"target": "pass"}）'
            )
        return "\n".join(lines)

    # ── Model call ─────────────────────────────────────────────────

    def _ask_model(self, prompt: str) -> str:
        cfg = self.config
        body = json.dumps(
            {
                "model": cfg.model,
                "stream": False,
                "options": {"temperature": cfg.temperature},
                "messages": [
                    {"role": "system", "content": "你是狼人杀玩家，严格按照要求的 JSON 格式输出。"},
                    {"role": "user", "content": prompt},
                ],
            }
        ).encode("utf-8")
        req = urllib.request.Request(f"{cfg.base_url}/api/chat", body, {"Content-Type": "application/json"})
        # 决策记录（审计 3.6 阻塞 I/O，2026-08-13）：本调用会阻塞当前
        # 线程最长 timeout 秒且无重试——本地单人演示可接受；平台服务
        # 对外暴露前需改线程池/任务队列（P2，见 docs/design/security-notes.md）。
        with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
            data = json.load(resp)
        return str(data.get("message", {}).get("content", ""))

    # ── Reply parsing ──────────────────────────────────────────────

    @staticmethod
    def _target_of(params: dict) -> object:
        """Normalize an action param's target value.

        Player-entity params are dicts (``{"id": ...}``); night_witch
        ``heal``'s target is the raw ``nightKill`` string (the generator
        stores it as ``{"expr": [{"var": "$env.nightKill"}]}``), which
        crashed the old ``.get("id")`` calls (审查 P1-15).
        """
        t = params.get("target")
        return t.get("id") if isinstance(t, dict) else t

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
        potion = data.get("potion")
        if potion is not None:
            # night_witch 双动作（heal/poison）歧义：potion 字段显式区分
            # （审查 P1-17）。heal 的目标由 nightKill 固定（do_heal 只读
            # nightKill），无需核对 target；poison 按 target 匹配。
            potion = str(potion)
            for a in legal:
                if a.template_id != potion:
                    continue
                if potion == "heal":
                    return a
                if target is not None and self._target_of(a.params) == str(target):
                    return a
            return None
        for a in legal:
            if a.template_id == "speak":
                if intent is not None and a.params.get("intent", {}).get("id") == str(intent):
                    return self._with_speech(a, speech)
                continue
            if target is not None and self._target_of(a.params) == str(target):
                return a
        return None

    @staticmethod
    def _sanitize_speech(speech, max_len: int = 200) -> str:
        """发言清洗：长度上限 + 剔除控制字符（审计 3.6 prompt 注入）。"""
        text = _CONTROL_CHARS_RE.sub("", str(speech or ""))
        return text[:max_len]

    def _with_speech(self, action: ActionInstance, speech) -> ActionInstance:
        from dataclasses import replace

        text = self._sanitize_speech(speech, getattr(self.config, "max_speech_len", 200))
        return replace(action, params={**action.params, "text": text})

    def _fallback(self, legal: list[ActionInstance]) -> ActionInstance | None:
        return legal[self._rng.randrange(len(legal))] if legal else None
