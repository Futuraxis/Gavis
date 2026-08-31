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
game never stalls on a bad model output.  The transport is the project's
unified LLM client (``layer2_engine.core.llm.LLMClient``, OpenAI-compatible
endpoint via stdlib ``urllib``).

Usage::

    solver = OllamaSolver(engine, OllamaConfig(model='qwen3:8b'), player_id='p3')
    action = solver.select_action(state)
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Optional

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.llm import LLMClient, LLMConfig, sanitize_text
from layer2_engine.core.state_graph import ActionInstance, State

from ..base import SolverBase, SolverConfig, SolverMetrics
from ..werewolf.belief import belief_obs

logger = logging.getLogger(__name__)

# 角色策略提示：每个身份按难度(easy/normal/hard)给一段行为指南——
# hard = 强伪装/带节奏/精准推理;easy = 直白发盲/易露馅。8B 模型推理上限内。
# 注意：guard 不在 ROLE_GUIDE 中 — 当前生成的 rules/werewolf.json 没有
# night_guard 阶段/protect 动作（with_guard=True 时守卫只是村民），
# 保留守卫提示属于死配置，等规则侧补上守卫再恢复。
ROLE_GUIDE: dict[str, dict[str, str]] = {
    "wolf": {
        "easy": "你是狼人。白天发言尽量像村民，但可能不自觉暴露；可以尝试指认别人是狼。",
        "normal": "你是狼人。白天要伪装成普通村民，不要暴露身份；可以伪装成预言家（悍跳）或指认其他玩家是狼；投票时优先投掉威胁大的好人（预言家、女巫）。",
        "hard": "你是狼人。白天精心伪装：发言节奏与村民无异，适时带节奏把怀疑引向好人；可悍跳预言家抢话语权或栽赃他人；投票精准针对预言家/女巫等神职；被验出也不慌乱，继续制造混乱。",
    },
    "seer": {
        "easy": "你是预言家。白天可以公布验人结果，但可能表述不清被质疑。",
        "normal": "你是预言家。夜晚验人后，白天要巧妙公布验人结果引导好人；如果验出狼就指认它，注意别让狼人知道你是预言家。",
        "hard": "你是预言家。白天精准报验：验出狼果断指认并给出逻辑链；未验出时低调蛰伏避免被狼优先刀；警觉悍跳狼的反咬，用验人结果拆穿其伪装；报验时机与措辞都要讲究。",
    },
    "witch": {
        "easy": "你是女巫。有药要省着用，别乱救乱毒。",
        "normal": "你是女巫。解药只在关键时用（被刀的是预言家/女巫时优先救），毒药用来毒掉确认的狼人；不要随便浪费药。",
        "hard": "你是女巫。解药留给预言家/女巫等关键神职；毒药精准毒掉已确认狼（发言矛盾、被验出者）；刀中自己时权衡自救价值不轻易暴露身份；白天发言低调不暴露女巫身份，留意谁在试探女巫。",
    },
    "hunter": {
        "easy": "你是猎人。死亡时可以开枪带走一个可疑的人。",
        "normal": "你是猎人。白天低调发言避免被狼人优先刀死；死亡开枪时带走你认为最可能是狼的玩家（不确定时选 pass 不开枪）。",
        "hard": "你是猎人。白天发言朴素无锋芒降低被刀概率；被刀/被放逐开枪时精准带走最可疑狼（综合发言逻辑链判断）；不确定宁可 pass 不误杀好人；注意被毒无法开枪，警惕被女巫误毒。",
    },
    "villager": {
        "easy": "你是普通村民。听发言找狼，投票给可疑的人。",
        "normal": "你是普通村民。白天听发言找狼：观察谁在说谎、谁在带节奏；投票给发言最可疑的玩家。",
        "hard": "你是普通村民。白天仔细分析每条发言的逻辑漏洞与阵营倾向；识别悍跳狼与带节奏者；投票前综合验人/死亡/发言信息理性判断；警惕被狼带偏投票误放逐好人。",
    },
}

#: 谁是卧底发言难度提示（叠加在身份隐藏 prompt 上）。
#: 注意：玩家只看到自己的词、不知道另一个词（my_role 隐藏），提示只约束
#: 「自己词的专属度」——挑泛化层面、避独特细节；露馅与否靠听别人的发言判断。
#: easy=简单描述;normal=克制模糊、每句只给一个泛化特征;hard=高度讲究模仿/识破。
_UNDERCOVER_HINT: dict[str, str] = {
    "easy": "简单一句话描述你的词即可（如特征、用途），不必太讲究策略。",
    "normal": "描述要克制：你只看到自己的词、不知道另一个词，所以宁可泛不可精——一句话只给一个泛泛的点（类别、外观、用途、手感任选其一），不要在同一句话里堆叠多个具体特征（例如材质、结构、用途、手感一起说）：描述越专属，若你是卧底越容易被识破。",
    "hard": "描述要高度讲究：平民要听出卧底描述的细微差异，卧底要精准模仿平民描述风格不露破绽；你只看到自己的词，就挑它最泛化的层面来说（类别、常见用法），避开最独特、只有它才成立的细节；露馅与否从别人的措辞与信息量判断，随时据此调整自己的说法。",
}

#: pacing → 发言温度：fast 发散(易露馅/不精准)、standard 居中、slow 精准(强伪装/强推理)。
#: 与平台 ``pacing`` 契约(fast/standard/slow)对齐——社交游戏难度第二维。
_PACING_TEMP: dict[str, float] = {"fast": 0.9, "standard": 0.7, "slow": 0.5}

#: 角色 id → 中文名（prompt 里给模型读“狼人”而不是裸 id ``wolf``）。
#: 模型输出仍是机器键（intent/target），这里只是输入侧读法 —— 属于
#: 「传给 LLM 的信息不过分技术化」的 Layer-3 出口（Layer 3 不依赖 Layer 4，
#: 本地维护这份最小映射，与 rules/werewolf.json 的 role 常量对齐）。
_ROLE_NAMES = {
    "wolf": "狼人",
    "seer": "预言家",
    "witch": "女巫",
    "hunter": "猎人",
    "villager": "村民",
}

SPEECH_PHASES = ("day_speech",)
# 意图枚举兜底（规则常量缺失时）：正常路径在 _build_prompt 里从
# 合法 speak 动作动态提取，避免与 rules intents 漂移（审查 P3-6）。
_FALLBACK_INTENTS = ("claim", "accuse", "defend", "question", "persuade")


@dataclass
class OllamaConfig(SolverConfig):
    model: str = "qwen3:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    timeout: float = 120.0  # 冷启动加载模型可能很慢
    max_speech_log: int = 40  # 拼进 prompt 的最大发言条数
    max_speech_len: int = 200  # 模型输出发言的长度上限（注入防护）
    fallback_seed: Optional[int] = None
    # 难度两维（与平台 difficulty×pacing 3×3 契约对齐）：
    #   difficulty → 策略提示强度（ROLE_GUIDE 分档 + 卧底提示 + undercover 词对档）；
    #   pacing → 发言温度（fast 发散 / standard 居中 / slow 精准）。
    difficulty: str = "normal"
    pacing: str = "standard"


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
        # 统一 LLM 客户端（layer2_engine.core.llm）——传输层唯一实现。
        self._llm = LLMClient(
            LLMConfig(
                model=cfg.model,
                base_url=cfg.base_url,
                timeout_s=cfg.timeout,
                temperature=cfg.temperature,
            )
        )
        #: 最近一次 select_action 是否成功走 LLM 决策（False = 传输失败/
        #: 输出不可用随机兜底）——调用方（如平台社交族）据此如实标注 ai_mode。
        self.last_call_ok = True

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
        self.last_call_ok = True
        try:
            reply = self._ask_model(self._build_prompt(flat, legal))
        except (TimeoutError, OSError):
            # 网络/响应质量问题才随机回退；编程错误（如规则形状不匹配）
            # 不再被吞掉，上浮以便修复（审查 P1-15 曾被裸 except 掩盖）。
            # 传输已由统一客户端 fail-soft 化，这里仅兜测试注入的异常。
            self.last_call_ok = False
            logger.warning("Ollama 求解器 %s 传输异常，随机兜底", self.player_id)
            return self._fallback(legal)
        if not reply:
            # 统一客户端 fail-soft：真实失败原因在 last_error（API 4xx/5xx、
            # 端点不可达等）——记录而不是把「模型在线但报错」误当「离线」。
            last = self._llm.last_error
            self.last_call_ok = False
            logger.warning(
                "Ollama 求解器 %s 未获得模型回复%s，随机兜底",
                self.player_id,
                f"（{last}）" if last is not None else "",
            )
            return self._fallback(legal)
        action = self._parse_reply(reply, legal)
        if action is None:
            self.last_call_ok = False
            logger.warning("Ollama 求解器 %s 输出不可解析，随机兜底", self.player_id)
            return self._fallback(legal)
        return action

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

        my_word = str(obs.get("my_word") or "")
        deaths = obs.get("deaths_arr") or obs.get("deaths") or []
        if my_word:
            # 谁是卧底：玩家只看自己的词，不知自己是平民/卧底/白板（my_role 隐藏）。
            # 白板的词是「白板」→ 靠词自知是白板；平民/卧底看真词，靠发言推断阵营。
            word_line = "你没有词——你是白板（靠听别人描述混入，存活到末尾或自爆猜词可胜）" if my_word == "白板" else f"你的词是「{my_word}」"
            hint = _UNDERCOVER_HINT.get(cfg.difficulty, "")
            lines = [
                f"你是《谁是卧底》玩家 {self.player_id}。{word_line}。"
                "你不知道自己是平民还是卧底——若你的词和多数人描述一致，你多半是平民；"
                "若不一致，你可能是卧底（可自爆猜对平民词直接取胜，但猜错或你是平民会出局）。",
                f"发言策略：{hint}" if hint else "",
                "",
                f"当前：第 {obs.get('round')} 轮，阶段 {phase}",
                f"存活玩家：{alive}",
                f"已出局：{'、'.join(str(d) for d in deaths) if deaths else '无'}",
            ]
        else:
            role_id = str(obs.get("my_role"))
            role_name = _ROLE_NAMES.get(role_id, role_id)
            guide = ROLE_GUIDE.get(role_id, {}).get(cfg.difficulty, "你是普通玩家。")
            lines = [
                f"你是《狼人杀》玩家 {self.player_id}，身份是{role_name}（{role_id}）。{guide}",
                "",
                f"当前：第 {obs.get('round')} 轮，阶段 {phase}",
                f"存活玩家：{alive}",
                # deathsArr 是死者 id 列表 — 直接列名而非 Python repr
                f"昨夜/近日死亡：{'、'.join(str(d) for d in deaths) if deaths else '无'}",
            ]
        if not my_word:
            # 狼人杀私密信息（谁是卧底无这些字段，跳过）
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
        # v5.2 通用自由文本发言（谁是卧底 describe 等）：凡合法动作里有
        # ``speak + text`` 槽即可判定为发言阶段——不限于狼人杀的 day_speech。
        # 带 intent 槽的 speak（狼人杀）保留意图契约；无 intent 槽的 free-text
        # speak（卧底 describe）走 ``{"speech": "..."}`` 纯文本契约。
        # 注意按**参数键存在性**判定（不是 isinstance(dict)）：text 槽在
        # ActionInstance.params 里是占位 ``""``（字符串），意图槽才是 dict
        # （``{"id": "claim" | "_index"/"value" 等}``）——按值类型判断会让
        # 卧底的 free-text 分支永远不触发、狼人杀无 speak 动作时误走 free-text。
        has_text_speak = any(a.template_id == "speak" and "text" in a.params for a in legal)
        has_intent_speak = any(a.template_id == "speak" and "intent" in a.params for a in legal)
        if phase in SPEECH_PHASES or (has_text_speak and not has_intent_speak):
            if has_intent_speak or phase in SPEECH_PHASES:
                # 意图契约（狼人杀 day_speech）：意图枚举从合法动作动态提取
                # （审查 P3-6，与 rules intents 同源）；day_speech 但无 speak
                # 动作（理论空窗口）时回退 _FALLBACK_INTENTS，契约不丢。
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
            else:
                # free-text speak（谁是卧底：每人一句话描述自己的词）——
                # 没有意图槽，模型只需产出发言文本。
                lines.append(
                    "请只输出一个 JSON 对象（不要任何其他文字），格式："
                    '{"speech": "你的发言（一句话，中文，描述你的词；不能说出词本身，也不能一口气罗列多个专属特征）"}'
                )
        elif any(a.template_id == "self_destruct" for a in legal):
            # 谁是卧底投票阶段：可投票放逐，或自爆猜词。
            vote_targets = sorted(
                {t for a in legal if a.template_id == "vote" and (t := self._target_of(a.params)) is not None}
            )
            lines.append(f"可投票目标：{vote_targets}")
            lines.append(
                "请只输出一个 JSON 对象（不要任何其他文字）。"
                "投票放逐：{\"target\": \"pX\"}；"
                "自爆猜词：{\"action\": \"self_destruct\", \"target\": \"pX\", \"guess\": \"你猜的词\"}"
                "（自爆：若你是卧底且猜对平民词→你胜；猜错或你是平民→你出局）"
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
        """Ask the unified LLM client (fail-soft: ``""`` on transport failure).

        决策记录（审计 3.6 阻塞 I/O，2026-08-13）：本调用会阻塞当前线程最长
        timeout 秒且无重试——本地单人演示可接受；平台服务对外暴露前需改线程池
        /任务队列（P2，见 docs/design/security-notes.md）。"""
        temp = _PACING_TEMP.get(self.config.pacing)
        return self._llm.complete_chat(
            "你是派对桌游玩家，严格按照要求的 JSON 格式输出。",
            prompt,
            temperature=temp,
        )

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
        action_kind = data.get("action")
        # 谁是卧底自爆：{"action":"self_destruct","target":"pX","guess":"词"}
        if action_kind == "self_destruct":
            guess = data.get("guess")
            for a in legal:
                if a.template_id != "self_destruct":
                    continue
                if target is not None and self._target_of(a.params) == str(target):
                    return self._with_param(a, "guess", guess)
            return None
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
                intent_spec = a.params.get("intent")
                # 狼人杀：speak 带 intent 槽 → 按意图匹配（审查 P3-6 同源动态提取）。
                if isinstance(intent_spec, dict) and intent_spec.get("id") is not None:
                    if intent is not None and str(intent_spec.get("id")) == str(intent):
                        return self._with_speech(a, speech)
                    continue
                # v5.2 自由文本 speak（谁是卧底 describe 等）：无 intent 槽，
                # 模型给出 speech 文本即直接附上。
                if speech:
                    return self._with_speech(a, speech)
                continue
            if target is not None and self._target_of(a.params) == str(target):
                return a
        return None

    @staticmethod
    def _sanitize_speech(speech, max_len: int = 200) -> str:
        """发言清洗：长度上限 + 剔除控制字符（统一清洗）。"""
        return sanitize_text(str(speech or ""), max_len)

    def _with_speech(self, action: ActionInstance, speech) -> ActionInstance:
        from dataclasses import replace

        text = self._sanitize_speech(speech, getattr(self.config, "max_speech_len", 200))
        return replace(action, params={**action.params, "text": text})

    def _with_param(self, action: ActionInstance, name: str, value) -> ActionInstance:
        """把自由文本参数（如自爆的 guess）清洗后塞进 ActionInstance.params。"""
        from dataclasses import replace

        cleaned = self._sanitize_speech(value, getattr(self.config, "max_speech_len", 200))
        return replace(action, params={**action.params, name: cleaned})

    def _fallback(self, legal: list[ActionInstance]) -> ActionInstance | None:
        return legal[self._rng.randrange(len(legal))] if legal else None
