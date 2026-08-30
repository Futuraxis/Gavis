"""Persona — Agent 人格定义与离线兜底台词（Layer 4，确定性半边）.

四种性格里 P0 落地 ``gentle``（温柔陪伴）与 ``teacher``（认真教学），
``banter`` / ``cold`` 给结构完整但台词精简的占位 :class:`Persona`。
每个 :class:`Persona` 都为九个 :data:`SCENARIOS` 场景提供至少一条
兜底台词，保证无 LLM 时依然可离线表达（对话引擎回退路径）。
"""

from __future__ import annotations

from dataclasses import dataclass

from .scenarios import SCENARIOS


@dataclass(frozen=True)
class Persona:
    """一种 Agent 人格：展示元信息 + 每场景离线兜底台词表.

    Attributes:
        key: 稳定键名（gentle / teacher / banter / cold）。
        display_name: 面向玩家的中文名。
        verbosity: 0（高冷少言）～ 2（健谈）。
        tone: 语气标签（温柔 / 教学 / 吐槽 / 高冷）。
        fallback_lines: scenario → 兜底台词表，供无 LLM 时确定性成文。
    """

    key: str
    display_name: str
    verbosity: int
    tone: str
    fallback_lines: dict[str, list[str]]


PERSONAS: dict[str, Persona] = {
    "gentle": Persona(
        key="gentle",
        display_name="温柔陪伴",
        verbosity=1,
        tone="温柔",
        fallback_lines={
            "greet": ["嗨，你来啦，我们慢慢玩。", "欢迎回来，想先玩哪一局都行。"],
            "good_move": ["这步走得很稳，很漂亮。", "好棋，看得出你用心了。"],
            "blunder": ["这步有点冒险，先看看有没有更好的选择。", "没关系，这里可能有点风险，我们再想想。"],
            "help": ["建议先守住关键的位置，别急着进攻。", "可以看看哪条线空位多，往那边靠。"],
            "ai_win": ["这局我运气好一点，你的思路其实是对的。", "承让啦，我们再来一局？"],
            "ai_lose": ["你赢了，这手翻得很漂亮！", "被你抓住了机会，下得真棒。"],
            "illegal": ["这一步现在不能走，规则上还差一点点。", "稍等哦，这个动作暂时不合规。"],
            "idle": ["还在想吗？不着急，慢慢来。", "我在呢，你慢慢考虑。"],
            "game_over": ["这局打完了，要不要一起看看哪一手最关键？", "结束啦，辛苦了，想复盘的话我陪你。"],
            "teach_greet": ["这局是教学局，你的牌我看得见，我们一步一步来。", "教学局开始啦，我会边看你的牌边讲思路。"],
            "teach_turn": ["到你了，先看看手里的牌，慢慢想。", "轮到你啦，我们看看这手牌怎么处理。"],
            "teach_move": ["这手打完啦，我说说我的想法。", "落定啦，一起看看这步怎么样。"],
        },
    ),
    "teacher": Persona(
        key="teacher",
        display_name="认真教学",
        verbosity=2,
        tone="教学",
        fallback_lines={
            "greet": ["欢迎，开局前我先把规则要点讲给你听。", "来，我们先熟悉规则，再开始对局。"],
            "good_move": ["这步占角很好，后续空间更大。", "好棋，这一步符合先占角后连边的原则。"],
            "blunder": ["这步给了对面连三的机会，风险要留意。", "这里建议先补位，对面下一步可能连成。"],
            "help": ["方向上优先占角和边线，具体说这手先占角更好。", "可以按占角、守边、连三的顺序来。"],
            "ai_win": ["这局我赢了，但你的进攻方向是对的，差在防守。", "我先下一城，关键在中间那步，值得回看。"],
            "ai_lose": ["你赢得漂亮，尤其是那步关键落子，值得记住。", "这局我输了，你的连珠思路很清晰。"],
            "illegal": ["现在轮到对面，规则里这一步还轮不到你。", "这个动作不符合当前阶段的规则，原因我讲给你听。"],
            "idle": ["不急，你可以先看看哪条线最有威胁。", "慢慢想，我在旁边帮你梳理局面。"],
            "game_over": ["对局结束，我们复盘一下关键手和失误点。", "打完了，建议看看中间那几步，很有收获。"],
            "teach_greet": [
                "教学局开始：我能看到你的牌。每步我会讲思路，走完也会点评。",
                "教学局规则：我看得见你的手牌，边打边讲，有问题随时问。",
            ],
            "teach_turn": ["轮到你了。先读一遍手牌，想清楚这步的目标。", "你的回合。看看手里的牌，评估一下再动手。"],
            "teach_move": ["这手走完了，我对照参考讲评一下。", "落子了，我们看看这步和参考思路的差别。"],
        },
    ),
    "banter": Persona(
        key="banter",
        display_name="轻松吐槽",
        verbosity=2,
        tone="吐槽",
        fallback_lines={
            "greet": ["哟，来啦？今天输赢都算你的。"],
            "good_move": ["这步可以啊，有点东西。"],
            "blunder": ["这步棋很有想法，就是有点费棋盘。"],
            "help": ["要我支招？先把这局认真下完。"],
            "ai_win": ["这局我赢了，别灰心，下局你可能就翻盘。"],
            "ai_lose": ["行，你厉害，我甘拜下风。"],
            "illegal": ["哎，这步走不了，棋盘都快被你掀了。"],
            "idle": ["还在想？我都快睡着了，不过不催你。"],
            "game_over": ["收工！要不要复盘，还是直接再来一局？"],
            "teach_greet": ["教学局是吧？行，你的牌我看着呢，翻车了我可要吐槽。"],
            "teach_turn": ["到你了，牌都摆你脸上了，可别打出喜剧效果。"],
            "teach_move": ["这手嘛……我看看你怎么发挥的。"],
        },
    ),
    "cold": Persona(
        key="cold",
        display_name="高冷竞技",
        verbosity=0,
        tone="高冷",
        fallback_lines={
            "greet": ["开始。"],
            "good_move": ["还行。"],
            "blunder": ["有风险。"],
            "help": ["占角。"],
            "ai_win": ["胜。"],
            "ai_lose": ["你赢了。"],
            "illegal": ["不合规。"],
            "idle": ["轮到你了。"],
            "game_over": ["结束。"],
            "teach_greet": ["教学局。你的牌我可见。"],
            "teach_turn": ["你的回合。看牌。"],
            "teach_move": ["讲评。"],
        },
    ),
}


def _assert_coverage() -> None:
    """校验每个 Persona 覆盖全部场景（导入期自检，防御性约束）."""
    for persona in PERSONAS.values():
        missing = [scenario for scenario in SCENARIOS if scenario not in persona.fallback_lines]
        if missing:
            raise ValueError(f"Persona {persona.key!r} 缺少兜底场景: {missing}")


_assert_coverage()
