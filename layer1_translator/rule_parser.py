"""Deterministic parser for known Layer 1 game templates.

The parser intentionally extracts only parameters that existing rules
templates can consume safely. Unknown details remain in natural language
for future translators instead of being guessed into executable rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

TEMPLATE_FILES = {
    "moon_chess": "moon_chess.json",
    "stochastic_gomoku": "stochastic_gomoku.json",
    "gomoku": "stochastic_gomoku.json",
    "texas_holdem": "texas_holdem.json",
    "mahjong": "mahjong.json",
    "werewolf": "werewolf.json",
    "uno": "uno.json",
}

ALIASES = {
    "月亮棋": "moon_chess",
    "moon chess": "moon_chess",
    "moon_chess": "moon_chess",
    "moon": "moon_chess",
    "随机五子棋": "stochastic_gomoku",
    "概率五子棋": "stochastic_gomoku",
    "消失五子棋": "stochastic_gomoku",
    "stochastic gomoku": "stochastic_gomoku",
    "stochastic_gomoku": "stochastic_gomoku",
    "五子棋": "stochastic_gomoku",
    "gomoku": "stochastic_gomoku",
    "德州扑克": "texas_holdem",
    "德州": "texas_holdem",
    "texas hold'em": "texas_holdem",
    "texas holdem": "texas_holdem",
    "texas_holdem": "texas_holdem",
    "texas": "texas_holdem",
    "poker": "texas_holdem",
    "麻将": "mahjong",
    "mahjong": "mahjong",
    "狼人杀": "werewolf",
    "狼人": "werewolf",
    "werewolf": "werewolf",
    "uno": "uno",
    "UNO": "uno",
    "尤诺": "uno",
    "乌诺": "uno",
    "优诺": "uno",
}

CHINESE_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


@dataclass(frozen=True)
class ParsedRuleRequest:
    """Known template id plus conservative parameter overrides."""

    game_id: str
    parameters: dict[str, Any] = field(default_factory=dict)


class RuleParser:
    """Parse game identity and supported template parameters from text."""

    def parse(self, *, rule_text: str, game_name: str | None = None) -> ParsedRuleRequest | None:
        """Return a parsed template request, or ``None`` if the game is unknown."""
        game_id = self.resolve_game_id(rule_text=rule_text, game_name=game_name)
        if game_id is None:
            return None
        return ParsedRuleRequest(game_id=game_id, parameters=self.parse_parameters(game_id, rule_text))

    def resolve_game_id(self, *, rule_text: str, game_name: str | None = None) -> str | None:
        """Resolve a supported game id from explicit name or natural-language hint."""
        candidates = [game_name or "", rule_text]
        for candidate in candidates:
            normalized = self._normalize(candidate)
            if normalized in TEMPLATE_FILES:
                return "stochastic_gomoku" if normalized == "gomoku" else normalized
            for alias, game_id in ALIASES.items():
                if self._normalize(alias) in normalized:
                    return game_id
        return None

    def parse_parameters(self, game_id: str, text: str) -> dict[str, Any]:
        """Extract supported parameter overrides for a known game template."""
        if game_id in ("stochastic_gomoku", "gomoku"):
            return self._parse_grid_game(text, include_vanish=True)
        if game_id == "moon_chess":
            return self._parse_grid_game(text, include_max_pieces=True)
        if game_id == "werewolf":
            return self._parse_werewolf(text)
        if game_id == "mahjong":
            return self._parse_mahjong(text)
        if game_id == "texas_holdem":
            return self._parse_texas_holdem(text)
        if game_id == "uno":
            return self._parse_uno(text)
        return {}

    def parse_grid_family_parameters(self, text: str) -> dict[str, Any]:
        """Extract generic square-board alignment parameters.

        This is intentionally broader than template parsing: it is used by
        rule-family generation after known game names fail to match.
        """
        return self._parse_grid_game(text, include_vanish=True)

    def _parse_grid_game(
        self,
        text: str,
        *,
        include_vanish: bool = False,
        include_max_pieces: bool = False,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        board_size = self._extract_board_size(text)
        win_length = self._extract_win_length(text)
        if board_size is not None:
            params["board_size"] = board_size
        if win_length is not None:
            params["win_length"] = win_length
        if include_vanish:
            vanish_probability = self._extract_probability(text)
            if vanish_probability is not None:
                params["vanish_probability"] = vanish_probability
        if include_max_pieces:
            max_pieces = self._extract_count_before(
                text,
                ("枚", "颗", "个", "子"),
                ("棋子", "落子", "子"),
            )
            if max_pieces is not None:
                params["max_pieces"] = max_pieces
        return params

    def _parse_werewolf(self, text: str) -> dict[str, Any]:
        params: dict[str, Any] = {}
        players = self._extract_count_before(text, ("人", "名"), ("玩家", "局", "狼人杀"))
        wolves = self._extract_count_before(text, ("狼", "狼人"), ("狼", "狼人"))
        seers = self._extract_count_before(text, ("预言家", "预"), ("预言家", "预"))
        villagers = self._extract_count_before(text, ("村民", "民"), ("村民", "民"))
        with_witch = self._flag_from_text(
            text,
            positive=("女巫", "witch"),
            negative=("无女巫", "没有女巫", "禁用女巫", "关闭女巫", "关掉女巫", "no witch"),
        )
        with_hunter = self._flag_from_text(
            text,
            positive=("猎人", "hunter"),
            negative=("无猎人", "没有猎人", "禁用猎人", "关闭猎人", "关掉猎人", "no hunter"),
        )
        with_guard = self._flag_from_text(
            text,
            positive=("守卫", "guard"),
            negative=("无守卫", "没有守卫", "禁用守卫", "关闭守卫", "关掉守卫", "no guard"),
        )

        if players is not None:
            params["players"] = players
        if wolves is not None:
            params["wolves"] = wolves
        if seers is not None:
            params["seers"] = seers
        if villagers is not None:
            params["villagers"] = villagers
        if with_witch is not None:
            params["with_witch"] = with_witch
        if with_hunter is not None:
            params["with_hunter"] = with_hunter
        if with_guard is not None:
            params["with_guard"] = with_guard
        return params

    def _parse_mahjong(self, text: str) -> dict[str, Any]:
        """Parse mahjong template parameters.

        变体关键词映射到 ``rules/mahjong.json`` 声明的六变体；先查更
        特异的中文词（血流成河/血战到底），再查英文/通用词。T2 修复：
        旧版只认 红中/血战/广东 —— "四川/长沙/台湾麻将" 解析不出变体，
        且"血战"（血战到底 = sichuan）被错映射到 blood（血流成河）。
        """
        params: dict[str, Any] = {}
        normalized = self._normalize(text)
        # 特异优先：血流成河 → blood；血战到底/四川 → sichuan。
        if "血流" in text or "blood" in normalized:
            params["variant"] = "blood"
        elif "血战" in text or "四川" in text or "sichuan" in normalized:
            params["variant"] = "sichuan"
        elif "红中" in text or "hongzhong" in normalized:
            params["variant"] = "hongzhong"
        elif "长沙" in text or "changsha" in normalized:
            params["variant"] = "changsha"
        elif "台湾" in text or "taiwan" in normalized:
            params["variant"] = "taiwan"
        elif "广东" in text or "鸡胡" in text or "guangdong" in normalized:
            params["variant"] = "guangdong"

        # 人数原样透传（int 时）：2/4 之外的值由 _apply_mahjong_params
        # 校验并给出"保留默认"警告 —— 在 parser 层过滤会让该警告成为
        # 死代码（"红中麻将 3人" 静默跑 4 人局，用户无从得知）。
        player_count = self._extract_count_before(text, ("人", "家"), ("麻将", "局", "玩家"))
        if player_count is not None:
            params["player_count"] = player_count
        return params

    def _parse_uno(self, text: str) -> dict[str, Any]:
        """Parse UNO template parameters: player count (2..10) and variant.

        Variant keywords map to the declared ``variants`` of
        ``rules/uno.json`` (classic / seven_zero / jump_in / stacking /
        draw_until / strict_wild4); only the first match wins.
        """
        params: dict[str, Any] = {}
        player_count = self._extract_count_before(
            text,
            ("人", "名", "位", "玩家", "个"),
            ("局", "游戏", "uno", "UNO"),
        )
        if player_count is not None and 2 <= player_count <= 10:
            params["player_count"] = player_count
        variant_map = {
            "7-0": "seven_zero",
            "7 0": "seven_zero",
            "7和0": "seven_zero",
            "7跟0": "seven_zero",
            "换手": "seven_zero",
            "移交": "seven_zero",
            "seven_zero": "seven_zero",
            "抢牌": "jump_in",
            "抢出": "jump_in",
            "抢": "jump_in",
            "jump_in": "jump_in",
            "叠加": "stacking",
            "叠牌": "stacking",
            "能叠": "stacking",
            "stacking": "stacking",
            "摸到能打": "draw_until",
            "摸到可打": "draw_until",
            "摸到能出": "draw_until",
            "draw_until": "draw_until",
            "严格加四": "strict_wild4",
            "严格+4": "strict_wild4",
            "严格4": "strict_wild4",
            "strict_wild4": "strict_wild4",
            "经典": "classic",
            "classic": "classic",
        }
        normalized = self._normalize(text)
        for key, variant in variant_map.items():
            if key in text or key in normalized:
                params["variant"] = variant
                break
        return params

    def _parse_texas_holdem(self, text: str) -> dict[str, Any]:
        params: dict[str, Any] = {}
        stack_size = self._extract_named_number(text, ("筹码", "stack", "stacks"))
        small_blind, big_blind = self._extract_blinds(text)
        if stack_size is not None:
            params["stack_size"] = stack_size
        if small_blind is not None:
            params["small_blind"] = small_blind
        if big_blind is not None:
            params["big_blind"] = big_blind
        return params

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _extract_board_size(text: str) -> int | None:
        match = re.search(r"(\d+)\s*[x×X]\s*(\d+)", text)
        if match and match.group(1) == match.group(2):
            return int(match.group(1))
        match = re.search(r"(\d+)\s*(?:路|格|行|列|阶|阶棋盘|棋盘)", text)
        return int(match.group(1)) if match else None

    @classmethod
    def _extract_win_length(cls, text: str) -> int | None:
        match = re.search(r"(\d+)\s*(?:子|连).{0,8}(?:获胜|胜利|赢|成线|连珠)", text)
        if match:
            return int(match.group(1))
        match = re.search(
            r"([一二两三四五六七八九十])\s*(?:子|连).{0,8}(?:获胜|胜利|赢|成线|连珠)",
            text,
        )
        if match:
            return cls._to_int(match.group(1))
        if "五子" in text:
            return 5
        if "三子" in text or "三连" in text:
            return 3
        return None

    @staticmethod
    def _extract_probability(text: str) -> float | None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:概率)?\s*(?:消失|消除|vanish)", text)
        if match:
            return float(match.group(1)) / 100.0
        match = re.search(r"(?:消失|消除|vanish)\s*(?:概率|chance)?\s*(\d+(?:\.\d+)?)\s*%", text)
        if match:
            return float(match.group(1)) / 100.0
        match = re.search(r"(?:消失|消除|vanish)\s*(?:概率|chance)?\s*(0(?:\.\d+)?|1(?:\.0+)?)", text)
        if match:
            return float(match.group(1))
        if "不消失" in text or "无随机" in text or "no vanish" in text.lower():
            return 0.0
        return None

    @classmethod
    def _extract_count_before(
        cls,
        text: str,
        units: tuple[str, ...],
        labels: tuple[str, ...],
    ) -> int | None:
        unit_pattern = "|".join(re.escape(unit) for unit in units)
        label_pattern = "|".join(re.escape(label) for label in labels)
        match = re.search(rf"(\d+)\s*(?:{unit_pattern})\s*(?:{label_pattern})?", text)
        if match:
            return int(match.group(1))
        match = re.search(rf"([一二两三四五六七八九十])\s*(?:{unit_pattern})\s*(?:{label_pattern})?", text)
        if match:
            return cls._to_int(match.group(1))
        return None

    @staticmethod
    def _extract_named_number(text: str, labels: tuple[str, ...]) -> int | None:
        label_pattern = "|".join(re.escape(label) for label in labels)
        match = re.search(rf"(?:{label_pattern})\s*(?:大小|size)?\s*[:：]?\s*(\d+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(rf"(\d+)\s*(?:{label_pattern})", text, re.IGNORECASE)
        return int(match.group(1)) if match else None

    @staticmethod
    def _extract_blinds(text: str) -> tuple[int | None, int | None]:
        match = re.search(r"(?:盲注|blind|blinds)\s*(\d+)\s*/\s*(\d+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))
        small = RuleParser._extract_named_number(text, ("小盲", "small blind", "small_blind"))
        big = RuleParser._extract_named_number(text, ("大盲", "big blind", "big_blind"))
        return small, big

    @staticmethod
    def _flag_from_text(
        text: str,
        *,
        positive: tuple[str, ...],
        negative: tuple[str, ...],
    ) -> bool | None:
        normalized = text.lower()
        if any(token.lower() in normalized for token in negative):
            return False
        if any(token.lower() in normalized for token in positive):
            return True
        return None

    @staticmethod
    def _to_int(value: str) -> int:
        return CHINESE_DIGITS[value]
