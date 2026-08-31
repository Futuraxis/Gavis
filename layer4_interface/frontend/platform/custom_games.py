"""Custom-game store and registry — user-created games for the platform.

A custom game is born from a Layer-1 translation (``translate_rules_json``
for from-scratch rules, ``translate_variant_rules`` for template variants),
is validated by the engine (Layer 1 does that inside the translation),
classified into a rule family (``families.detect_family``), turned into a
platform ``GameSpec`` by the family module, and persisted as one JSON file
under ``data/custom_games/<game_id>.json`` (atomic write, strict id
whitelist — mirroring ``platform/history.py``).

Entry points:

- ``CustomGameStore`` — filesystem persistence (save / load / list / delete).
- ``CustomGameRegistry`` — the orchestration: create → detect → build →
  persist; ``spec_for`` re-builds a cached ``GameSpec`` for sessions;
  ``list_games`` feeds the ``/api/games`` merge.

``translate_variant_rules`` is a parallel (A1) delivery, so it is imported
lazily inside the call path only — a missing symbol surfaces as a clear
error instead of breaking this module's import.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from layer1_translator import ValidationResult, translate_rules_json

from ..engine_helpers import RULES_DIR
from .families import detect_family
from .games import GAMES, GameSpec

#: game_id 白名单 — 仅小写字母数字、下划线、连字符（路径安全，对齐 history 审计项）。
_GAME_ID_RE = re.compile(r"^[a-z0-9_-]{1,48}$")

#: 族 → 运行时求解器可选项（平台条目展现用；真实装配由族模块经 provider 完成）。
_FAMILY_SOLVER_OPTIONS = {
    "grid": ("mcts", "random"),
    "poker": ("hybrid", "mcts", "random"),
    "mahjong": ("mahjong", "random"),
    "social": ("ollama", "random"),
}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class CustomGameError(Exception):
    """Custom-game creation failure (invalid rules / unsupported family / …)."""

    def __init__(
        self,
        message: str,
        *,
        validation: ValidationResult | None = None,
        diff_summary: str | None = None,
    ) -> None:
        """Initialize with a Chinese message plus optional error context."""
        super().__init__(message)
        self.validation = validation
        self.diff_summary = diff_summary


def _slug_id(text: str) -> str:
    """Lowercase slug of ``text`` (``[a-z0-9_-]{1,48}``); empty when unusable."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).strip().lower()).strip("-")
    return slug[:48]


def _section_count(rules: dict, section: str) -> int:
    """Element count of ``rules[section]`` (dict/list), 0 when absent."""
    value = rules.get(section)
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    return 0


def _diff_summary(base: dict, modified: dict) -> str:
    """One-sentence top-level-key diff (added/removed/changed + section counts)."""
    base_keys, new_keys = set(base), set(modified)
    added = sorted(new_keys - base_keys)
    removed = sorted(base_keys - new_keys)
    changed = sorted(key for key in base_keys & new_keys if base[key] != modified[key])
    parts: list[str] = []
    if added:
        parts.append(f"新增 {', '.join(added)}")
    if removed:
        parts.append(f"删除 {', '.join(removed)}")
    if changed:
        parts.append(f"修改 {', '.join(changed)}")
    counts = [
        f"{section} {_section_count(base, section)}→{_section_count(modified, section)}"
        for section in ("constants", "actions", "effectors")
        if _section_count(base, section) != _section_count(modified, section)
    ]
    summary = "；".join(parts + counts)
    return f"顶层键级差异：{summary}。" if summary else "与基础模板顶层结构一致。"


class CustomGameStore:
    """Filesystem-backed persistence of custom-game registry entries."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def check_game_id(game_id: str) -> str:
        """Validate ``game_id`` against the whitelist; raise on violation."""
        if not isinstance(game_id, str) or not _GAME_ID_RE.fullmatch(game_id):
            raise CustomGameError(f"非法游戏 id: {game_id!r}（仅允许 [a-z0-9_-]{'{1,48}'}）")
        return game_id

    def _atomic_write(self, path: Path, entry: dict) -> None:
        """Write via a temp file in the same directory, then rename."""
        fd, tmp_name = tempfile.mkstemp(dir=self.data_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def save(self, entry: dict) -> str:
        """Persist one registry entry; returns its ``game_id``."""
        game_id = self.check_game_id(entry.get("game_id", ""))
        self._atomic_write(self.data_dir / f"{game_id}.json", entry)
        return game_id

    def load(self, game_id: str) -> dict:
        """Load one entry; raises ``CustomGameError`` when missing/corrupt."""
        self.check_game_id(game_id)
        path = self.data_dir / f"{game_id}.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
        except OSError:
            raise CustomGameError(f"自定义游戏不存在: {game_id}") from None
        except ValueError:
            raise CustomGameError(f"自定义游戏记录损坏: {game_id}") from None
        if not isinstance(entry, dict) or entry.get("game_id") != game_id:
            raise CustomGameError(f"自定义游戏记录损坏: {game_id}")
        return entry

    def list(self) -> list[dict]:
        """All stored entries (corrupt files skipped), oldest first."""
        entries: list[dict] = []
        for path in self.data_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                if isinstance(entry, dict) and entry.get("game_id"):
                    entries.append(entry)
            except (OSError, ValueError):
                continue
        entries.sort(key=lambda e: (e.get("created_at", ""), e.get("game_id", "")))
        return entries

    def delete(self, game_id: str) -> bool:
        """Remove one entry; ``False`` when it did not exist."""
        self.check_game_id(game_id)
        path = self.data_dir / f"{game_id}.json"
        try:
            path.unlink()
            return True
        except OSError:
            return False


class CustomGameRegistry:
    """Create / detect / build / persist custom games from Layer-1 output."""

    def __init__(self, store: CustomGameStore) -> None:
        """Initialize with the persistence store; specs are cached lazily."""
        self._store = store
        self._spec_cache: dict[str, GameSpec] = {}

    # ── Creation ─────────────────────────────────────────────────

    def create(
        self,
        *,
        mode: str = "from_scratch",
        rule_text: str | None = None,
        base_game_id: str | None = None,
        change_text: str | None = None,
        game_name: str | None = None,
        source_lang: str = "zh",
        use_llm: bool = False,
        strict_llm: bool | None = None,
        llm_client: Any | None = None,
        llm_model: str | None = None,
        llm_model_path: str | None = None,
    ) -> dict:
        """Translate, validate, classify, spec-build and persist a game.

        Args:
            mode: ``"from_scratch"`` (rule text) or ``"variant"`` (template
                base + change text, needs the A1 ``translate_variant_rules``).
            rule_text: 从零描述的游戏规则文本（from_scratch 模式必填）。
            base_game_id: 基础模板游戏 id（variant 模式必填，见
                ``layer1_translator.rule_parser.TEMPLATE_FILES``）。
            change_text: 对基础模板的变更描述（variant 模式必填）。
            game_name: 可选游戏名（用于展示与 id 生成）。
            source_lang: 规则文本语言。
            use_llm: 是否走 LLM 翻译路径。
            strict_llm: LLM 失败是否直接报错（不外兜底）。None 时随
                ``use_llm`` 走：显式要求 LLM 翻译 → 严格，API 错误/传输
                失败如实上报，防止静默产出与描述不符的模板游戏。
            llm_client / llm_model_path: 透传给翻译器的 LLM 参数。

        Returns:
            The persisted registry entry dict.

        Raises:
            CustomGameError: 校验失败 / 族不支持 / 参数缺失 / 变体翻译不可用。
        """
        if strict_llm is None:
            strict_llm = use_llm
        if mode == "from_scratch":
            if not rule_text or not str(rule_text).strip():
                raise CustomGameError("缺少规则文本 (rule_text)")
            response = translate_rules_json(
                str(rule_text),
                source_lang=source_lang,
                game_name=game_name,
                run_engine_validation=True,
                use_llm=use_llm,
                strict_llm=bool(strict_llm),
                llm_client=llm_client,
                llm_model=llm_model,
                llm_model_path=llm_model_path,
            )
            diff_summary: str | None = None
        elif mode == "variant":
            if not base_game_id:
                raise CustomGameError("缺少基础游戏 (base_game_id)")
            if not change_text or not str(change_text).strip():
                raise CustomGameError("缺少变更文本 (change_text)")
            base_rules = self._base_template(str(base_game_id))
            response = self._translate_variant(
                str(base_game_id),
                str(change_text),
                game_name,
                source_lang,
                use_llm,
                bool(strict_llm),
                llm_client,
                llm_model,
                llm_model_path,
            )
            diff_summary = _diff_summary(base_rules, response.rules_json)
        else:
            raise CustomGameError(f"未知模式: {mode}（仅支持 from_scratch / variant）")

        rules = response.rules_json
        validation = response.validation
        if not rules or validation is None or not validation.valid:
            raise CustomGameError(
                "规则校验未通过",
                validation=validation,
                diff_summary=diff_summary,
            )

        family = detect_family(rules)
        if family is None:
            invalid = ValidationResult(
                valid=False,
                errors=["该规则暂不支持平台对弈"],
                warnings=list(validation.warnings),
            )
            raise CustomGameError(
                "该规则暂不支持平台对弈",
                validation=invalid,
                diff_summary=diff_summary,
            )

        game_id = self._next_game_id(rules, game_name)
        spec = family.build_spec(game_id, rules)
        spec = self._with_display_name(spec, rules, game_name, game_id)
        entry = self._entry(game_id, spec, family.FAMILY_ID, rules, response.confidence, validation, diff_summary)
        self._store.save(entry)
        self._spec_cache[game_id] = spec
        return entry

    # ── Lookup ───────────────────────────────────────────────────

    def spec_for(self, game_id: str) -> GameSpec | None:
        """Rebuild (and cache) the ``GameSpec`` for a stored custom game."""
        cached = self._spec_cache.get(game_id)
        if cached is not None:
            return cached
        try:
            entry = self._store.load(game_id)
        except CustomGameError:
            return None
        spec = self._spec_from_entry(entry)
        self._spec_cache[game_id] = spec
        return spec

    def family_of(self, game_id: str) -> str | None:
        """Family id of a stored custom game (``None`` when unknown)."""
        try:
            entry = self._store.load(game_id)
        except CustomGameError:
            return None
        family = entry.get("family")
        return family if isinstance(family, str) else None

    def has(self, game_id: str) -> bool:
        """Whether ``game_id`` is a stored custom game."""
        try:
            self._store.load(game_id)
            return True
        except CustomGameError:
            return False

    def delete(self, game_id: str) -> bool:
        """Delete a stored custom game; ``False`` when it did not exist."""
        self._spec_cache.pop(game_id, None)
        return self._store.delete(game_id)

    def list_games(self) -> list[dict]:
        """Stored entries for the ``/api/games`` merge (each is a listing dict)."""
        return self._store.list()

    # ── Internals ─────────────────────────────────────────────────

    def _translate_variant(
        self,
        base_game_id: str,
        change_text: str,
        game_name: str | None,
        source_lang: str,
        use_llm: bool,
        strict_llm: bool,
        llm_client: Any | None,
        llm_model: str | None,
        llm_model_path: str | None,
    ) -> Any:
        """Invoke ``translate_variant_rules`` (lazy import — A1 parallel delivery)."""
        try:
            from layer1_translator import translate_variant_rules
        except ImportError as exc:
            raise CustomGameError(f"变体翻译不可用: {exc}") from exc
        return translate_variant_rules(
            base_game_id,
            change_text,
            source_lang=source_lang,
            game_name=game_name,
            use_llm=use_llm,
            strict_llm=strict_llm,
            llm_client=llm_client,
            llm_model=llm_model,
            llm_model_path=llm_model_path,
            run_engine_validation=True,
        )

    @staticmethod
    def _base_template(base_game_id: str) -> dict:
        """Load the base template rules JSON for diffing (``TEMPLATE_FILES``)."""
        try:
            from layer1_translator.rule_parser import TEMPLATE_FILES
        except ImportError as exc:
            raise CustomGameError(f"基础模板定位失败: {exc}") from exc
        file_name = TEMPLATE_FILES.get(base_game_id)  # type: ignore[attr-defined]
        if not file_name:
            raise CustomGameError(f"未知基础游戏: {base_game_id}")
        path = RULES_DIR / file_name
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except OSError:
            raise CustomGameError(f"基础模板文件缺失: {file_name}") from None

    def _next_game_id(self, rules: dict, game_name: str | None) -> str:
        """Slug from ``game_name`` first, then ``meta.gameId``; unique vs store + GAMES.

        用户显式填的名字优先作为 id 来源——否则模板/LLM 的
        ``meta.gameId``（常是内置 slug 如 ``stochastic_gomoku``）会让
        自定义游戏与内置撞名。仅在用户没填名字时回退到 ``meta.gameId``；
        中文名 slug 为空时再次回退 ``meta.gameId`` / ``custom_game``。
        """
        meta = rules.get("meta", {})
        meta_id = meta.get("gameId") if isinstance(meta, dict) else None
        raw = game_name or meta_id or "custom_game"
        # 中文名 slug 为空时回退到 meta_id 的 slug（而非原始 meta_id）——
        # 保持 id 一律 slug 化（连字符），避免 stochastic_gomoku-2 这种
        # 下划线+后缀的混搭，也避免与内置 underscore id 形态撞形。
        base = _slug_id(raw) or _slug_id(str(meta_id or "")) or "custom_game"
        taken = {entry.get("game_id") for entry in self._store.list()} | set(GAMES)
        candidate = base
        n = 2
        while candidate in taken:
            candidate = f"{base}-{n}"
            n += 1
        return candidate

    @staticmethod
    def _resolve_display_name(
        rules: dict, game_name: str | None, game_id: str
    ) -> str:
        """人类可读展示名优先级：用户名 > meta.gameName > meta.gameId > game_id。

        修复前各族 ``build_spec`` 把 ``display_name`` 直接设成
        ``meta.gameId``（一个 slug），用户填的名字被丢弃——横幅出现
        《stochastic_gomoku》而非用户起的名。此函数集中修正该优先级。
        """
        meta = rules.get("meta", {})
        meta_name = meta.get("gameName") if isinstance(meta, dict) else None
        meta_id = meta.get("gameId") if isinstance(meta, dict) else None
        return str(game_name or meta_name or meta_id or game_id)

    def _with_display_name(
        self, spec: GameSpec, rules: dict, game_name: str | None, game_id: str
    ) -> GameSpec:
        """Return ``spec`` with ``display_name`` set per :meth:`_resolve_display_name`.

        ``GameSpec`` is frozen; a non-matching resolved name rebuilds the spec
        via :func:`dataclasses.replace` so the human-readable name flows into
        the persisted entry and rebuilt specs (``spec_for`` / ``_spec_from_entry``).
        """
        display_name = self._resolve_display_name(rules, game_name, game_id)
        if display_name != spec.display_name:
            spec = replace(spec, display_name=display_name)
        return spec

    def _spec_from_entry(self, entry: dict) -> GameSpec:
        """Rebuild the spec from a stored entry (detect family + build).

        与 create 路径共用 ``_with_display_name``：重建 spec 时以持久化的
        ``display_name`` 为准——用户起的名字不能因为只存了 rules 而回退成
        ``meta.gameId`` 的 slug。规则族无法识别时明确拒绝（防御性红线，
        ``spec_for`` 对无族条目必须报错而不是静默生成一个不可玩的 spec）。
        """
        rules = entry.get("rules", {})
        family = detect_family(rules)
        if family is None:
            raise CustomGameError(f"自定义游戏 {entry['game_id']} 无法识别规则族")
        game_id = str(entry["game_id"])
        spec = family.build_spec(game_id, rules)
        stored_name = entry.get("display_name")
        if isinstance(stored_name, str) and stored_name:
            spec = self._with_display_name(spec, rules, stored_name, game_id)
        return spec

    def _entry(
        self,
        game_id: str,
        spec: GameSpec,
        family_id: str,
        rules: dict,
        confidence: float,
        validation: ValidationResult,
        diff_summary: str | None,
    ) -> dict:
        """Assemble the persisted listing entry for a custom game."""
        return {
            "game_id": game_id,
            "display_name": spec.display_name,
            "description": spec.description,
            "kind": spec.kind,
            "family": family_id,
            "board_size": spec.board_size,
            "seat_options": list(spec.seat_options),
            "seat_label": spec.seat_label,
            "player_counts": list(spec.player_counts),
            "difficulties": list(spec.difficulty_budgets),
            "solver_options": list(_FAMILY_SOLVER_OPTIONS.get(family_id, ("random",))),
            "custom": True,
            "confidence": confidence,
            "validation": {
                "valid": validation.valid,
                "errors": list(validation.errors),
                "warnings": list(validation.warnings),
            },
            "diff_summary": diff_summary,
            "rules": rules,
            "created_at": _now_iso(),
        }


__all__ = ["CustomGameError", "CustomGameRegistry", "CustomGameStore"]
