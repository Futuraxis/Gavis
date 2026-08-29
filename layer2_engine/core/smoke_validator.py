"""Engine smoke-validation service for rule producers.

Owned by Layer 2: the question "can this rules JSON boot a ``GameEngine``
and expose basic dynamics" belongs to the engine, not to the translator.
``layer1_translator.engine_validator`` consumes this service through its
single authorized L1→L2 channel (review round 2026-08).

Variant-aware smoke (v5.5): when the rules JSON declares a ``variants``
section, ``smoke_validate(..., variants="all")`` boots **every declared
option** (plus the default selection) — a variant whose ``constants``
patch breaks construction or the initial transition is caught at
translation/creation time instead of silently failing at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .engine import GameEngine

#: Sentinel meaning "base engine + every declared ``variants.options``".
_ALL_VARIANTS = "all"


@dataclass
class SmokeValidation:
    """Errors and warnings from one smoke-validation run."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def smoke_validate(
    rules: dict[str, Any],
    seed: int = 42,
    variants: str | Sequence[str] | None = None,
) -> SmokeValidation:
    """Boot the engine and probe one initial transition per selection.

    Construction and state progression are captured separately so a
    real engine defect surfaces as a distinct, typed error instead of
    being collapsed into a generic "validation failed".

    ``variants`` selects which declared ``variants.options`` to probe:

    - ``None`` (default) — the base engine only (the default variant
      selection; backward compatible with the pre-v5.5 behaviour).
    - ``"all"`` — base plus every option declared in ``variants.options``
      (covers the default selection as a plain boot, then each named
      variant).  Any option whose ``constants`` patch breaks engine
      construction or the initial transition is reported with a
      ``[variant=...]`` label.
    - an explicit sequence of option names — just those (each probed
      with its constants patch applied).

    A rules dict without a ``variants`` section probes the base engine
    exactly once regardless of ``variants``.
    """
    result = SmokeValidation()
    selections = _selections(rules, variants)
    for label, variant, player_count in selections:
        try:
            engine = GameEngine(rules, seed=seed, variant=variant, player_count=player_count)
        except Exception as exc:
            result.errors.append(f"[{label}] GameEngine 构造失败: {type(exc).__name__}: {exc}")
            continue
        _probe_initial_transition(engine, result, label)
    return result


def _selections(
    rules: dict[str, Any], variants: str | Sequence[str] | None
) -> list[tuple[str, str | None, int | None]]:
    """Resolve ``(label, variant_arg, player_count_arg)`` probe selections."""
    if not isinstance(rules.get("variants"), dict):
        return [("base", None, None)]
    spec = rules["variants"]
    options = spec.get("options", {}) or {}
    if not isinstance(options, dict) or not options:
        return [("base", None, None)]
    if variants is None:
        return [("base", None, None)]
    if variants == _ALL_VARIANTS:
        picks: list[str] = []
        default = spec.get("variant")
        if isinstance(default, str) and default in options:
            picks.append(default)
        picks.extend(name for name in options if name != default)
        names = picks
    elif isinstance(variants, str):
        names = [variants]
    else:
        names = list(variants)
    count = spec.get("player_count")
    player_count = count if isinstance(count, int) else None
    out: list[tuple[str, str | None, int | None]] = [("base", None, None)]
    for name in names:
        if not isinstance(name, str) or name not in options:
            # 未声明的变体在引擎里会抛 ValueError —— 这里直接标为构造失败，
            # 与 GameEngine 的语义一致（schema 校验也应先拦住）。
            out.append((f"variant={name}", name, player_count))
            continue
        out.append((f"variant={name}", name, player_count))
    return out


def _probe_initial_transition(engine: GameEngine, result: SmokeValidation, label: str) -> None:
    """One initial transition: resolve chance nodes or take the first legal move."""
    try:
        state = engine.create_initial_state()
        node_type = engine.get_node_type(state)
        if node_type == "player":
            actions = engine.get_legal_actions(state)
            if not actions:
                result.warnings.append(f"[{label}] 初始 player 节点没有合法动作")
            else:
                engine.apply_action(state, actions[0])
        elif node_type == "chance":
            outcomes = engine.get_chance_outcomes(state)
            if not outcomes:
                result.errors.append(f"[{label}] 初始 chance 节点没有 chance outcomes")
            else:
                engine.apply_chance(state, outcomes[0])
        elif node_type != "terminal":
            result.errors.append(f"[{label}] 未知节点类型: {node_type}")
    except Exception as exc:
        result.errors.append(f"[{label}] Engine smoke validation failed: {type(exc).__name__}: {exc}")


__all__ = ["SmokeValidation", "smoke_validate"]
