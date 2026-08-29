"""完整 LLM 增量补丁协议 — 变体翻译的补丁载体（RFC-6902 风格子集）。

完整模板重写（variant_translator 的旧 LLM 路径）要求模型原样复述整个
rules JSON：巨型模板（mahjong ≈87k 字符）在 8k token 回复上限下必然截断
（``_MAX_LLM_TEMPLATE_CHARS`` 护栏直接跳过 LLM）。增量补丁协议改让模型
只输出**改动操作**：``{"patch": [{"op": "replace|add|remove", "path":
"constants.board_size", "value": 9}, ...]}``，由本模块应用到基础模板，
其余规则面保持不动 —— 输出极小、结构零复述风险，大模板因此也能走 LLM。

路径语法：点号分隔键名，数字段表示数组下标（``actions.0.legal``）。
操作语义：
- ``replace``：目标必须已存在 → 覆盖（类型随 value）。
- ``add``：目标不存在 → 创建；已存在 → 覆盖（宽松，容忍 LLM 用错）。
- ``remove``：目标必须已存在 → 删除（dict 删键 / list 删下标元素）。

任何违例（未知 op、路径解析失败、越界下标、缺 value、操作数超上限）抛
``PatchError``，调用方回退确定性路径——绝不应用半份补丁。
"""

from __future__ import annotations

import copy
from typing import Any, Sequence

#: 单次回复的操作数上限（防跑飞；超限视为无效产物）。
MAX_PATCH_OPS = 64

_OPS = ("replace", "add", "remove")


class PatchError(Exception):
    """Invalid patch payload (schema, path resolution, or application)."""


def parse_patch(payload: Any) -> list[dict[str, Any]]:
    """Normalize an LLM reply into validated patch ops.

    Accepts the ``{"patch": [...]}`` envelope (or the bare list).  Raises
    ``PatchError`` on a missing/non-list ``patch``, unknown ops, empty or
    non-string paths, and exceeding ``MAX_PATCH_OPS``.
    """
    if not isinstance(payload, dict) or "patch" not in payload:
        raise PatchError("补丁回复缺少 'patch' 字段")
    ops = payload["patch"]
    if not isinstance(ops, list):
        raise PatchError("'patch' 必须是数组")
    if len(ops) > MAX_PATCH_OPS:
        raise PatchError(f"补丁操作数 {len(ops)} 超过上限 {MAX_PATCH_OPS}")
    out: list[dict[str, Any]] = []
    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            raise PatchError(f"patch[{i}] 必须是对象")
        kind = op.get("op")
        if kind not in _OPS:
            raise PatchError(f"patch[{i}] op 未知: {kind!r}（可选 {_OPS}）")
        path = op.get("path")
        if not isinstance(path, str) or not path.strip():
            raise PatchError(f"patch[{i}] 缺少非空字符串 path")
        if kind in ("replace", "add") and "value" not in op:
            raise PatchError(f"patch[{i}] {kind} 缺少 value")
        out.append({"op": kind, "path": path.strip(), "value": op.get("value")})
    return out


def apply_patch(base: dict[str, Any], ops: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return a deep copy of ``base`` with ``ops`` applied.

    Ops are applied in order (so a later op can address a key created by
    an earlier one).  Any failure raises ``PatchError`` and leaves the
    input untouched (the caller never receives a partial product).
    """
    rules = copy.deepcopy(base)
    for i, op in enumerate(ops):
        kind = op["op"]
        segments = _split_path(op["path"])
        parent, key = _resolve_parent(rules, segments, op_index=i)
        try:
            if kind == "replace":
                if not _has_own(parent, key):
                    raise PatchError(f"replace 目标不存在: {op['path']}")
                _set(parent, key, op.get("value"))
            elif kind == "add":
                _set(parent, key, op.get("value"))
            else:  # remove
                if not _has_own(parent, key):
                    raise PatchError(f"remove 目标不存在: {op['path']}")
                _delete(parent, key)
        except PatchError as exc:
            raise PatchError(f"patch[{i}] {kind} {op['path']} 失败: {exc}") from exc
    return rules


def _split_path(path: str) -> list[str]:
    return [segment for segment in path.split(".") if segment != ""]


def _resolve_parent(root: dict[str, Any], segments: list[str], *, op_index: int) -> tuple[Any, str | int]:
    """Walk ``segments[:-1]`` from ``root``; return (container, final key)."""
    if not segments:
        raise PatchError("空 path 不允许（顶层替换会破坏 rules 结构）")
    node: Any = root
    for depth, segment in enumerate(segments[:-1]):
        if isinstance(node, dict):
            if segment not in node:
                raise PatchError(f"path 段 {segment!r} 不存在")
            node = node[segment]
        elif isinstance(node, list):
            if not segment.isdigit():
                raise PatchError(f"list 下标段 {segment!r} 必须是数字")
            index = int(segment)
            if not 0 <= index < len(node):
                raise PatchError(f"list 下标 {segment!r} 越界（len={len(node)}）")
            node = node[index]
        else:
            raise PatchError(f"path 段 {segment!r} 落在一个 {type(node).__name__} 上")
        _ = depth
    final = segments[-1]
    if isinstance(node, dict):
        return node, final
    if isinstance(node, list):
        if not final.isdigit():
            raise PatchError(f"list 下标段 {final!r} 必须是数字")
        index = int(final)
        if not (0 <= index < len(node)):
            raise PatchError(f"list 下标 {final!r} 越界（len={len(node)}）")
        return node, index
    raise PatchError(f"path 最后一段 {final!r} 落在一个 {type(node).__name__} 上")


def _has_own(parent: Any, key: str | int) -> bool:
    if isinstance(parent, dict):
        return key in parent
    if isinstance(parent, list) and isinstance(key, int):
        return 0 <= key < len(parent)
    return False


def _set(parent: Any, key: str | int, value: Any) -> None:
    if isinstance(parent, dict):
        parent[key] = value
        return
    if isinstance(parent, list) and isinstance(key, int) and 0 <= key < len(parent):
        parent[key] = value
        return
    raise PatchError("无法写入目标容器")


def _delete(parent: Any, key: str | int) -> None:
    if isinstance(parent, dict):
        del parent[key]
        return
    if isinstance(parent, list) and isinstance(key, int) and 0 <= key < len(parent):
        del parent[key]
        return
    raise PatchError("无法删除目标")


__all__ = ["MAX_PATCH_OPS", "PatchError", "apply_patch", "parse_patch"]
