"""Match history storage for the platform frontend.

Each finished match is persisted as one JSON file under
``data/matches/<match_id>.json``.  Writes are atomic (temp file +
``os.replace``) so a crash never leaves a partial record on disk.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

#: match_id 白名单：仅字母数字、下划线、连字符（审计 3.6 路径遍历修复——
#: 不含路径分隔符/`..`，杜绝 `../../` 逃逸出 data 目录）。
_MATCH_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class HistoryError(Exception):
    """Missing, unreadable, or corrupt match record."""


class MatchHistory:
    """Filesystem-backed store of finished match records."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ── Writing ──────────────────────────────────────────────────

    def record(self, match: dict[str, Any]) -> str:
        """Persist a finished match record; returns its match_id."""
        match_id = match.get("match_id")
        if not match_id:
            raise HistoryError("match record is missing match_id")
        _check_match_id(match_id)
        match = dict(match)
        # 末手快照：人数 / 变体等元数据的唯一来源（旧记录可能没有 moves）。
        moves = match.get("moves") if isinstance(match.get("moves"), list) else []
        last_snap = moves[-1].get("snapshot") if moves and isinstance(moves[-1], dict) else None
        if not isinstance(last_snap, dict):
            last_snap = None
        match["meta"] = {
            "match_id": match.get("match_id"),
            "game_id": match.get("game_id"),
            "player_pid": match.get("player_pid"),
            "ai_pid": match.get("ai_pid"),
            "difficulty": match.get("difficulty"),
            "winner": match.get("winner"),
            "over": match.get("over"),
            "moves": len(match.get("moves", [])),
            # 玩家视角胜负（layer4_interface/result 解析；阵营胜者正确归边——
            # 旧记录缺省 None，前端回退 pid 比较）。供战绩/历史/聊天统计复用，
            # 避免社交阵营胜者在列表页被误标胜负。
            "won": match.get("won"),
            "started_at": match.get("started_at"),
            "finished_at": match.get("finished_at"),
            # 陪伴感扩展（PRD 4.1.5 / 4.4.4）：性格、是否用过提示、本局 AI 强度档
            # （旧记录缺省 None，前端按可选字段处理）。
            "persona": match.get("persona"),
            "hinted": match.get("hinted"),
            "ai_strength": match.get("ai_strength"),
            # 教学对局标记（旧记录缺省 None = 非教学局）。
            "teaching": match.get("teaching"),
            # 自适应难度标记（旧记录缺省 None；前端按可选字段展示）。
            "adaptive": match.get("adaptive"),
            # 元数据补全（对局记录页「详细元数据」视图用；旧记录缺省 None）：
            # seed / family / custom 本身已落盘在记录顶层，此前没有进 meta，
            # 列表页因此看不到；player_count / variant 从快照与 game_id 推导。
            "seed": match.get("seed"),
            "family": match.get("family") or (last_snap.get("family") if last_snap else None),
            "custom": match.get("custom"),
            "player_count": _snapshot_player_count(last_snap),
            "variant": _variant_for(str(match.get("game_id") or ""), last_snap),
        }
        path = self.data_dir / f"{match_id}.json"
        self._atomic_write(path, match)
        return match_id

    def _atomic_write(self, path: Path, match: dict[str, Any]) -> None:
        """Write via a temp file in the same directory, then rename."""
        fd, tmp_name = tempfile.mkstemp(dir=self.data_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(match, f, ensure_ascii=False, indent=2)
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    # ── Reading ──────────────────────────────────────────────────

    def list_matches(self, limit: int | None = 100, game_id: str | None = None) -> list[dict]:
        """Metadata of finished matches, newest first.

        ``limit=None`` returns every match (the platform history page filters
        server-side and therefore needs the full set before slicing; the
        default keeps the historical bounded behaviour for existing callers).

        Corrupt or unreadable files are skipped silently.
        """
        matches: list[dict] = []
        for path in self.data_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    record = json.load(f)
                meta = record.get("meta") if isinstance(record, dict) else None
                if not isinstance(meta, dict) or not meta.get("match_id"):
                    continue
            except (OSError, ValueError):
                continue
            if game_id and meta.get("game_id") != game_id:
                continue
            matches.append(meta)
        matches.sort(key=lambda m: (m.get("started_at") or "", m.get("match_id") or ""), reverse=True)
        return matches if limit is None else matches[:limit]

    def get(self, match_id: str) -> dict:
        """Full match record including the move log."""
        _check_match_id(match_id)
        path = self.data_dir / f"{match_id}.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except OSError:
            raise HistoryError(f"match not found: {match_id}") from None
        except ValueError:
            raise HistoryError(f"match record is corrupt: {match_id}") from None
        if not isinstance(record, dict) or record.get("match_id") != match_id:
            raise HistoryError(f"match record is corrupt: {match_id}")
        return record

    def delete(self, match_id: str) -> None:
        """Remove a stored match record."""
        _check_match_id(match_id)
        path = self.data_dir / f"{match_id}.json"
        try:
            path.unlink()
        except FileNotFoundError:
            raise HistoryError(f"match not found: {match_id}") from None


def _check_match_id(match_id: str) -> None:
    """Validate match_id against the path-traversal-safe whitelist."""
    if not isinstance(match_id, str) or not _MATCH_ID_RE.fullmatch(match_id):
        raise HistoryError(f"invalid match_id: {match_id!r}")


def _snapshot_player_count(snapshot: dict[str, Any] | None) -> int | None:
    """Seat count from a match snapshot (``None`` when the snapshot omits it).

    平台快照不统一暴露 ``pids``：牌类游戏用 ``hand_counts``（每座张数，键即座位），
    棋类没有座位表。取不到就返回 ``None``（记录页按「缺就不显示」渲染），
    绝不猜 2 —— 猜错会让四麻对局显示成两人局。
    """
    if not isinstance(snapshot, dict):
        return None
    for key in ("player_count",):
        value = snapshot.get(key)
        if isinstance(value, int) and value > 0:
            return value
    for key in ("hand_counts", "pids", "players", "seats"):
        value = snapshot.get(key)
        if isinstance(value, dict) and value:
            return len(value)
        if isinstance(value, list) and value:
            return len(value)
    return None


#: game_id 里带变体后缀的游戏 → 变体名（麻将/UNO 的变体在平台注册表里就是独立
#: game_id，快照本身不带 variant 字段，所以从 id 派生；见 games.py 的 GAMES）。
_VARIANT_SUFFIXES: dict[str, str] = {
    "mahjong_guangdong": "guangdong",
    "mahjong_hongzhong": "hongzhong",
    "mahjong_blood": "blood",
    "mahjong_sichuan": "sichuan",
    "mahjong_changsha": "changsha",
    "mahjong_taiwan": "taiwan",
    "mahjong_international": "international",
    "uno_seven_zero": "seven_zero",
    "uno_jump_in": "jump_in",
    "uno_stacking": "stacking",
    "uno_draw_until": "draw_until",
    "uno_strict_wild4": "strict_wild4",
}


def _variant_for(game_id: str, snapshot: dict[str, Any] | None) -> str | None:
    """Variant label for a match record (``None`` when the game has no variant).

    优先取快照自带字段（自定义游戏可由规则写入），其次按 game_id 派生
    （麻将七变种 / UNO 六变体）；基础游戏（``uno`` / ``werewolf`` 等）无变体
    概念 → ``None``，记录页显示「—」而不是把 game_id 当变体名。
    """
    if isinstance(snapshot, dict):
        for key in ("variant", "base_variant"):
            value = snapshot.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return _VARIANT_SUFFIXES.get(game_id)
