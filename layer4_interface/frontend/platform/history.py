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

    def list_matches(self, limit: int = 100, game_id: str | None = None) -> list[dict]:
        """Metadata of finished matches, newest first.

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
        return matches[:limit]

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
