"""Append-only JSONL persistence of learning trajectories (Layer 4).

One file per game: ``data/online_learning/<game_id>/trajectories.jsonl``.
Each line is one JSON object:

  - a decision record: ``{match_id, game_id, step, actor (human|ai),
    player, state, action, info_key, legal}``
  - a terminal record: ``{match_id, game_id, terminal: true, winner,
    utilities, human_pid, ai_pid, difficulty, started_at, finished_at}``

A match's decision lines always precede its terminal line, and the whole
match is appended as one atomic block (single lock + flush), so a crash
never leaves a half-written match interleaved with another one.  Reads
mirror ``MatchHistory``'s defensive style: corrupt lines are skipped
silently.

The data lives under ``data/`` (already gitignored) and is **never**
merged into the user-facing ``data/matches/`` records — hidden
information captured here must not leak into replay/history JSON.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

#: File name used inside each per-game directory.
TRAJECTORIES_FILE = "trajectories.jsonl"


class LearningStoreError(Exception):
    """Missing, unreadable, or corrupt learning data."""


class LearningStore:
    """Filesystem-backed, thread-safe append store of decision records."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    def _path(self, game_id: str) -> Path:
        _check_game_id(game_id)
        return self._root / game_id / TRAJECTORIES_FILE

    # ── Writing ──────────────────────────────────────────────────

    def append_match(self, game_id: str, decisions: list[dict], terminal: dict) -> None:
        """Append one finished match (decision lines + terminal line) atomically."""
        _check_game_id(game_id)
        records = [*decisions, terminal]
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        with self._lock:
            path = self._path(game_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(payload)
                f.flush()

    def clear(self, game_id: str) -> None:
        """Remove all stored data for one game (tests / config reset)."""
        _check_game_id(game_id)
        with self._lock:
            path = self._path(game_id)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            try:
                path.parent.rmdir()
            except OSError:
                pass

    def game_ids(self) -> list[str]:
        """Directories under the root that have ever been written."""
        if not self._root.is_dir():
            return []
        return sorted(p.name for p in self._root.iterdir() if p.is_dir())

    # ── Reading ──────────────────────────────────────────────────

    def read_records(self, game_id: str) -> list[dict]:
        """All records (decisions + terminals); corrupt lines skipped."""
        _check_game_id(game_id)
        path = self._path(game_id)
        if not path.exists():
            return []
        records: list[dict] = []
        with self._lock:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(record, dict):
                        records.append(record)
        return records

    def read_matches(self, game_id: str) -> list[dict]:
        """``[{"terminal": ..., "decisions": [...]}]`` for every finished match.

        Decisions are grouped by ``match_id`` and sorted by ``step``; the
        terminal record carries the outcome and utilities.  Oldest first.
        """
        records = self.read_records(game_id)
        by_match: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for record in records:
            match_id = record.get("match_id")
            if not match_id:
                continue
            if record.get("terminal"):
                block = by_match.setdefault(match_id, {"terminal": record, "decisions": []})
                block["terminal"] = record
                if match_id not in order:
                    order.append(match_id)
            else:
                block = by_match.setdefault(match_id, {"terminal": None, "decisions": []})
                block["decisions"].append(record)
                if match_id not in order:
                    order.append(match_id)
        result: list[dict] = []
        for match_id in order:
            block = by_match[match_id]
            if block["terminal"] is None:
                continue  # unfinished (store never writes these, but be safe)
            block["decisions"].sort(key=lambda d: int(d.get("step", 0)))
            result.append({"match_id": match_id, **block})
        return result

    def counts(self, game_id: str) -> dict[str, int]:
        """`{matches, decisions, human_decisions, ai_decisions}` for one game."""
        _check_game_id(game_id)
        matches = decisions = human = ai = 0
        for record in self.read_records(game_id):
            if record.get("terminal"):
                matches += 1
                continue
            decisions += 1
            if record.get("actor") == "ai":
                ai += 1
            else:
                human += 1
        return {
            "matches": matches,
            "decisions": decisions,
            "human_decisions": human,
            "ai_decisions": ai,
        }

    def trim(self, game_id: str, keep_matches: int = 500) -> int:
        """Rewrite the file keeping only the newest ``keep_matches`` matches.

        Returns the number of dropped matches.  The bound is defensive:
        learning data must not grow without limit on a long-lived server.
        """
        _check_game_id(game_id)
        if keep_matches <= 0:
            raise ValueError("keep_matches must be positive")
        with self._lock:
            path = self._path(game_id)
            if not path.exists():
                return 0
            blocks: list[list[str]] = []
            current: list[str] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    current.append(line + "\n")
                    if record.get("terminal"):
                        blocks.append(current)
                        current = []
            if current:  # dangling non-terminal tail (unfinished) — drop it
                pass
            if len(blocks) <= keep_matches:
                return 0
            dropped = len(blocks) - keep_matches
            kept = blocks[-keep_matches:]
            payload = "".join(line for block in kept for line in block)
            tmp = path.with_suffix(".jsonl.tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(path)
            return dropped


def _check_game_id(game_id: str) -> None:
    """Validate game_id against a path-traversal-safe whitelist."""
    if (
        not isinstance(game_id, str)
        or not game_id
        or "/" in game_id
        or "\\" in game_id
        or ".." in game_id
        or game_id != game_id.strip()
    ):
        raise LearningStoreError(f"invalid game_id: {game_id!r}")
