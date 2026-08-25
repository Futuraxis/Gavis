"""Player profile & preference persistence (Layer 4).

One JSON file ``data/profile.json`` holds nickname, agent call, persona,
difficulty, pacing, hint level, learning/adaptive switches and per-game
win/play tallies.  Writes are atomic (temp file + ``os.replace``) and
serialized behind a ``threading.Lock`` so concurrent requests never
corrupt the file; ``clear`` deletes the file for a one-click reset.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

#: 默认数据目录（profile 文件为 ``data/profile.json``）。
DEFAULT_ROOT = Path("data")

#: profile 文件名。
PROFILE_FILENAME = "profile.json"

#: 完整默认 profile（约定 schema，C5/集成依赖）。``recent`` 按
#: ``<game_id>`` 归类 ``{wins, plays}`` 计数，默认空表。
DEFAULT_PROFILE: dict[str, Any] = {
    "nickname": "",
    "agent_call": "",
    "default_persona": "gentle",
    "default_difficulty": "normal",
    "hint_level": "off",
    "pacing": "standard",
    "adaptive": False,
    "difficulty_locked": False,
    "learning_enabled": True,
    "theme": "light",
    "recent": {},
}


def _default_profile() -> dict[str, Any]:
    """Return a fresh copy of the default profile (never the shared constant)."""
    return copy.deepcopy(DEFAULT_PROFILE)


class ProfileStore:
    """Filesystem-backed, thread-safe store of the player profile."""

    def __init__(self, root: Path = DEFAULT_ROOT) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._path = self._root / PROFILE_FILENAME
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        """Directory that holds ``profile.json``."""
        return self._root

    @property
    def path(self) -> Path:
        """Path of the profile file."""
        return self._path

    # ── Reading ──────────────────────────────────────────────────

    def load(self) -> dict[str, Any]:
        """Return the stored profile, or a fresh default when absent/corrupt."""
        with self._lock:
            try:
                text = self._path.read_text(encoding="utf-8")
            except (FileNotFoundError, OSError):
                return _default_profile()
            try:
                data = json.loads(text)
            except ValueError:
                return _default_profile()
        if not isinstance(data, dict):
            return _default_profile()
        return data

    # ── Writing ──────────────────────────────────────────────────

    def save(self, profile: dict[str, Any]) -> None:
        """Persist ``profile`` atomically (temp file + ``os.replace``)."""
        with self._lock:
            self._atomic_write(self._path, profile)

    def _atomic_write(self, path: Path, profile: dict[str, Any]) -> None:
        """Write via a temp file in the same directory, then rename.

        Mirrors ``MatchHistory._atomic_write``: ``mkstemp`` in the target
        directory + ``os.replace`` so a crash never leaves a partial file;
        the temp file is cleaned up on failure.
        """
        fd, tmp_name = tempfile.mkstemp(dir=self._root, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    # ── Clearing ─────────────────────────────────────────────────

    def clear(self) -> None:
        """Delete the profile file; missing file is a no-op (idempotent)."""
        with self._lock:
            try:
                self._path.unlink()
            except FileNotFoundError:
                pass
