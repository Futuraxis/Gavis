"""Published online-learning models (Layer 4).

The empirical opponent table is plain data: ``{info_key: {action_key:
count}}`` — no Layer-3 types involved.  ``OnlineModelStore`` keeps the
published table per game (versioned, with gate metadata and one previous
version for rollback) and persists it under
``data/online_learning/models/<game_id>.json``.

App-layer assembly (``train-cli/games.py``, via the ``train_cli``
import bridge) consults this store when creating Texas Hold'em hybrid
solvers, so a newly published model
reaches every session started AFTER publication; running sessions keep
their own table (documented limitation of the single-process design).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _check_game_id(game_id: str) -> None:
    """Path-traversal-safe game_id whitelist (mirrors the store)."""
    if (
        not isinstance(game_id, str)
        or not game_id
        or "/" in game_id
        or "\\" in game_id
        or ".." in game_id
        or game_id != game_id.strip()
    ):
        raise ValueError(f"invalid game_id: {game_id!r}")


@dataclass
class PublishedModel:
    """One published empirical opponent table for a game."""

    game_id: str
    table: dict[str, dict[str, int]]
    version: int = 1
    samples: int = 0  # human decisions the table was built from
    coverage: int = 0  # info keys with counts
    published_at: str = field(default_factory=_now_iso)
    gate: dict | None = None  # candidate-vs-baseline gate metrics
    #: Rollback target (previous published version); not serialized.
    previous: PublishedModel | None = field(default=None, repr=False)

    def to_json(self) -> dict:
        """Serializable view (without the rollback pointer)."""
        return {
            "game_id": self.game_id,
            "table": self.table,
            "version": self.version,
            "samples": self.samples,
            "coverage": self.coverage,
            "published_at": self.published_at,
            "gate": self.gate,
        }

    @classmethod
    def from_json(cls, payload: dict) -> PublishedModel:
        return cls(
            game_id=str(payload.get("game_id", "")),
            table=payload.get("table", {}) or {},
            version=int(payload.get("version", 1)),
            samples=int(payload.get("samples", 0)),
            coverage=int(payload.get("coverage", 0)),
            published_at=str(payload.get("published_at", "")),
            gate=payload.get("gate"),
        )


class OnlineModelStore:
    """Thread-safe, file-backed registry of published empirical models."""

    def __init__(self, root: Path) -> None:
        #: root/<game_id>.json — the learner's data tree root.
        self._root = Path(root)
        self._models: dict[str, PublishedModel] = {}
        self._lock = threading.Lock()
        self._load_all()

    @property
    def root(self) -> Path:
        return self._root

    def _path(self, game_id: str) -> Path:
        _check_game_id(game_id)
        return self._root / f"{game_id}.json"

    # ── Persistence ──────────────────────────────────────────────

    def _load_all(self) -> None:
        for path in self._root.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                model = PublishedModel.from_json(payload)
                if model.game_id and Path(model.game_id + ".json").name == path.name:
                    self._models[model.game_id] = model
            except (OSError, ValueError, TypeError):
                continue

    def _persist(self, model: PublishedModel) -> None:
        path = self._path(model.game_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(model.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    # ── Registry API ────────────────────────────────────────────

    def current(self, game_id: str) -> PublishedModel | None:
        """The published model for a game (or None)."""
        _check_game_id(game_id)
        with self._lock:
            return self._models.get(game_id)

    def current_table(self, game_id: str) -> dict[str, dict[str, int]] | None:
        model = self.current(game_id)
        return model.table if model is not None else None

    def publish(
        self,
        game_id: str,
        table: dict[str, dict[str, int]],
        *,
        samples: int = 0,
        coverage: int = 0,
        gate: dict | None = None,
    ) -> PublishedModel:
        """Publish a new version (previous kept as the rollback target)."""
        _check_game_id(game_id)
        with self._lock:
            previous = self._models.get(game_id)
            model = PublishedModel(
                game_id=game_id,
                table=table,
                version=(previous.version + 1) if previous else 1,
                samples=samples,
                coverage=coverage,
                gate=gate,
                previous=previous,
            )
            self._models[game_id] = model
            self._persist(model)
            return model

    def revert(self, game_id: str) -> PublishedModel | None:
        """Roll back to the previous version (None if there is none)."""
        _check_game_id(game_id)
        with self._lock:
            model = self._models.get(game_id)
            previous = model.previous if model else None
            if previous is None:
                return None
            prev = PublishedModel(
                game_id=previous.game_id,
                table=previous.table,
                version=previous.version,
                samples=previous.samples,
                coverage=previous.coverage,
                published_at=previous.published_at,
                gate=previous.gate,
            )
            self._models[game_id] = prev
            self._persist(prev)
            return prev

    def clear(self, game_id: str) -> None:
        """Drop the stored model for one game (tests / reset)."""
        _check_game_id(game_id)
        with self._lock:
            self._models.pop(game_id, None)
            try:
                self._path(game_id).unlink()
            except FileNotFoundError:
                pass

    def game_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._models)

    def status(self, game_id: str) -> dict | None:
        """Public status view for the platform API."""
        model = self.current(game_id)
        if model is None:
            return None
        return {
            "version": model.version,
            "samples": model.samples,
            "coverage": model.coverage,
            "published_at": model.published_at,
            "gate": model.gate,
            "preview": list(model.table.items())[:5],  # small sample for transparency
        }
