"""Tests for ProfileStore — atomic persistence and one-click clear."""

from __future__ import annotations

from pathlib import Path

import pytest

from layer4_interface.profile import DEFAULT_PROFILE, ProfileStore


@pytest.fixture
def store(tmp_path: Path) -> ProfileStore:
    return ProfileStore(tmp_path)


class TestLoad:
    def test_missing_returns_default(self, store: ProfileStore) -> None:
        assert store.load() == DEFAULT_PROFILE
        assert store.load()["recent"] == {}

    def test_corrupt_json_returns_default(self, store: ProfileStore) -> None:
        store.path.write_text("{not json", encoding="utf-8")
        assert store.load() == DEFAULT_PROFILE

    def test_non_object_json_returns_default(self, store: ProfileStore) -> None:
        store.path.write_text("[1, 2, 3]", encoding="utf-8")
        assert store.load() == DEFAULT_PROFILE


class TestSaveLoad:
    def test_round_trip(self, store: ProfileStore) -> None:
        profile = {
            "nickname": "阿远",
            "agent_call": "阿远",
            "default_persona": "teacher",
            "default_difficulty": "hard",
            "hint_level": "direction",
            "pacing": "fast",
            "adaptive": True,
            "difficulty_locked": False,
            "learning_enabled": True,
            "theme": "dark",
            "recent": {"moon_chess": {"wins": 3, "plays": 5}},
        }
        store.save(profile)
        assert store.load() == profile


class TestClear:
    def test_clear_removes_file(self, store: ProfileStore) -> None:
        store.save({"nickname": "x"})
        assert store.path.is_file()
        store.clear()
        assert not store.path.exists()

    def test_clear_is_idempotent(self, store: ProfileStore) -> None:
        store.clear()
        store.clear()
        assert not store.path.exists()


class TestDefaults:
    def test_default_root_creates_data_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        store = ProfileStore()
        assert (tmp_path / "data").is_dir()
        assert store.load() == DEFAULT_PROFILE
