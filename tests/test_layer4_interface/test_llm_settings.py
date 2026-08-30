"""Tests for the platform LLM settings store (data/llm_config.json) and env bridge.

Covers: persistence round-trip, effective-value resolution precedence
(平台配置 > env > 内置默认), validation, api-key omit/clear semantics, the
``sync_env`` bridge, and the chat-client cache rebuild on config change.
"""

from __future__ import annotations

import os

import pytest

from layer4_interface.frontend.platform.llm_settings import (
    LLMSettings,
    LLMSettingsStore,
    probe_llm,
    sync_env,
)


class TestLLMSettingsStore:
    def test_defaults_empty_and_effective_fallthrough(self, tmp_path: pytest.TempPathFactory) -> None:
        store = LLMSettingsStore(tmp_path / "llm_config.json")
        assert store.load() == LLMSettings(base_url="", model="", api_key="")
        assert store.effective_base_url() == "http://127.0.0.1:11434"
        assert store.effective_model() == "qwen3:8b"
        assert store.effective_api_key() == ""
        assert store.source() == "default"
        assert store.signature() == ("", "", False)

    def test_save_roundtrip_and_signature(self, tmp_path: pytest.TempPathFactory) -> None:
        store = LLMSettingsStore(tmp_path / "llm_config.json")
        store.save(base_url="https://api.deepseek.com", model="deepseek-chat", api_key="sk-x")
        assert store.load().base_url == "https://api.deepseek.com"
        assert store.effective_base_url() == "https://api.deepseek.com"
        assert store.effective_model() == "deepseek-chat"
        assert store.effective_api_key() == "sk-x"
        assert store.source() == "platform"
        assert store.signature() == ("https://api.deepseek.com", "deepseek-chat", True)
        # 持久化到 JSON 文件（原子写路径）
        raw = (tmp_path / "llm_config.json").read_text(encoding="utf-8")
        assert '"model": "deepseek-chat"' in raw

    def test_save_omitted_field_keeps(self, tmp_path: pytest.TempPathFactory) -> None:
        store = LLMSettingsStore(tmp_path / "llm_config.json")
        store.save(model="m1")
        store.save(base_url="http://127.0.0.1:1")  # 不动 model
        assert store.load().model == "m1"

    def test_empty_clears_and_scheme_validation(self, tmp_path: pytest.TempPathFactory) -> None:
        store = LLMSettingsStore(tmp_path / "llm_config.json")
        store.save(base_url="http://127.0.0.1:1")
        store.save(base_url="")
        assert store.load().base_url == ""
        with pytest.raises(ValueError, match="http"):
            store.save(base_url="not-a-url")

    def test_env_is_lower_priority_than_stored(self, tmp_path: pytest.TempPathFactory, monkeypatch) -> None:
        monkeypatch.setenv("LLM_BASE_URL", "http://env:1")
        monkeypatch.setenv("LLM_MODEL", "env-model")
        store = LLMSettingsStore(tmp_path / "llm_config.json")
        assert store.effective_base_url() == "http://env:1"
        assert store.source() == "env"
        store.save(base_url="http://platform:2")
        assert store.effective_base_url() == "http://platform:2"
        assert store.source() == "platform"
        store.clear()
        assert store.effective_base_url() == "http://env:1"  # 清空后回退 env
        assert store.source() == "env"

    def test_reload_from_disk_after_save(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "llm_config.json"
        LLMSettingsStore(path).save(base_url="http://h:1", model="m")
        again = LLMSettingsStore(path)
        assert again.load().base_url == "http://h:1"
        assert again.load().model == "m"


class TestSyncEnv:
    def test_sync_sets_then_restores_snapshot(self) -> None:
        from layer4_interface.frontend.platform.llm_settings import _ORIG_ENV

        before = {key: os.environ.get(key) for key in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY")}
        try:
            sync_env(LLMSettings(base_url="http://h:9", model="m", api_key="k"))
            assert os.environ.get("LLM_BASE_URL") == "http://h:9"
            assert os.environ.get("LLM_MODEL") == "m"
            assert os.environ.get("LLM_API_KEY") == "k"
            sync_env(LLMSettings())  # 清空 → 还原导入快照
            assert os.environ.get("LLM_BASE_URL") == _ORIG_ENV.get("LLM_BASE_URL")
            assert os.environ.get("LLM_MODEL") == _ORIG_ENV.get("LLM_MODEL")
            assert os.environ.get("LLM_API_KEY") == _ORIG_ENV.get("LLM_API_KEY")
        finally:
            for key, value in before.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class TestProbeLlm:
    def test_unreachable_reports_error(self) -> None:
        reachable, error = probe_llm("http://127.0.0.1:59990")
        assert reachable is False
        assert error

    def test_bad_scheme_reports_error(self) -> None:
        reachable, error = probe_llm("localhost:11434")
        assert reachable is False
        assert error


class TestChatLlmRebuild:
    """平台 LLM 配置保存后，聊天共享客户端必须用新端点/模型重建（缓存失效）。"""

    def test_rebuild_on_config_change(self, tmp_path: pytest.TempPathFactory, monkeypatch) -> None:
        import layer4_interface.frontend.platform.server as server_mod

        settings = LLMSettingsStore(tmp_path / "llm_config.json")
        monkeypatch.setattr(server_mod, "_LLM_SETTINGS", settings)
        monkeypatch.setattr(server_mod, "_CHAT_LLM", None)
        monkeypatch.setattr(server_mod, "_last_chat_probe", 0.0)
        monkeypatch.setattr(server_mod, "_CHAT_LLM_SIG", ())
        monkeypatch.setattr(server_mod, "_CHAT_LLM_PROBE_INTERVAL_S", 0.0)

        built: list[tuple[str, str]] = []

        def fake_build_client() -> object:
            built.append((settings.effective_base_url(), settings.effective_model()))
            return object()

        monkeypatch.setattr(settings, "build_client", fake_build_client)
        monkeypatch.setattr(server_mod.LLMClient, "available", staticmethod(lambda *a, **k: True))

        assert server_mod._get_chat_llm() is not None
        assert len(built) == 1
        # 配置变更 → 缓存失效 → 重建（用新模型）
        settings.save(model="changed-model")
        assert server_mod._get_chat_llm() is not None
        assert built[-1] == ("http://127.0.0.1:11434", "changed-model")
        # 未变更 → 复用缓存
        assert server_mod._get_chat_llm() is not None
        assert len(built) == 2
