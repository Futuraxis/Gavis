"""Regression tests for the audit 3.6 (security & performance) fixes.

Each test pins one fixed item from `docs/design/security-notes.md`:

  - 路径遍历: match_id whitelist in platform/history.py
  - 请求体无限制: read_json_body size cap (413 path)
  - Prompt 注入: LLM speech sanitization (length + control chars)
  - 统一 api_key 读取流程: resolve_api_key precedence
  - infoset key: compact sha256 keys (old full-JSON format retired)
  - PSRO 并行化: gamescape parallel evaluation + GymAdapter.clone
  - benchmark job 上限: _prune_locked
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from layer2_engine.interfaces.api_key import resolve_api_key

# ── 路径遍历 ──────────────────────────────────────────────────────


class TestMatchIdWhitelist:
    def test_record_get_delete_reject_traversal(self, tmp_path):
        from layer4_interface.frontend.platform.history import HistoryError, MatchHistory

        store = MatchHistory(tmp_path)
        for bad in ("../evil", "a/b", "a\\b", "a..b", "", "x" * 100):
            with pytest.raises(HistoryError):
                store.record({"match_id": bad})
            with pytest.raises(HistoryError):
                store.get(bad)
            with pytest.raises(HistoryError):
                store.delete(bad)
        # 无文件被写出到目录外/非法名
        assert list(tmp_path.iterdir()) == []

    def test_whitelisted_ids_roundtrip(self, tmp_path):
        from layer4_interface.frontend.platform.history import MatchHistory

        store = MatchHistory(tmp_path)
        match_id = store.record({"match_id": "ok-1_a9", "game_id": "moon_chess", "moves": []})
        assert match_id == "ok-1_a9"
        assert store.get("ok-1_a9")["match_id"] == "ok-1_a9"
        store.delete("ok-1_a9")
        assert list(tmp_path.iterdir()) == []


# ── 请求体上限 ────────────────────────────────────────────────────


class _FakeHandler:
    """Minimal handler surface for read_json_body."""

    def __init__(self, content_length: str) -> None:
        self.headers = {"Content-Length": content_length}
        self.rfile = io.BytesIO(b'{"a": 1}')

    def _send_cors_headers(self) -> None:
        pass


class TestBodySizeLimit:
    def test_oversized_body_rejected_without_read(self):
        from layer4_interface.frontend.common.http_utils import BodyTooLargeError, read_json_body

        handler = _FakeHandler(str(11 * 1024 * 1024))
        with pytest.raises(BodyTooLargeError):
            read_json_body(handler)
        assert handler.rfile.tell() == 0  # 未读取任何字节

    def test_invalid_content_length_rejected(self):
        from layer4_interface.frontend.common.http_utils import BodyTooLargeError, read_json_body

        with pytest.raises(BodyTooLargeError):
            read_json_body(_FakeHandler("abc"))
        with pytest.raises(BodyTooLargeError):
            read_json_body(_FakeHandler("-5"))

    def test_small_body_parsed(self):
        from layer4_interface.frontend.common.http_utils import read_json_body

        assert read_json_body(_FakeHandler("8")) == {"a": 1}


# ── Prompt 注入（发言清洗） ────────────────────────────────────────


class TestSpeechSanitization:
    def test_ollama_sanitize_caps_and_strips_control_chars(self):
        from layer3_solvers.llm.ollama_solver import OllamaSolver

        text = "a\x00b\x1fc" + "x" * 300
        cleaned = OllamaSolver._sanitize_speech(text, max_len=200)  # noqa: SLF001
        assert len(cleaned) == 200
        assert all(ord(ch) >= 0x20 for ch in cleaned)
        assert cleaned.startswith("abc")

    def test_llm_policy_output_sanitized(self):
        from layer3_solvers.social.base import LanguageObservation
        from layer3_solvers.social.llm_policy import LLMPolicy

        class FakeClient:
            def complete(self, messages, max_tokens: int = 200) -> str:
                return "\x00" + "很可疑。" * 100

        policy = LLMPolicy(FakeClient())
        speech = policy.decide_speech(LanguageObservation(role="villager", phase="speech"))
        assert "\x00" not in speech
        assert len(speech) <= 200


# ── 统一 api_key 读取流程 ─────────────────────────────────────────


class TestApiKeyResolution:
    def test_precedence_param_env_default(self, monkeypatch):
        monkeypatch.setenv("GAVIS_TEST_KEY", "from-env")
        assert resolve_api_key("param", "GAVIS_TEST_KEY", "dflt") == "param"
        assert resolve_api_key(None, "GAVIS_TEST_KEY", "dflt") == "from-env"
        assert resolve_api_key("", "GAVIS_TEST_KEY", "dflt") == "from-env"
        monkeypatch.delenv("GAVIS_TEST_KEY")
        assert resolve_api_key(None, "GAVIS_TEST_KEY", "dflt") == "dflt"
        assert resolve_api_key(None, "GAVIS_TEST_KEY") == ""

    def test_openai_client_resolution(self, monkeypatch):
        from layer3_solvers.social.llm_policy import OpenAICompatibleClient

        monkeypatch.setenv("LLM_API_KEY", "env-key")
        assert OpenAICompatibleClient().api_key == "env-key"
        assert OpenAICompatibleClient(api_key="param-key").api_key == "param-key"
        monkeypatch.delenv("LLM_API_KEY")
        assert OpenAICompatibleClient().api_key == "ollama"  # 本地默认

    def test_qwen_client_resolution(self, monkeypatch):
        from layer4_interface.binding.qwen_vision import QwenVisionClient

        monkeypatch.setenv("DASHSCOPE_API_KEY", "dash-key")
        assert QwenVisionClient().api_key == "dash-key"
        assert QwenVisionClient(api_key="param-key").api_key == "param-key"


# ── infoset key 紧凑哈希 ──────────────────────────────────────────


class TestInfoSetKey:
    def test_compact_sha256_key(self):
        from layer2_engine.games.moon_chess.moon_env_adapter import MoonChessAdapter

        adapter = MoonChessAdapter(seed=42)
        state = adapter.create_initial_state()
        key = adapter.get_info_set_key(state, "p_black")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)
        # 相同状态 → 相同 key（规范性）；完美信息空棋盘下双方观察
        # 完全一致，key 也一致（同观察 ⇒ 同信息集，这是正确语义）。
        assert adapter.get_info_set_key(state, "p_black") == key
        assert adapter.get_info_set_key(state, "p_white") == key

    def test_key_changes_after_move(self):
        from layer2_engine.games.moon_chess.moon_env_adapter import MoonChessAdapter

        adapter = MoonChessAdapter(seed=42)
        state = adapter.create_initial_state()
        before = adapter.get_info_set_key(state, "p_black")
        action = adapter.get_legal_actions(state)[0]
        after = adapter.get_info_set_key(adapter.apply_action(state, action), "p_black")
        assert before != after


# ── PSRO 并行化 ───────────────────────────────────────────────────


class TestPSROParallel:
    def test_gym_adapter_clone_is_independent(self):
        from layer2_engine.games.moon_chess.moon_env_adapter import MoonChessAdapter
        from layer3_solvers.psro.gym_adapter import GymAdapter

        env = GymAdapter(MoonChessAdapter(seed=42))
        clone = env.clone()
        assert clone._state is None  # noqa: SLF001
        env.reset()
        assert clone._state is None  # noqa: SLF001 — 互不影响
        clone.reset()
        assert env._state is not None and clone._state is not None  # noqa: SLF001

    def test_gamescape_parallel_antisymmetric(self):
        from layer2_engine.games.moon_chess.moon_env_adapter import MoonChessAdapter
        from layer3_solvers.psro.gym_adapter import GymAdapter
        from layer3_solvers.psro.meta_game import gamescape

        env = GymAdapter(MoonChessAdapter(seed=42))
        rng = np.random.RandomState(7)
        pi = [np.eye(9)[rng.randint(0, 9, 19683)] for _ in range(3)]
        matrix = gamescape(env, pi, Ne=1, num_workers=4)
        assert matrix.shape == (3, 3)
        assert np.allclose(matrix, -matrix.T)  # 反对称
        assert np.all(np.isfinite(matrix))
        assert np.allclose(np.diag(matrix), 0.0)


# ── benchmark job 清理 ────────────────────────────────────────────


class TestBenchmarkJobPruning:
    def test_prune_drops_finished_jobs_only(self):
        from demos.solver_provider import default_provider
        from layer4_interface.frontend.platform.benchmark import MAX_JOBS, BenchmarkJob, BenchmarkRunner

        runner = BenchmarkRunner(provider=default_provider, seed=1)
        # 直接填充注册表：1 个运行中 + MAX_JOBS 个已完成
        with runner._lock:  # noqa: SLF001
            for i in range(MAX_JOBS + 1):
                job = BenchmarkJob(
                    job_id=f"job{i:04d}", game_id="moon_chess", solver_a="mcts", solver_b="random", iterations=1
                )
                if i == 0:
                    job.status = "running"
                else:
                    job.status = "done"
                runner._jobs[job.job_id] = job  # noqa: SLF001
            runner._prune_locked()  # noqa: SLF001
            assert "job0000" in runner._jobs  # 运行中的保留
            assert len(runner._jobs) <= MAX_JOBS
