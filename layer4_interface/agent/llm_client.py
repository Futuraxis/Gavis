"""llm_client — 最小 urllib 客户端到本地 Ollama（Layer 4，可选）.

对话引擎的 LLM 半边是可选的：本模块用标准库 ``urllib`` 直连本地
``http://127.0.0.1:11434``，短超时，一切异常吞掉 —— ``complete`` 失败
返回 ``""``（对话引擎据此回退兜底台词），``available()`` 探测失败返回
``False``。不依赖任何第三方 HTTP 库，也不 import Layer 3。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

_DEFAULT_BASE_URL = "http://127.0.0.1:11434"
_DEFAULT_MODEL = "qwen3:8b"
_TIMEOUT_S = 3.0
_PROBE_TIMEOUT_S = 1.0


class OllamaClient:
    """Minimal Ollama ``/api/chat`` client with fail-soft semantics."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_s: float = _TIMEOUT_S,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Return the assistant reply text, or ``""`` on any failure.

        Args:
            system: 系统提示（人设 + 隐藏信息红线）。
            user: 用户提示（场景 + 机械事实）。
            max_tokens: 生成上限（映射到 ``options.num_predict``）。

        Returns:
            助手文本；超时 / 连接失败 / 异常均返回空串。
        """
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload.get("message", {}).get("content", "")
            return content if isinstance(content, str) else ""
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            return ""

    @staticmethod
    def available() -> bool:
        """Probe the local Ollama server; ``False`` when unreachable."""
        try:
            with urllib.request.urlopen(f"{_DEFAULT_BASE_URL}/api/tags", timeout=_PROBE_TIMEOUT_S) as response:
                return response.status == 200
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return False
