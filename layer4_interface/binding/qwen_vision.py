"""Qwen-VL (通义千问视觉) client implementation.

Connects to Alibaba Cloud's Qwen-VL API for board recognition.
"""

from __future__ import annotations

import base64
import json
import os

from layer2_engine.core.api_key import resolve_api_key

from .exceptions import VisionModelResponseError


class QwenVisionClient:
    """Client for Qwen-VL (通义千问视觉大模型).

    Configuration is read from environment variables:
        DASHSCOPE_API_KEY  — API key for DashScope (显式参数 > 环境变量)
        QWEN_BASE_URL      — base URL (optional)
        QWEN_MODEL         — model name (optional, default: qwen-vl-plus)
        QWEN_SKIP_SSL_VERIFY — set to 1 to skip SSL verification (optional)

    QWEN_SKIP_SSL_VERIFY 决策记录（审计 3.6，2026-08-13）：有意保留，仅
    用于本地开发/自签证书环境；生产环境必须走正常证书校验（可用
    QWEN_CA_BUNDLE / SSL_CERT_FILE 指定 CA 包），不要设置本变量。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        # 统一 api_key 读取流程（审计 3.6 决策 6）：显式参数 > 环境变量。
        self.api_key = resolve_api_key(api_key, "DASHSCOPE_API_KEY")
        self.base_url = base_url or os.environ.get(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model or os.environ.get("QWEN_MODEL", "qwen-vl-plus")
        skip = os.environ.get("QWEN_SKIP_SSL_VERIFY", "0")
        self.skip_ssl_verify = skip.lower() in ("1", "true", "yes")

    def infer_observation(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
    ) -> dict:
        """Send an image to Qwen-VL and parse the response."""
        import httpx

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"

        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }

        # 空 key 不发 Authorization 头（审查 P2：未配置 DASHSCOPE_API_KEY
        # 时不能带 "Bearer None" 头）；服务端自会返回 401 由调用方上报。
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        client_kwargs = {"verify": not self.skip_ssl_verify} if self.skip_ssl_verify else {}
        try:
            with httpx.Client(**client_kwargs) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=60.0,
                )
                resp.raise_for_status()
        except Exception as e:
            raise VisionModelResponseError(f"Qwen-VL API call failed: {e}") from e

        data = resp.json()
        # 裸 key 访问（审查：上游响应缺字段时避免 KeyError 穿透）——
        # VisionModelResponseError 由 vision server 映射为 400。
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise VisionModelResponseError(f"Qwen-VL 响应缺少内容字段: {e}") from e

        # Try to parse structured output from the response
        return self._parse_llm_output(content)

    def _parse_llm_output(self, content: str) -> dict:
        """Parse the LLM response into boardObservation and confidence."""
        # Try to find JSON in the response
        try:
            parsed = json.loads(content)
            if "boardObservation" in parsed:
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to extract JSON from markdown code blocks
        import re

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if "boardObservation" in parsed:
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass

        # Fallback: return raw content (caller may handle)
        return {
            "boardObservation": [[None] * 3 for _ in range(3)],
            "confidence": [[0.0] * 3 for _ in range(3)],
            "_raw_content": content,
        }
