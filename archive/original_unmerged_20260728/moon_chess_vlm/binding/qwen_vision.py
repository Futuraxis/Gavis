"""Qwen 视觉识别客户端与命令行入口。"""

from __future__ import annotations

import argparse
import base64
import json
import os
import ssl
from pathlib import Path
from typing import Any
from urllib import error, request

from .exceptions import VisionModelResponseError
from .vision_binding import VisionLLMBinding


class QwenVisionClient:
    """使用 Qwen OpenAI 兼容接口调用视觉识别。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen3-vl-plus",
        temperature: float = 0.0,
        max_tokens: int = 512,
        stream: bool = False,
        timeout: int = 60,
        cafile: str | None = None,
        skip_ssl_verify: bool | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.base_url = os.environ.get("QWEN_BASE_URL", base_url).rstrip("/")
        self.model = os.environ.get("QWEN_MODEL", model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.timeout = timeout
        self.cafile = cafile or os.environ.get("QWEN_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
        env_skip_ssl = os.environ.get("QWEN_SKIP_SSL_VERIFY", "").strip().lower()
        self.skip_ssl_verify = (
            skip_ssl_verify
            if skip_ssl_verify is not None
            else env_skip_ssl in {"1", "true", "yes", "on"}
        )

    def infer_observation(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
    ) -> dict[str, Any] | str:
        if not self.api_key:
            raise VisionModelResponseError(
                "未检测到 Qwen API Key。请先设置环境变量 DASHSCOPE_API_KEY。"
            )

        payload = self._build_payload(image_bytes=image_bytes, mime_type=mime_type, prompt=prompt)
        response_json = self._post_json(payload)
        try:
            return response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VisionModelResponseError(f"Qwen 返回结构不符合预期: {response_json}") from exc

    def _build_payload(self, *, image_bytes: bytes, mime_type: str, prompt: str) -> dict[str, Any]:
        image_base64 = base64.b64encode(image_bytes).decode("ascii")
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a precise board-state recognition assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            "response_format": {"type": "json_object"},
        }

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout, context=self._build_ssl_context()) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise VisionModelResponseError(f"Qwen API 返回 HTTP {exc.code}: {details}") from exc
        except error.URLError as exc:
            reason = str(exc.reason)
            if "CERTIFICATE_VERIFY_FAILED" in reason:
                raise VisionModelResponseError(
                    "Qwen API SSL 证书校验失败。可先设置 QWEN_CA_BUNDLE=/path/to/cacert.pem，"
                    "或仅在本地调试时设置 QWEN_SKIP_SSL_VERIFY=1。"
                ) from exc
            raise VisionModelResponseError(f"Qwen API 请求失败: {exc.reason}") from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise VisionModelResponseError(f"Qwen 返回了无法解析的 JSON: {raw}") from exc

    def _build_ssl_context(self) -> ssl.SSLContext:
        if self.skip_ssl_verify:
            return ssl._create_unverified_context()
        if self.cafile:
            return ssl.create_default_context(cafile=self.cafile)
        return ssl.create_default_context()


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 Qwen 识别月亮棋页面截图。")
    parser.add_argument("image_path", type=str, help="页面截图路径")
    parser.add_argument("--frame-seq", type=int, default=0)
    parser.add_argument("--game-id", type=str, default="moon_demo_001")
    parser.add_argument("--base-url", type=str, default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--model", type=str, default="qwen3-vl-plus")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--cafile", type=str, default=None)
    parser.add_argument("--skip-ssl-verify", action="store_true")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise SystemExit(f"图片不存在: {image_path}")

    client = QwenVisionClient(
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        cafile=args.cafile,
        skip_ssl_verify=args.skip_ssl_verify,
    )
    binding = VisionLLMBinding(client=client, game_id=args.game_id, source_name="qwen_vision")
    observation = binding.parse_image(str(image_path), frame_seq=args.frame_seq)
    print(json.dumps(observation.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
