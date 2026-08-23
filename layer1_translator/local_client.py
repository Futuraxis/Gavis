"""Local and optional debug LLM clients for Layer 1 rule translation."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

DEFAULT_LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "layer1-rule-llm"
_MAX_LLM_RESPONSE_BYTES = 4 * 1024 * 1024

logger = logging.getLogger(__name__)


class RuleLLMClient(Protocol):
    """Minimal chat-completion surface for rule generation."""

    def complete(self, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
        """Return the assistant's reply text for ``messages``."""


class LLMTranslatorError(Exception):
    """LLM translation failed before a valid candidate could be validated."""


# Inline copy of ``resolve_api_key`` (Layer 2 helper): Layer 1 keeps a
# single authorized L1→L2 channel (engine smoke validation only), so this
# pure convenience cascade lives here instead of importing Layer 2.
def _resolve_api_key(param: str | None, env_var: str, default: str = "") -> str:
    """Resolve an API key: explicit parameter > env var > default.

    Empty/whitespace values are treated as unset, so callers receive a
    clean ``""`` when nothing is configured.
    """
    if param and param.strip():
        return param
    env_value = os.environ.get(env_var, "").strip()
    if env_value:
        return env_value
    return default


@dataclass
class LocalTransformersRuleClient:
    """Local Hugging Face causal-LM client for trainable Layer 1 translation."""

    model_path: str | Path = DEFAULT_LOCAL_MODEL_DIR
    device: str | None = None
    max_new_tokens: int = 8192
    temperature: float = 0.2
    _torch: Any = field(init=False, default=None, repr=False)
    _tokenizer: Any = field(init=False, default=None, repr=False)
    _model: Any = field(init=False, default=None, repr=False)
    _device: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LLMTranslatorError("本地 Layer1 LLM 需要安装可选依赖: pip install 'gavis[llm]'") from exc

        path = Path(self.model_path)
        if not path.exists():
            raise LLMTranslatorError(f"本地 Layer1 LLM 模型不存在: {path}")
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(resolved_device)
        self._model = AutoModelForCausalLM.from_pretrained(path).to(self._device)
        self._model.eval()

    def complete(self, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
        """Generate a completion with a local model checkpoint.

        Any runtime failure (CUDA OOM, tokenizer error, context overflow)
        is wrapped as ``LLMTranslatorError`` so the orchestrator's
        template-fallback contract holds; ``KeyboardInterrupt`` passes
        through.
        """
        try:
            prompt = format_messages(messages, self._tokenizer)
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
            with self._torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, self.max_new_tokens),
                    do_sample=self.temperature > 0,
                    temperature=max(self.temperature, 1e-5),
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            generated = output[0][inputs["input_ids"].shape[-1] :]
            return self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            raise LLMTranslatorError(f"本地 LLM 生成失败: {type(exc).__name__}: {exc}") from exc

    @staticmethod
    def format_messages(messages: list[dict[str, str]], tokenizer: Any | None = None) -> str:
        """Compatibility wrapper around module-level chat formatting."""
        return format_messages(messages, tokenizer)


@dataclass
class OpenAICompatibleRuleClient:
    """Optional OpenAI-compatible debug adapter for local servers."""

    base_url: str = "http://127.0.0.1:11434/v1"
    api_key: str = ""
    model: str = "qwen3:8b"
    timeout_s: float = 60.0
    temperature: float = 0.2

    def __post_init__(self) -> None:
        self.api_key = _resolve_api_key(self.api_key, "LLM_API_KEY", default="ollama")

    def complete(self, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
        """Call an OpenAI-compatible chat-completions endpoint.

        Transport/parse failures are wrapped as ``LLMTranslatorError``;
        the response body is read with a size cap so an unexpected or
        malicious endpoint cannot balloon memory.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read(_MAX_LLM_RESPONSE_BYTES + 1)
                if len(raw) > _MAX_LLM_RESPONSE_BYTES:
                    raise LLMTranslatorError(f"LLM 响应超过 {_MAX_LLM_RESPONSE_BYTES} 字节上限")
                body = json.loads(raw.decode("utf-8"))
        except LLMTranslatorError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMTranslatorError(f"LLM request failed: {exc}") from exc
        try:
            return str(body["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMTranslatorError(f"Unexpected LLM response: {body!r}") from exc


def format_messages(messages: list[dict[str, str]], tokenizer: Any | None = None) -> str:
    """Format chat messages for local training/inference."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as exc:  # graceful fallback to the manual format
            logger.debug("apply_chat_template 回退到手动格式: %s", exc)
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        lines.append(f"<|{role}|>\n{content}")
    lines.append("<|assistant|>\n")
    return "\n".join(lines)
