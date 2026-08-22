"""Local and optional debug LLM clients for Layer 1 rule translation."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from layer2_engine.interfaces.api_key import resolve_api_key

DEFAULT_LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "layer1-rule-llm"


class RuleLLMClient(Protocol):
    """Minimal chat-completion surface for rule generation."""

    def complete(self, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
        """Return the assistant's reply text for ``messages``."""


class LLMTranslatorError(Exception):
    """LLM translation failed before a valid candidate could be validated."""


@dataclass
class LocalTransformersRuleClient:
    """Local Hugging Face causal-LM client for trainable Layer 1 translation."""

    model_path: str | Path = DEFAULT_LOCAL_MODEL_DIR
    device: str | None = None
    max_new_tokens: int = 8192
    temperature: float = 0.2

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
        """Generate a completion with a local model checkpoint."""
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
        self.api_key = resolve_api_key(self.api_key, "LLM_API_KEY", default="ollama")

    def complete(self, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
        """Call an OpenAI-compatible chat-completions endpoint."""
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
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
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
        except Exception:
            pass
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        lines.append(f"<|{role}|>\n{content}")
    lines.append("<|assistant|>\n")
    return "\n".join(lines)
