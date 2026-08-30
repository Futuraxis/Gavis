"""Platform LLM settings — persisted endpoint/model/key + env bridge.

The platform exposes a configuration page (``/api/llm/config``) so the
endpoint and model used by chat, agent dialogue, rule translation and the
Ollama solvers can be changed at runtime without code edits.  Persistence
is one JSON file in the data dir (``data/llm_config.json``, atomic write),
and the precedence is:

    **平台持久化配置 > 环境变量 (LLM_BASE_URL / LLM_MODEL / LLM_API_KEY) > 内置默认**

The env bridge (:func:`sync_env`) mirrors the stored values into the
process environment so consumers that construct ``LLMClient`` without
explicit parameters (the Layer-1 translator default client, the Ollama
solvers via ``train-cli`` registries) pick the platform settings up too.
Clearing the platform config restores the original process env (snapshot
taken at import), so a shell-level ``LLM_BASE_URL`` keeps working after
“恢复默认”.

The API key is stored locally in plaintext (single-machine tool, same
tradeoff as the rest of the platform data) but never echoed back —
``GET`` only reports ``has_api_key``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from layer2_engine.core.llm import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMClient

#: 平台配置会同步进这三个环境变量（见模块文档的优先级说明）。
ENV_KEYS = ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY")

#: 导入时刻的进程环境快照 —— “恢复默认”时还原，避免覆盖用户 shell 配置。
_ORIG_ENV: dict[str, str | None] = {key: os.environ.get(key) for key in ENV_KEYS}


@dataclass
class LLMSettings:
    """Persisted platform LLM settings (empty field = 未配置 → 回退 env/默认)."""

    base_url: str = ""
    model: str = ""
    api_key: str = ""


def _clean(value: object) -> str:
    """Coerce to a stripped string; ``None``/blank → ``""``."""
    return "" if value is None else str(value).strip()


def sync_env(settings: LLMSettings) -> None:
    """把平台配置镜像进进程环境变量；空字段还原导入时的快照值。

    这样 Layer-1 翻译默认客户端与 Ollama 求解器（两者都不以显式参数
    构造 ``LLMClient``）也能用上平台配置 —— 统一客户端按
    ``显式参数 > 环境变量 > 默认`` 解析，env 桥保证平台值生效。
    """
    for key, value in zip(ENV_KEYS, (settings.base_url, settings.model, settings.api_key)):
        if value:
            os.environ[key] = value
        else:
            original = _ORIG_ENV.get(key)
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


def probe_llm(base_url: str, api_key: str = "", timeout_s: float = 3.0) -> tuple[bool, str]:
    """探测 ``{base_url}/v1/models``；返回 ``(reachable, error_message)``。

    带可选 Bearer 鉴权头，方便云端点在保存前先验证端点+密钥。
    """
    url = f"{base_url.rstrip('/')}/v1/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if resp.status == 200:
                return True, ""
            return False, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code} {exc.reason}"
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return False, f"端点不可达/非法: {exc}"


class LLMSettingsStore:
    """``data/llm_config.json`` 的读写（原子写；结构宽容读）。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._settings = self._load()

    def _load(self) -> LLMSettings:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        return LLMSettings(
            base_url=_clean(raw.get("base_url")),
            model=_clean(raw.get("model")),
            api_key=_clean(raw.get("api_key")),
        )

    def _write(self, settings: LLMSettings) -> None:
        payload = {
            "base_url": settings.base_url,
            "model": settings.model,
            "api_key": settings.api_key,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def load(self) -> LLMSettings:
        return LLMSettings(**self._settings.__dict__)

    def save(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> LLMSettings:
        """合并保存：字段省略 → 保持不变；空串 → 清除；非空 → 设置。

        ``base_url`` 会做 ``http(s)://`` 格式校验（``ValueError``）。
        """
        updated = self._load()
        if base_url is not None:
            updated.base_url = _clean(base_url)
            if updated.base_url and not updated.base_url.startswith(("http://", "https://")):
                raise ValueError(f"非法端点: {updated.base_url!r}（需要 http(s):// 前缀）")
            updated.base_url = updated.base_url.rstrip("/")
        if model is not None:
            updated.model = _clean(model)
        if api_key is not None:
            updated.api_key = _clean(api_key)
        self._write(updated)
        self._settings = updated
        return self.load()

    def clear(self) -> LLMSettings:
        """清空平台配置（恢复 env/默认解析）。"""
        return self.save(base_url="", model="", api_key="")

    def signature(self) -> tuple[str, str, bool]:
        """配置指纹：平台值变化时调用方用来失效缓存（如聊天 LLM 单例）。"""
        settings = self._settings
        return (settings.base_url, settings.model, bool(settings.api_key))

    # ── 生效值解析（平台 > env > 默认）────────────────────────────

    def effective_base_url(self) -> str | None:
        """已生效端点（平台存储值 > LLM_BASE_URL > 内置默认）。"""
        if self._settings.base_url:
            return self._settings.base_url
        return os.environ.get("LLM_BASE_URL", "").strip() or DEFAULT_BASE_URL

    def effective_model(self) -> str | None:
        """已生效模型（平台存储值 > LLM_MODEL > 内置默认）。"""
        if self._settings.model:
            return self._settings.model
        return os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODEL

    def effective_api_key(self) -> str:
        """已生效密钥（平台存储值 > LLM_API_KEY > 空）。"""
        if self._settings.api_key:
            return self._settings.api_key
        return os.environ.get("LLM_API_KEY", "").strip()

    def source(self) -> str:
        """生效来源: ``platform`` / ``env`` / ``default``。"""
        if self._settings.base_url or self._settings.model:
            return "platform"
        if os.environ.get("LLM_BASE_URL", "").strip() or os.environ.get("LLM_MODEL", "").strip():
            return "env"
        return "default"

    def build_client(self) -> LLMClient:
        """用当前生效值构造统一客户端（供聊天/Agent 等显式参数场景）。"""
        return LLMClient(
            base_url=self.effective_base_url(),
            model=self.effective_model(),
            api_key=self.effective_api_key(),
        )

    def info(self) -> dict:
        """``GET /api/llm/config`` 的响应体（不回显密钥原文）。"""
        return {
            "base_url": self._settings.base_url,
            "model": self._settings.model,
            "has_api_key": bool(self._settings.api_key),
            "effective_base_url": self.effective_base_url(),
            "effective_model": self.effective_model(),
            "source": self.source(),
        }
