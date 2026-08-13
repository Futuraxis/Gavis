"""Unified API-key resolution for LLM-backed components.

One reading flow for every LLM client in the project (audit 3.6,
decision 2026-08-13) — priority:

    显式构造参数 > 环境变量 > 默认值

The helper lives in Layer 2 because it is the lowest layer that both
Layer 3 (solvers: ollama / social LLM policy) and Layer 4 (binding:
qwen_vision) may legally import.
"""

from __future__ import annotations

import os


def resolve_api_key(param: str | None, env_var: str, default: str = "") -> str:
    """Resolve an API key: explicit parameter > env var > default.

    Empty/whitespace values are treated as unset, so callers receive a
    clean ``""`` when nothing is configured and can fail loudly at
    request time (or fall back to a local-endpoint convention via
    ``default``).
    """
    if param and param.strip():
        return param
    env_value = os.environ.get(env_var, "").strip()
    if env_value:
        return env_value
    return default
