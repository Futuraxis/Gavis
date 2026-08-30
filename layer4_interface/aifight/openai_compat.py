"""OpenAI-compatible local API for AIFight Bridge.

AIFight can call an OpenAI-compatible endpoint.  This server lets the bridge
treat local Gavis solvers as a model without exposing the solver as a public API.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from layer4_interface.botzone.runner import decide

DEFAULT_MODEL = "gavis-local"
MAX_BODY_BYTES = 2 * 1024 * 1024


def make_handler(*, token: str = "", model: str = DEFAULT_MODEL) -> type[BaseHTTPRequestHandler]:
    class AIFightOpenAICompatHandler(BaseHTTPRequestHandler):
        server_version = "GavisAIFightCompat/0.1"

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/v1/models":
                self._send_json(
                    HTTPStatus.OK,
                    {"object": "list", "data": [{"id": model, "object": "model", "created": 0, "owned_by": "gavis"}]},
                )
                return
            if self.path == "/healthz":
                self._send_json(HTTPStatus.OK, {"ok": True})
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found", "type": "not_found"}})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found", "type": "not_found"}})
                return
            if token and self.headers.get("Authorization") != f"Bearer {token}":
                self._send_json(HTTPStatus.UNAUTHORIZED, {"error": {"message": "unauthorized", "type": "auth_error"}})
                return
            try:
                payload = self._read_json()
                content = _decide_chat_completion(payload)
                self._send_json(HTTPStatus.OK, _chat_completion_envelope(payload, content, model))
            except Exception as exc:  # noqa: BLE001 - caller expects OpenAI-style JSON.
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": {"message": f"{type(exc).__name__}: {exc}", "type": "bad_request"}},
                )

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            raw_len = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_len)
            except ValueError as exc:
                raise ValueError("bad Content-Length") from exc
            if length > MAX_BODY_BYTES:
                raise ValueError("request body too large")
            body = self.rfile.read(length)
            value = json.loads(body.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("request body must be a JSON object")
            return value

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return AIFightOpenAICompatHandler


def _decide_chat_completion(payload: dict[str, Any]) -> str:
    text = _messages_text(payload.get("messages"))
    obj = _extract_latest_json(text)
    if isinstance(obj, dict):
        botzone_response = _try_botzone_decide(obj)
        if botzone_response is not None:
            return _json_text(botzone_response)
        legal_response = _choose_from_legal_actions(obj)
        if legal_response is not None:
            return _json_text(legal_response)
    return _json_text({"action": "pass"})


def _try_botzone_decide(obj: dict[str, Any]) -> Any | None:
    if isinstance(obj.get("requests"), list) or isinstance(obj.get("request"), (dict, str)):
        decision = decide(obj)
        return decision.response
    if "botzone" in obj and isinstance(obj["botzone"], dict):
        decision = decide(obj["botzone"])
        return decision.response
    return None


def _choose_from_legal_actions(obj: dict[str, Any]) -> Any | None:
    legal = obj.get("legal_actions") or obj.get("legalActions") or obj.get("actions")
    if not isinstance(legal, list) or not legal:
        return None
    for preferred in ("win", "hu", "ron", "tsumo", "gang", "kong", "peng", "pong", "chi", "call", "check"):
        for action in legal:
            if _action_name(action) == preferred:
                return action
    return legal[0]


def _action_name(action: Any) -> str:
    if isinstance(action, str):
        return action.strip().lower()
    if isinstance(action, dict):
        for key in ("action", "type", "name", "move"):
            value = action.get(key)
            if isinstance(value, str):
                return value.strip().lower()
    return ""


def _messages_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    chunks: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            chunks.extend(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return "\n".join(chunks)


def _extract_latest_json(text: str) -> Any | None:
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = fenced + _balanced_json_candidates(text)
    parsed: list[Any] = []
    for candidate in reversed(candidates):
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed.append(json.loads(candidate))
        except json.JSONDecodeError:
            continue
    for value in parsed:
        if isinstance(value, dict) and _looks_like_decision_context(value):
            return value
    for value in parsed:
        if isinstance(value, dict):
            return value
    if parsed:
        return parsed[0]
    return None


def _looks_like_decision_context(value: dict[str, Any]) -> bool:
    return any(key in value for key in ("requests", "request", "botzone", "legal_actions", "legalActions", "actions", "state"))


def _balanced_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for start, opener, closer in _json_starts(text):
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break
    return candidates


def _json_starts(text: str) -> list[tuple[int, str, str]]:
    starts: list[tuple[int, str, str]] = []
    for idx, char in enumerate(text):
        if char == "{":
            starts.append((idx, "{", "}"))
        elif char == "[":
            starts.append((idx, "[", "]"))
    return starts


def _chat_completion_envelope(request: dict[str, Any], content: str, model: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(request.get("model") or model),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an OpenAI-compatible local API for AIFight")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8789)
    parser.add_argument("--token", default="", help="optional bearer token AIFight must send as its API key")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(token=args.token, model=args.model))
    print(f"Gavis AIFight OpenAI-compatible API: http://{args.host}:{args.port}/v1")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
