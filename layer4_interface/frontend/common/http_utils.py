"""Shared HTTP helpers for frontend applications.

Each application under ``layer4_interface/frontend/`` runs its own
server; code that is genuinely shared across applications lives here.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler

#: 请求体大小上限（审计 3.6：按 Content-Length 无上限读取可被 OOM）。
MAX_BODY_BYTES = 10 * 1024 * 1024


def send_json(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: dict) -> None:
    """Write a JSON response body.

    CORS headers are the handler's responsibility (each server overrides
    ``end_headers`` once — review P2: 双重 CORS 头移除, send_json 不再
    与 end_headers 各发一次 Access-Control-Allow-*).
    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def send_error_json(handler: BaseHTTPRequestHandler, status: HTTPStatus, message: str) -> None:
    """Write a JSON error response."""
    send_json(handler, status, {"ok": False, "error": message})


class BodyTooLargeError(ValueError):
    """Request body exceeds ``MAX_BODY_BYTES``."""


def read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    """Read and parse the request body as JSON.

    Raises ``BodyTooLargeError`` when the declared Content-Length
    exceeds ``MAX_BODY_BYTES``; callers catch it to answer 413.
    """
    try:
        content_length = int(handler.headers.get("Content-Length", "0"))
    except ValueError as exc:
        raise BodyTooLargeError("invalid Content-Length header") from exc
    if content_length < 0 or content_length > MAX_BODY_BYTES:
        raise BodyTooLargeError(f"request body too large: {content_length} bytes")
    raw_body = handler.rfile.read(content_length)
    return json.loads(raw_body.decode("utf-8"))
