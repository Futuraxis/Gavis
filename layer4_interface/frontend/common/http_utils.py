"""Shared HTTP helpers for frontend applications.

Each application under ``layer4_interface/frontend/`` runs its own
server; code that is genuinely shared across applications lives here.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler


def send_json(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: dict) -> None:
    """Write a JSON response with CORS headers."""
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Content-Length', str(len(body)))
    handler._send_cors_headers()
    handler.end_headers()
    handler.wfile.write(body)


def send_error_json(handler: BaseHTTPRequestHandler, status: HTTPStatus, message: str) -> None:
    """Write a JSON error response."""
    send_json(handler, status, {'ok': False, 'error': message})


def read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    """Read and parse the request body as JSON."""
    content_length = int(handler.headers.get('Content-Length', '0'))
    raw_body = handler.rfile.read(content_length)
    return json.loads(raw_body.decode('utf-8'))
