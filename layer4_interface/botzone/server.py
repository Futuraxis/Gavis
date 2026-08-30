"""HTTP endpoint for remotely running Botzone decisions.

Run this on the machine that has the full Gavis project and Python 3.11+:

    python -m layer4_interface.botzone.server --host 0.0.0.0 --port 8788

Botzone can then upload the small Python 3.6 zip client built by
``scripts/build_botzone_zip.py --remote-url ...``.
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from layer4_interface.botzone.runner import decide

MAX_BODY_BYTES = 1024 * 1024


def make_handler(token: str = "") -> type[BaseHTTPRequestHandler]:
    class BotzoneRemoteHandler(BaseHTTPRequestHandler):
        server_version = "GavisBotzoneRemote/0.1"

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send_json(HTTPStatus.OK, {"ok": True})
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:
            if self.path not in {"/botzone/decide", "/api/botzone/decide"}:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                return
            if token and self.headers.get("Authorization") != f"Bearer {token}":
                self._send_json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "unauthorized"})
                return
            try:
                payload = self._read_json()
                decision = decide(payload)
                self._send_json(HTTPStatus.OK, decision.to_envelope())
            except Exception as exc:  # noqa: BLE001 - Botzone client needs a JSON envelope.
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"response": "PASS", "debug": f"{type(exc).__name__}: {exc}"[:1024], "data": "", "globaldata": ""},
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

    return BotzoneRemoteHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Gavis Botzone remote decision API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--token", default="", help="optional bearer token required from the Botzone upload client")
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(args.token))
    print(f"Gavis Botzone remote API: http://{args.host}:{args.port}/botzone/decide")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
