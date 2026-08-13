"""Moon Chess play app — human vs AI, independent HTTP server.

API:
  - POST /api/start  {player_color, difficulty} → session snapshot
  - POST /api/move   {game_id, cell_index}      → snapshot (AI replies)
  - POST /api/state  {game_id}                  → snapshot

The page itself is served from ``static/index.html``.

Usage:  python -m layer4_interface.frontend.play_moon_chess.server [--host H] [--port P]
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..common.http_utils import BodyTooLargeError, read_json_body, send_error_json, send_json
from .session import PlayError, PlayManager

ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"


class PlayHandler(SimpleHTTPRequestHandler):
    server_version = "GavisMoonChess/0.1"
    manager = PlayManager()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    # ── CORS ────────────────────────────────────────────────────────

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def end_headers(self) -> None:
        self._send_cors_headers()
        super().end_headers()

    def do_OPTIONS(self) -> None:
        if self.path.startswith("/api/"):
            self.send_response(HTTPStatus.NO_CONTENT)
            self._send_cors_headers()
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    # ── API ─────────────────────────────────────────────────────────

    def do_POST(self) -> None:
        try:
            payload = read_json_body(self)
            if self.path == "/api/start":
                self._handle_start(payload)
            elif self.path == "/api/move":
                self._handle_move(payload)
            elif self.path == "/api/state":
                self._handle_state(payload)
            else:
                send_error_json(self, HTTPStatus.NOT_FOUND, "Not found")
        except BodyTooLargeError as exc:
            send_error_json(self, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, str(exc))
        except PlayError as exc:
            send_error_json(self, HTTPStatus.BAD_REQUEST, str(exc))
        except KeyError as exc:
            send_error_json(self, HTTPStatus.BAD_REQUEST, f"Missing field: {exc.args[0]}")
        except Exception as exc:
            send_error_json(self, HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _handle_start(self, payload: dict) -> None:
        session = self.manager.start(
            player_color=payload.get("playerColor", "p_black"),
            difficulty=payload.get("difficulty", "normal"),
        )
        send_json(self, HTTPStatus.OK, {"ok": True, "session": session.snapshot()})

    def _handle_move(self, payload: dict) -> None:
        session = self.manager.get(str(payload["gameId"]))
        cell_index = int(payload["cellIndex"])
        session.human_move(cell_index)
        session.ai_move()
        send_json(self, HTTPStatus.OK, {"ok": True, "session": session.snapshot()})

    def _handle_state(self, payload: dict) -> None:
        session = self.manager.get(str(payload["gameId"]))
        send_json(self, HTTPStatus.OK, {"ok": True, "session": session.snapshot()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Moon Chess play app.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), PlayHandler)
    print(f"Moon Chess play app running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
