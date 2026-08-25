"""Texas Hold'em play app — human vs AI, independent HTTP server.

API (same shape as play_moon_chess / play_gomoku):
  - POST /api/start  {playerColor, difficulty} → session snapshot
  - POST /api/move   {game_id, choice, amount} → snapshot (AI replies)
  - POST /api/state  {game_id}                  → snapshot

The page itself is served from ``static/index.html``.

Usage:  python -m layer4_interface.frontend.play_texas_holdem.server [--host H] [--port P]
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


class TexasHoldemHandler(SimpleHTTPRequestHandler):
    server_version = "GavisTexasHoldem/0.1"
    # PlayManager 由 main() 注入（SolverProvider 装配在 train-cli/games.py）
    manager: PlayManager | None = None

    def __init__(self, *args, **kwargs) -> None:
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
        except (ValueError, UnicodeDecodeError) as exc:
            # 畸形 JSON / 非法 int 与 platform 语义一致（审查 M3）
            send_error_json(self, HTTPStatus.BAD_REQUEST, f"参数错误: {exc}")
        except Exception as exc:
            send_error_json(self, HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _handle_start(self, payload: dict) -> None:
        session = self.manager.start(
            playerColor=payload.get("playerColor", "random"),
            difficulty=payload.get("difficulty", "normal"),
        )
        send_json(self, HTTPStatus.OK, {"ok": True, "session": session.snapshot()})

    def _handle_move(self, payload: dict) -> None:
        session = self.manager.get(str(payload["gameId"]))
        amount = payload.get("amount")
        with session.lock:
            session.human_move(str(payload["choice"]), None if amount is None else int(amount))
        if session.over:
            self.manager.remove(session.game_id)
        send_json(self, HTTPStatus.OK, {"ok": True, "session": session.snapshot()})

    def _handle_state(self, payload: dict) -> None:
        session = self.manager.get(str(payload["gameId"]))
        send_json(self, HTTPStatus.OK, {"ok": True, "session": session.snapshot()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Texas Hold'em play app.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()

    # 求解器由注册表装配（SolverProvider 注入，L4 不 import L3）
    from train_cli import default_provider

    TexasHoldemHandler.manager = PlayManager(provider=default_provider)
    server = ThreadingHTTPServer((args.host, args.port), TexasHoldemHandler)
    print(f"Texas Hold'em app running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
