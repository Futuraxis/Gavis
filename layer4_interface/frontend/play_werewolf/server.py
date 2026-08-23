"""Werewolf play app — human vs local-LLM AI players, HTTP server.

API (same shape as play_texas_holdem):
  - POST /api/start  {players, wolves, model, humanPid} → snapshot
  - POST /api/move   {gameId, action, intent, text, target} → snapshot
  - POST /api/state  {gameId}                              → snapshot

``action`` is one of: speak / kill / check / poison / heal / guard /
shoot / shoot_lynched / vote.  For ``speak`` the human provides
``intent`` + ``text``; for the others ``target`` ("pX" or "pass").

The page itself is served from ``static/index.html``.

Usage:  python -m layer4_interface.frontend.play_werewolf.server [--host H] [--port P]
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


class WerewolfHandler(SimpleHTTPRequestHandler):
    server_version = "GavisWerewolf/0.1"
    # PlayManager 由 main() 注入（SolverProvider 装配在应用层 demos/）
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
            players=int(payload.get("players", 9)),
            wolves=int(payload.get("wolves", 3)),
            model=str(payload.get("model", "qwen3:8b")),
            human_pid=payload.get("humanPid"),
        )
        send_json(self, HTTPStatus.OK, {"ok": True, "session": session.snapshot()})

    def _handle_move(self, payload: dict) -> None:
        session = self.manager.get(str(payload["gameId"]))
        action = str(payload.get("action", ""))
        params = {
            "intent": payload.get("intent"),
            "text": str(payload.get("text", "")),
            "target": payload.get("target"),
        }
        with session.lock:
            session.human_move(action, params)
        if session.over:
            self.manager.remove(session.game_id)
        send_json(self, HTTPStatus.OK, {"ok": True, "session": session.snapshot()})

    def _handle_state(self, payload: dict) -> None:
        session = self.manager.get(str(payload["gameId"]))
        send_json(self, HTTPStatus.OK, {"ok": True, "session": session.snapshot()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Werewolf play app.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()

    # 求解器由应用层装配（SolverProvider 注入，L4 不 import L3）
    from demos.solver_provider import default_provider

    WerewolfHandler.manager = PlayManager(provider=default_provider)
    server = ThreadingHTTPServer((args.host, args.port), WerewolfHandler)
    print(f"Werewolf app running at http://{args.host}:{args.port}/")
    print("  (本地 ollama 需运行: ollama serve; 默认模型 qwen3:8b)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
