"""Vision recognition app — independent HTTP server.

The vision app accepts screenshots and returns AI observations via the
VLM binding pipeline.  It runs as its own server (no shared frontend
server); shared HTTP helpers live in ``frontend.common``.

Usage:  python -m layer4_interface.frontend.vision.server [--host H] [--port P]
"""

from __future__ import annotations

import argparse
import base64
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from layer4_interface.binding.exceptions import VisionModelResponseError
from layer4_interface.binding.qwen_vision import QwenVisionClient
from layer4_interface.binding.vision_binding import VisionLLMBinding

from ..common.http_utils import read_json_body, send_error_json, send_json

ROOT_DIR = Path(__file__).resolve().parent


class VisionHandler(SimpleHTTPRequestHandler):
    server_version = "GavisVision/0.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR / 'static'), **kwargs)

    # ── CORS ────────────────────────────────────────────────────────

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def end_headers(self) -> None:
        self._send_cors_headers()
        super().end_headers()

    def do_OPTIONS(self) -> None:
        if self.path == "/api/recognize":
            self.send_response(HTTPStatus.NO_CONTENT)
            self._send_cors_headers()
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    # ── API ─────────────────────────────────────────────────────────

    def do_POST(self) -> None:
        if self.path != "/api/recognize":
            send_error_json(self, HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            payload = read_json_body(self)
            image_bytes, mime_type = self._decode_image_payload(payload["imageDataUrl"])
            frame_seq = int(payload.get("frameSeq", 0))

            binding = VisionLLMBinding(
                client=QwenVisionClient(),
                game_id=str(payload.get("gameId", "moon_demo_001")),
                source_name="qwen_vision",
            )
            observation = binding.parse_bytes(
                image_bytes, mime_type=mime_type, frame_seq=frame_seq,
            )
            send_json(self, HTTPStatus.OK,
                      {"ok": True, "observation": observation.model_dump()})
        except KeyError as exc:
            send_error_json(self, HTTPStatus.BAD_REQUEST, f"Missing field: {exc.args[0]}")
        except (ValueError, VisionModelResponseError) as exc:
            send_error_json(self, HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            send_error_json(self, HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    @staticmethod
    def _decode_image_payload(data_url: str) -> tuple[bytes, str]:
        if not data_url.startswith("data:") or "," not in data_url:
            raise ValueError("imageDataUrl is not a valid data URL.")
        header, encoded = data_url.split(",", 1)
        mime_type = header[5:].split(";")[0] or "image/png"
        return base64.b64decode(encoded), mime_type


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the vision recognition app.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), VisionHandler)
    print(f"Vision app running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
