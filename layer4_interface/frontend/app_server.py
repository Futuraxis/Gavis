"""Moon Chess AI Advisor — web server.

Exposes:
  - ``/`` — serves the frontend HTML
  - ``/api/recognize`` — accepts a screenshot, returns AI suggestion
"""

from __future__ import annotations

import argparse
import base64
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from layer4_interface.binding.qwen_vision import QwenVisionClient
from layer4_interface.binding.vision_binding import VisionLLMBinding
from layer4_interface.binding.exceptions import VisionModelResponseError

ROOT_DIR = Path(__file__).resolve().parent


class MoonChessHandler(SimpleHTTPRequestHandler):
    server_version = "MoonChessHTTP/0.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def do_OPTIONS(self) -> None:
        if self.path == "/api/recognize":
            self.send_response(HTTPStatus.NO_CONTENT)
            self._send_cors_headers()
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/api/recognize":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
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
            self._send_json(HTTPStatus.OK, {"ok": True, "observation": observation.model_dump()})
        except KeyError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"Missing field: {exc.args[0]}"})
        except (ValueError, VisionModelResponseError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})

    def end_headers(self) -> None:
        self._send_cors_headers()
        super().end_headers()

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _decode_image_payload(data_url: str) -> tuple[bytes, str]:
        if not data_url.startswith("data:") or "," not in data_url:
            raise ValueError("imageDataUrl is not a valid data URL.")
        header, encoded = data_url.split(",", 1)
        mime_type = header[5:].split(";")[0] or "image/png"
        return base64.b64decode(encoded), mime_type


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Moon Chess AI Advisor.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), MoonChessHandler)
    print(f"Server running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
