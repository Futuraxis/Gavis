from __future__ import annotations

import json
from urllib import error

import pytest

from binding.exceptions import VisionModelResponseError
from binding.qwen_vision import QwenVisionClient


def test_build_payload_contains_qwen_model_and_image() -> None:
    client = QwenVisionClient(api_key="test-key")

    payload = client._build_payload(
        image_bytes=b"fake",
        mime_type="image/png",
        prompt="recognize board",
    )

    assert payload["model"] == "qwen3-vl-plus"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_infer_observation_requires_api_key() -> None:
    client = QwenVisionClient(api_key=None)

    with pytest.raises(VisionModelResponseError):
        client.infer_observation(image_bytes=b"fake", mime_type="image/png", prompt="recognize")


def test_post_json_parses_content(monkeypatch: pytest.MonkeyPatch) -> None:
    client = QwenVisionClient(api_key="test-key")

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"boardObservation":[["X",null,null],[null,null,null],[null,null,null]],"confidence":[[0.9,0.9,0.9],[0.9,0.9,0.9],[0.9,0.9,0.9]]}'
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        "binding.qwen_vision.request.urlopen",
        lambda req, timeout, context=None: _Response(),
    )
    result = client.infer_observation(image_bytes=b"fake", mime_type="image/png", prompt="recognize")
    assert isinstance(result, str)
    assert "boardObservation" in result


def test_post_json_surfaces_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    client = QwenVisionClient(api_key="test-key")

    class _HttpError(error.HTTPError):
        def __init__(self) -> None:
            super().__init__("http://test", 400, "Bad Request", hdrs=None, fp=None)

        def read(self) -> bytes:
            return b'{"error":"bad_request"}'

    def _raise(req, timeout, context=None):
        raise _HttpError()

    monkeypatch.setattr("binding.qwen_vision.request.urlopen", _raise)
    with pytest.raises(VisionModelResponseError):
        client.infer_observation(image_bytes=b"fake", mime_type="image/png", prompt="recognize")
