from __future__ import annotations

from pathlib import Path

import pytest

from binding.exceptions import ImageLoadError, InvalidFrameSequenceError, VisionModelResponseError
from binding.vision_binding import VisionLLMBinding


def test_vision_binding_parses_dict_response(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake_image_bytes")
    binding = VisionLLMBinding(client=_StubVisionClient())

    observation = binding.parse_image(str(image_path), frame_seq=0, observed_at=123456789)

    assert observation.frameSeq == 0
    assert observation.boardObservation[0][0] == "X"
    assert observation.confidence[2][2] == pytest.approx(0.92)
    assert observation.observedAt == 123456789


def test_vision_binding_accepts_json_string_response(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake_image_bytes")
    binding = VisionLLMBinding(client=_StubVisionClient(as_json_string=True))

    observation = binding.parse_image(str(image_path), frame_seq=0)

    assert observation.boardObservation[1][1] == "X"
    assert observation.source == "vision_model"


def test_vision_binding_rejects_missing_image() -> None:
    binding = VisionLLMBinding(client=_StubVisionClient())

    with pytest.raises(ImageLoadError):
        binding.parse_image("missing-page.png")


def test_vision_binding_rejects_invalid_model_payload(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake_image_bytes")
    binding = VisionLLMBinding(client=_BrokenVisionClient())

    with pytest.raises(VisionModelResponseError):
        binding.parse_image(str(image_path), frame_seq=0)


def test_vision_binding_requires_increasing_frame_seq(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake_image_bytes")
    binding = VisionLLMBinding(client=_StubVisionClient())
    binding.parse_image(str(image_path), frame_seq=1)

    with pytest.raises(InvalidFrameSequenceError):
        binding.parse_image(str(image_path), frame_seq=1)


class _StubVisionClient:
    def __init__(self, as_json_string: bool = False) -> None:
        self.as_json_string = as_json_string

    def infer_observation(self, *, image_bytes: bytes, mime_type: str, prompt: str) -> dict | str:
        assert image_bytes == b"fake_image_bytes"
        assert mime_type == "image/png"
        assert "3x3" in prompt
        payload = {
            "boardObservation": [["X", None, "O"], [None, "X", None], ["O", None, None]],
            "confidence": [[0.98, 0.93, 0.96], [0.91, 0.97, 0.95], [0.96, 0.94, 0.92]],
        }
        if self.as_json_string:
            return (
                '{"boardObservation":[["X",null,"O"],[null,"X",null],["O",null,null]],'
                '"confidence":[[0.98,0.93,0.96],[0.91,0.97,0.95],[0.96,0.94,0.92]]}'
            )
        return payload


class _BrokenVisionClient:
    def infer_observation(self, *, image_bytes: bytes, mime_type: str, prompt: str) -> str:
        return "not-json"
