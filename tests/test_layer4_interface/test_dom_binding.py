"""Tests for DOM binding payload ingestion."""

from __future__ import annotations

import json

import pytest

from layer4_interface.binding import DomBinding, Observation
from layer4_interface.binding.exceptions import InvalidBoardError


def _payload(frame_seq: int = 0) -> dict:
    return {
        "gameId": "moon_dom",
        "source": "dom",
        "frameSeq": frame_seq,
        "boardObservation": [["X", None, "O"], [None, "X", None], ["O", None, None]],
        "confidence": [[1.0] * 3 for _ in range(3)],
        "observedAt": 123,
    }


class TestDomBinding:
    def test_parse_payload_returns_observation(self) -> None:
        binding = DomBinding()

        obs = binding.parse_payload(_payload())

        assert isinstance(obs, Observation)
        assert obs.source == "dom"
        assert obs.boardObservation[0] == ["X", None, "O"]

    def test_parse_json_string(self) -> None:
        binding = DomBinding()

        obs = binding.parse(json.dumps(_payload()))

        assert obs.gameId == "moon_dom"
        assert obs.confidence[1][1] == 1.0

    def test_parse_bytes_accepts_json(self) -> None:
        binding = DomBinding()

        obs = binding.parse_bytes(json.dumps(_payload()).encode("utf-8"), "application/json")

        assert obs.frameSeq == 0

    def test_rejects_invalid_board_shape(self) -> None:
        binding = DomBinding()
        payload = _payload()
        payload["boardObservation"] = [[None]]

        with pytest.raises(InvalidBoardError):
            binding.parse_payload(payload)

    def test_rejects_unknown_piece_value(self) -> None:
        binding = DomBinding()
        payload = _payload()
        payload["boardObservation"][0][0] = "Z"

        with pytest.raises(InvalidBoardError):
            binding.parse_payload(payload)

    def test_rejects_non_increasing_frame_seq(self) -> None:
        binding = DomBinding()
        binding.parse_payload(_payload(3))

        with pytest.raises(InvalidBoardError):
            binding.parse_payload(_payload(3))
