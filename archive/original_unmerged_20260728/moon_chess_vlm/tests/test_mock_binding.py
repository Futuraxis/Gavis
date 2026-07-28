from __future__ import annotations

import pytest

from binding.exceptions import (
    InvalidBoardError,
    InvalidConfidenceError,
    InvalidFrameSequenceError,
)
from binding.mock_binding import MockBinding


def test_mock_binding_outputs_observation() -> None:
    binding = MockBinding()
    observation = binding.parse(
        [
            ["X", None, "O"],
            [None, "X", None],
            ["O", None, None],
        ],
        frame_seq=0,
    )

    assert observation.frameSeq == 0
    assert observation.boardObservation[0][0] == "X"
    assert observation.confidence[2][2] == 1.0


def test_mock_binding_rejects_non_square_input() -> None:
    binding = MockBinding()

    with pytest.raises(InvalidBoardError):
        binding.parse([["X", None], [None, "O"]], frame_seq=0)


def test_mock_binding_rejects_invalid_symbol() -> None:
    binding = MockBinding()

    with pytest.raises(InvalidBoardError):
        binding.parse([["X", None, "A"], [None, "O", None], [None, None, None]], frame_seq=0)


def test_mock_binding_rejects_invalid_confidence() -> None:
    binding = MockBinding()

    with pytest.raises(InvalidConfidenceError):
        binding.parse(
            [["X", None, "O"], [None, "X", None], ["O", None, None]],
            confidence=[[1.0, 1.0, 1.0], [1.0, -0.1, 1.0], [1.0, 1.0, 1.0]],
            frame_seq=0,
        )


def test_mock_binding_requires_increasing_frame_seq() -> None:
    binding = MockBinding()
    binding.parse([["X", None, None], [None, None, None], [None, None, None]], frame_seq=1)

    with pytest.raises(InvalidFrameSequenceError):
        binding.parse([["X", None, "O"], [None, None, None], [None, None, None]], frame_seq=1)
