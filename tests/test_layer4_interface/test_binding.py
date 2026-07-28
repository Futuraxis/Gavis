"""Tests for Binding layer (Layer 4)."""

from __future__ import annotations

import pytest
import numpy as np

from layer4_interface.binding import (
    MockBinding,
    Observation,
    ImageBinding,
    TemplateMatchingClassifier,
    StateTracker,
    StateChange,
)


class TestObservation:
    def test_create(self):
        obs = Observation(
            gameId="test",
            source="manual",
            frameSeq=0,
            boardObservation=[[None, "X", None], [None, None, None], [None, "O", None]],
            confidence=[[0.0, 0.9, 0.0], [0.0, 0.0, 0.0], [0.0, 0.85, 0.0]],
        )
        assert obs.gameId == "test"
        assert obs.boardObservation[0][1] == "X"
        assert obs.boardObservation[2][1] == "O"

    def test_invalid_board_shape(self):
        with pytest.raises((ValueError, Exception)):
            Observation(
                boardObservation=[[None, None], [None]],  # not square
                confidence=[[0.0, 0.0], [0.0]],
            )


class TestMockBinding:
    def test_parse_returns_observation(self):
        binding = MockBinding()
        obs = binding.parse("")
        assert isinstance(obs, Observation)
        assert obs.source == "mock"

    def test_parse_image_increments_frame(self):
        binding = MockBinding()
        obs1 = binding.parse_image("")
        obs2 = binding.parse_image("")
        assert obs2.frameSeq > obs1.frameSeq
        assert obs2.frameSeq == obs1.frameSeq + 1


class TestStateTracker:
    def test_tracks_changes(self):
        tracker = StateTracker()
        obs1 = Observation(
            boardObservation=[[None, None, None], [None, None, None], [None, None, None]],
            confidence=[[0.0] * 3 for _ in range(3)],
        )
        change1 = tracker.update(obs1)
        assert len(change1.added) == 0

        obs2 = Observation(
            boardObservation=[["X", None, None], [None, None, None], [None, None, None]],
            confidence=[[0.9, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
        change2 = tracker.update(obs2)
        assert len(change2.added) == 1
        assert change2.added[0] == (0, 0, "X")

    def test_reset(self):
        tracker = StateTracker()
        obs = Observation(
            boardObservation=[[None] * 3 for _ in range(3)],
            confidence=[[0.0] * 3 for _ in range(3)],
        )
        tracker.update(obs)
        tracker.reset()
        assert tracker._last_board is None


class TestTemplateMatchingClassifier:
    def test_empty_cell(self):
        """A completely empty cell should return None."""
        classifier = TemplateMatchingClassifier()
        # All-white image → should be classified as empty
        empty = np.ones((30, 30, 3), dtype=np.uint8) * 255
        label, conf = classifier.classify(empty)
        assert label is None
        assert conf >= 0.0
