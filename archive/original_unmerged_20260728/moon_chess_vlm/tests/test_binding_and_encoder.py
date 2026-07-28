from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from binding.exceptions import AmbiguousObservationError, ImageLoadError, MissingHistoryError
from binding.image_binding import ImageBinding
from binding.mock_binding import MockBinding
from binding.state_tracker import StateTracker
from encoding import GameStateAdapter, MoonStateEncoder


def _make_state() -> dict:
    return {
        "gameId": "moon_demo_001",
        "seq": 4,
        "currentPlayerId": "player_x",
        "board": [["X", None, "O"], [None, "X", None], ["O", None, None]],
        "pieceOrder": {
            "player_x": [{"cellId": "cell_0_0", "placedSeq": 1}, {"cellId": "cell_1_1", "placedSeq": 3}],
            "player_o": [{"cellId": "cell_0_2", "placedSeq": 2}, {"cellId": "cell_2_0", "placedSeq": 4}],
        },
        "legalActions": ["cell_0_1", "cell_1_0", "cell_1_2", "cell_2_1", "cell_2_2"],
        "stepCount": 4,
        "status": "running",
        "winnerId": None,
    }


def test_image_binding_missing_image_raises() -> None:
    binding = ImageBinding()
    with pytest.raises(ImageLoadError):
        binding.parse_image("missing.png")


def test_image_binding_splits_board_into_nine_cells() -> None:
    binding = ImageBinding(classifier=_StubClassifier())
    image = np.zeros((90, 90, 3), dtype=np.uint8)
    cells = binding.split_board(image)
    assert len(cells) == 3
    assert len(cells[0]) == 3
    assert cells[2][2].shape[:2] == (30, 30)


def test_image_binding_uses_custom_classifier(tmp_path: Path) -> None:
    classifier = _StubClassifier(return_label="O", return_confidence=0.7)
    binding = ImageBinding(classifier=classifier)
    image_path = tmp_path / "board.png"
    _write_blank_board_image(image_path)
    observation = binding.parse_image(str(image_path), frame_seq=0)
    assert observation.boardObservation[1][1] == "O"
    assert observation.confidence[1][1] == pytest.approx(0.7)
    assert classifier.calls == 9


def test_state_tracker_identifies_single_added_piece() -> None:
    tracker = StateTracker()
    previous_state = _make_state()
    observation = MockBinding().parse(
        [["X", "X", "O"], [None, "X", None], ["O", None, None]],
        confidence=[[0.9] * 3 for _ in range(3)],
        frame_seq=0,
    )
    change = tracker.infer_state_change(previous_state, observation)
    assert change.added_cells == ["cell_0_1"]
    assert change.removed_cells == []
    assert change.inferred_actor_id == "player_x"


def test_state_tracker_identifies_add_and_same_color_removal() -> None:
    tracker = StateTracker()
    previous_state = _make_state()
    previous_state["board"] = [["X", None, "O"], ["X", "X", None], ["O", None, None]]
    previous_state["pieceOrder"]["player_x"] = [
        {"cellId": "cell_1_0", "placedSeq": 0},
        {"cellId": "cell_0_0", "placedSeq": 1},
        {"cellId": "cell_1_1", "placedSeq": 3},
    ]
    observation = MockBinding().parse(
        [[None, "X", "O"], ["X", "X", None], ["O", None, None]],
        confidence=[[0.95] * 3 for _ in range(3)],
        frame_seq=0,
    )
    change = tracker.infer_state_change(previous_state, observation)
    assert change.added_cells == ["cell_0_1"]
    assert change.removed_cells == ["cell_0_0"]
    assert change.ambiguous is False


def test_state_tracker_marks_multiple_changes_ambiguous() -> None:
    tracker = StateTracker()
    previous_state = _make_state()
    observation = MockBinding().parse(
        [["X", "X", "O"], ["O", "X", None], ["O", None, None]],
        confidence=[[0.95] * 3 for _ in range(3)],
        frame_seq=0,
    )
    with pytest.raises(AmbiguousObservationError):
        tracker.infer_state_change(previous_state, observation)


def test_state_tracker_ignores_low_confidence_cells() -> None:
    tracker = StateTracker(confidence_threshold=0.8)
    previous_state = _make_state()
    observation = MockBinding().parse(
        [["X", "X", "O"], [None, "X", None], ["O", None, None]],
        confidence=[[0.3, 0.3, 0.3], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3]],
        frame_seq=0,
    )
    change = tracker.infer_state_change(previous_state, observation)
    assert change.added_cells == []
    assert change.removed_cells == []


def test_state_tracker_requires_history() -> None:
    tracker = StateTracker()
    observation = MockBinding().parse(
        [["X", None, "O"], [None, "X", None], ["O", None, None]],
        confidence=[[0.95] * 3 for _ in range(3)],
        frame_seq=0,
    )
    with pytest.raises(MissingHistoryError):
        tracker.infer_state_change(None, observation)


def test_encoder_has_fixed_length() -> None:
    encoder = MoonStateEncoder(GameStateAdapter())
    encoded = encoder.encode(_make_state(), "player_x")
    assert encoded.shape == (38,)


def test_encoder_respects_perspective() -> None:
    encoder = MoonStateEncoder()
    encoded_x = encoder.encode(_make_state(), "player_x")
    encoded_o = encoder.encode(_make_state(), "player_o")
    assert not np.array_equal(encoded_x[:27], encoded_o[:27])


def test_encoder_distinguishes_piece_order() -> None:
    encoder = MoonStateEncoder()
    state_a = _make_state()
    state_b = _make_state()
    state_b["pieceOrder"]["player_x"] = [
        {"cellId": "cell_1_1", "placedSeq": 1},
        {"cellId": "cell_0_0", "placedSeq": 3},
    ]
    encoded_a = encoder.encode(state_a, "player_x")
    encoded_b = encoder.encode(state_b, "player_x")
    assert not np.array_equal(encoded_a, encoded_b)


def test_action_mask_has_expected_shape_and_values() -> None:
    encoder = MoonStateEncoder()
    mask = encoder.get_action_mask(_make_state())
    assert mask.shape == (9,)
    assert mask[0] == 0
    assert mask[1] == 1


def test_terminal_state_returns_all_zero_mask() -> None:
    encoder = MoonStateEncoder()
    state = _make_state()
    state["status"] = "finished"
    state["legalActions"] = []
    mask = encoder.get_action_mask(state)
    assert np.count_nonzero(mask) == 0


class _StubClassifier:
    def __init__(self, return_label: str | None = "X", return_confidence: float = 0.9) -> None:
        self.return_label = return_label
        self.return_confidence = return_confidence
        self.calls = 0

    def classify(self, cell_image: np.ndarray) -> tuple[str | None, float]:
        self.calls += 1
        return self.return_label, self.return_confidence


def _write_blank_board_image(path: Path) -> None:
    import cv2

    image = np.full((90, 90, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(path), image)
