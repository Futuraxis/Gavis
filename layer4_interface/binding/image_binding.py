"""OpenCV-based image binding — splits a cropped board screenshot into cells."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from .exceptions import ImageLoadError, InvalidBoardError
from .schemas import Observation

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


class CellClassifier(Protocol):
    """Single-cell classifier protocol — replaceable by a neural net."""

    def classify(self, cell_image: np.ndarray) -> tuple[str | None, float]:
        """Return (piece_label, confidence) or (None, confidence) if empty."""
        ...


class TemplateMatchingClassifier:
    """Heuristic classifier for simple board screenshots.

    Works on light-background, dark-piece-line screenshots only.
    """

    def __init__(self, x_threshold: float = 0.12, o_threshold: float = 0.18) -> None:
        self.x_threshold = x_threshold
        self.o_threshold = o_threshold

    def classify(self, cell_image: np.ndarray) -> tuple[str | None, float]:
        gray = self._to_gray(cell_image)
        edges = self._detect_edges(gray)
        edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)
        if edge_ratio < 0.03:
            return None, max(0.0, 1.0 - (0.03 - edge_ratio) * 10)

        lines = self._detect_lines(edges)
        circles = self._detect_circles(gray)

        if lines >= 2 and circles == 0:
            confidence = min(0.99, 0.55 + edge_ratio + 0.1 * min(lines, 4))
            return "X", confidence
        if circles >= 1 and edge_ratio >= self.x_threshold:
            confidence = min(0.99, 0.6 + edge_ratio + 0.1 * circles)
            return "O", confidence
        if edge_ratio >= self.o_threshold and circles >= 1:
            return "O", min(0.95, 0.55 + edge_ratio)
        if lines >= 1:
            return "X", min(0.8, 0.45 + edge_ratio)
        return None, max(0.1, 0.5 - edge_ratio)

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ModuleNotFoundError("OpenCV required for ImageBinding.")
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ModuleNotFoundError("OpenCV required for ImageBinding.")
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, 50, 150)

    def _detect_lines(self, edges: np.ndarray) -> int:
        if cv2 is None:
            raise ModuleNotFoundError("OpenCV required for ImageBinding.")
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15, minLineLength=10, maxLineGap=8)
        return 0 if lines is None else len(lines)

    def _detect_circles(self, gray: np.ndarray) -> int:
        if cv2 is None:
            raise ModuleNotFoundError("OpenCV required for ImageBinding.")
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(8, gray.shape[0] // 4),
            param1=80,
            param2=18,
            minRadius=max(6, gray.shape[0] // 6),
            maxRadius=max(8, gray.shape[0] // 2),
        )
        return 0 if circles is None else circles.shape[1]


class ImageBinding:
    """Converts a pre-cropped 3×3 board screenshot to an Observation."""

    def __init__(
        self,
        classifier: CellClassifier | None = None,
        game_id: str = "moon_demo_001",
        source_name: str = "screen_capture",
    ) -> None:
        self.classifier = classifier or TemplateMatchingClassifier()
        self.game_id = game_id
        self.source_name = source_name
        self._last_frame_seq = -1
        # 帧序号自增是读-改-写，ThreadingHTTPServer 下需加锁（审计 3.6）。
        self._seq_lock = threading.Lock()

    def parse(self, source: str) -> Observation:
        return self.parse_image(source)

    def parse_image(
        self,
        image_path: str,
        *,
        frame_seq: int | None = None,
        observed_at: int | None = None,
    ) -> Observation:
        image = self._load_image(image_path)
        cells = self.split_board(image)
        board: list[list[str | None]] = []
        confidence: list[list[float]] = []

        for row in range(3):
            board_row: list[str | None] = []
            conf_row: list[float] = []
            for col in range(3):
                label, score = self.classifier.classify(cells[row][col])
                board_row.append(label)
                conf_row.append(float(score))
            board.append(board_row)
            confidence.append(conf_row)

        next_frame_seq = self._resolve_frame_seq(frame_seq)
        return Observation(
            gameId=self.game_id,
            source=self.source_name,
            frameSeq=next_frame_seq,
            boardObservation=board,
            confidence=confidence,
            observedAt=observed_at if observed_at is not None else int(time.time() * 1000),
        )

    def parse_bytes(self, data: bytes, mime_type: str, **kwargs) -> Observation:
        if cv2 is None:
            raise ModuleNotFoundError("OpenCV required for ImageBinding.")
        nparr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ImageLoadError("Failed to decode image bytes.")
        return self.parse_image_inline(image, **kwargs)

    def parse_image_inline(self, image: np.ndarray, **kwargs) -> Observation:
        cells = self.split_board(image)
        board: list[list[str | None]] = []
        confidence: list[list[float]] = []
        for row in range(3):
            board_row: list[str | None] = []
            conf_row: list[float] = []
            for col in range(3):
                label, score = self.classifier.classify(cells[row][col])
                board_row.append(label)
                conf_row.append(float(score))
            board.append(board_row)
            confidence.append(conf_row)
        return Observation(
            gameId=self.game_id,
            source=self.source_name,
            frameSeq=self._resolve_frame_seq(kwargs.get("frame_seq")),
            boardObservation=board,
            confidence=confidence,
            observedAt=kwargs.get("observed_at", int(time.time() * 1000)),
        )

    def split_board(self, image: np.ndarray) -> list[list[np.ndarray]]:
        if image.ndim not in (2, 3):
            raise InvalidBoardError("Image must be grayscale or color.")
        h, w = image.shape[:2]
        if h != w:
            raise InvalidBoardError("Only square board images are supported.")
        if h < 30:
            raise InvalidBoardError("Image too small to split into 3×3.")
        cell_size = h // 3
        cells: list[list[np.ndarray]] = []
        for row in range(3):
            row_cells: list[np.ndarray] = []
            for col in range(3):
                y0, x0 = row * cell_size, col * cell_size
                y1 = h if row == 2 else (row + 1) * cell_size
                x1 = w if col == 2 else (col + 1) * cell_size
                row_cells.append(image[y0:y1, x0:x1].copy())
            cells.append(row_cells)
        return cells

    def _load_image(self, path: str) -> np.ndarray:
        if cv2 is None:
            raise ModuleNotFoundError("OpenCV required for ImageBinding.")
        p = Path(path)
        if not p.exists():
            raise ImageLoadError(f"Image not found: {path}")
        img = cv2.imread(str(p))
        if img is None:
            raise ImageLoadError(f"Failed to load image: {path}")
        return img

    def _resolve_frame_seq(self, seq: int | None) -> int:
        with self._seq_lock:
            if seq is None:
                self._last_frame_seq += 1
                return self._last_frame_seq
            if seq <= self._last_frame_seq:
                raise InvalidBoardError(f"frameSeq must strictly increase. Last was {self._last_frame_seq}, got {seq}.")
            self._last_frame_seq = seq
            return seq
