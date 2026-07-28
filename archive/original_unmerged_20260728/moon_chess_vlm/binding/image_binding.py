"""基于 OpenCV 的简单图片 Binding。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import numpy as np

from .base_binding import BaseBinding
from .exceptions import ImageLoadError, InvalidBoardError
from .schemas import Observation

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - 依赖缺失时保留到运行期报错
    cv2 = None


class CellClassifier(Protocol):
    """单格分类器接口，便于后续替换为神经网络。"""

    def classify(self, cell_image: np.ndarray) -> tuple[str | None, float]:
        """返回棋子类别和置信度。"""


class TemplateMatchingClassifier:
    """第一版启发式分类器，仅适用于浅色背景、深色棋子线条的固定截图。"""

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

    def _to_gray(self, cell_image: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ModuleNotFoundError("需要安装 opencv-python 才能使用 ImageBinding。")
        if cell_image.ndim == 2:
            return cell_image
        return cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)

    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ModuleNotFoundError("需要安装 opencv-python 才能使用 ImageBinding。")
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, 50, 150)

    def _detect_lines(self, edges: np.ndarray) -> int:
        if cv2 is None:
            raise ModuleNotFoundError("需要安装 opencv-python 才能使用 ImageBinding。")
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15, minLineLength=10, maxLineGap=8)
        return 0 if lines is None else len(lines)

    def _detect_circles(self, gray: np.ndarray) -> int:
        if cv2 is None:
            raise ModuleNotFoundError("需要安装 opencv-python 才能使用 ImageBinding。")
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


class ImageBinding(BaseBinding):
    """将已经裁剪好的 3x3 棋盘截图转换为 Observation。"""

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

    def split_board(self, image: np.ndarray) -> list[list[np.ndarray]]:
        if image.ndim not in (2, 3):
            raise InvalidBoardError("输入图片维度不正确，必须是灰度图或彩色图。")
        height, width = image.shape[:2]
        if height != width:
            raise InvalidBoardError("当前版本只支持已经裁剪好的正方形棋盘截图。")
        if height < 30:
            raise InvalidBoardError("输入图片过小，无法稳定切分为 3x3 棋盘。")

        cell_size = height // 3
        cells: list[list[np.ndarray]] = []
        for row in range(3):
            row_cells: list[np.ndarray] = []
            for col in range(3):
                y0 = row * cell_size
                x0 = col * cell_size
                y1 = height if row == 2 else (row + 1) * cell_size
                x1 = width if col == 2 else (col + 1) * cell_size
                row_cells.append(image[y0:y1, x0:x1].copy())
            cells.append(row_cells)
        return cells

    def _load_image(self, image_path: str) -> np.ndarray:
        if cv2 is None:
            raise ModuleNotFoundError("需要安装 opencv-python 才能使用 ImageBinding。")
        path = Path(image_path)
        if not path.exists():
            raise ImageLoadError(f"图片不存在: {image_path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ImageLoadError(f"图片读取失败或格式不受支持: {image_path}")
        return image

    def _resolve_frame_seq(self, frame_seq: int | None) -> int:
        if frame_seq is None:
            self._last_frame_seq += 1
            return self._last_frame_seq
        if frame_seq <= self._last_frame_seq:
            raise InvalidBoardError(
                f"frameSeq 必须严格递增，上一帧为 {self._last_frame_seq}，当前收到 {frame_seq}。"
            )
        self._last_frame_seq = frame_seq
        return frame_seq
