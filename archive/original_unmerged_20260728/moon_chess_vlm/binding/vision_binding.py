"""基于大模型的纯视觉 Binding。"""

from __future__ import annotations

import json
import mimetypes
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from .base_binding import BaseBinding
from .exceptions import ImageLoadError, InvalidFrameSequenceError, VisionModelResponseError
from .schemas import Observation


VISION_PROMPT = """你是月亮棋页面截图识别器。

任务要求：
1. 输入是一张前端页面截图，页面中包含一个 3x3 月亮棋棋盘。
2. 只识别棋盘当前占用状态，不要尝试恢复完整历史顺序。
3. 对每个格子输出 "X"、"O" 或 null。
4. 同时为每个格子输出 0 到 1 之间的置信度。
5. 必须只返回一个 JSON 对象，不要输出解释文字，不要输出 Markdown。

JSON 字段要求：
{
  "boardObservation": [["X", null, "O"], [null, "X", null], ["O", null, null]],
  "confidence": [[0.98, 0.93, 0.96], [0.91, 0.97, 0.95], [0.96, 0.94, 0.92]]
}

额外约束：
- 如果页面上还有历史记录、状态面板、JSON 文本框，只把中间 3x3 棋盘作为识别目标。
- 不要从顺序 badge 或旁边文本中推断不存在于棋盘上的棋子。
- 如果某格不确定，仍然输出最可能的类别，并降低该格 confidence。
"""


class VisionModelClient(Protocol):
    """视觉模型客户端协议。"""

    def infer_observation(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
    ) -> dict[str, Any] | str:
        """返回可解析为 Observation 片段的 JSON 数据。"""


class VisionLLMBinding(BaseBinding):
    """将整张页面截图交给视觉大模型，返回 Observation。"""

    def __init__(
        self,
        client: VisionModelClient,
        *,
        game_id: str = "moon_demo_001",
        source_name: str = "vision_model",
        prompt: str = VISION_PROMPT,
    ) -> None:
        self.client = client
        self.game_id = game_id
        self.source_name = source_name
        self.prompt = prompt
        self._last_frame_seq = -1

    def parse(self, source: str) -> Observation:
        return self.parse_image(source)

    def parse_bytes(
        self,
        image_bytes: bytes,
        *,
        mime_type: str = "image/png",
        frame_seq: int | None = None,
        observed_at: int | None = None,
    ) -> Observation:
        if not image_bytes:
            raise ImageLoadError("图片为空或读取失败。")

        raw_result = self.client.infer_observation(
            image_bytes=image_bytes,
            mime_type=mime_type,
            prompt=self.prompt,
        )
        normalized = self._normalize_model_response(raw_result)
        next_frame_seq = self._resolve_frame_seq(frame_seq)
        try:
            return Observation(
                gameId=str(normalized.get("gameId", self.game_id)),
                source=str(normalized.get("source", self.source_name)),
                frameSeq=next_frame_seq,
                boardObservation=normalized["boardObservation"],
                confidence=normalized["confidence"],
                observedAt=int(normalized.get("observedAt", observed_at or int(time.time() * 1000))),
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise VisionModelResponseError(f"视觉模型返回字段不合法: {exc}") from exc

    def parse_image(
        self,
        image_path: str,
        *,
        frame_seq: int | None = None,
        observed_at: int | None = None,
    ) -> Observation:
        path = Path(image_path)
        if not path.exists():
            raise ImageLoadError(f"图片不存在: {image_path}")
        image_bytes = path.read_bytes()
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return self.parse_bytes(
            image_bytes,
            mime_type=mime_type,
            frame_seq=frame_seq,
            observed_at=observed_at,
        )

    def _normalize_model_response(self, raw_result: dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(raw_result, dict):
            return raw_result
        if not isinstance(raw_result, str):
            raise VisionModelResponseError("视觉模型返回类型不支持，必须是 dict 或 JSON 字符串。")

        payload = raw_result.strip()
        if payload.startswith("```"):
            lines = [line for line in payload.splitlines() if not line.strip().startswith("```")]
            payload = "\n".join(lines).strip()
        start_index = payload.find("{")
        end_index = payload.rfind("}")
        if start_index == -1 or end_index == -1 or end_index < start_index:
            raise VisionModelResponseError("视觉模型未返回可解析的 JSON 对象。")
        try:
            return json.loads(payload[start_index : end_index + 1])
        except json.JSONDecodeError as exc:
            raise VisionModelResponseError(f"视觉模型返回的 JSON 解析失败: {exc}") from exc

    def _resolve_frame_seq(self, frame_seq: int | None) -> int:
        if frame_seq is None:
            self._last_frame_seq += 1
            return self._last_frame_seq
        if frame_seq <= self._last_frame_seq:
            raise InvalidFrameSequenceError(
                f"frameSeq 必须严格递增，上一帧为 {self._last_frame_seq}，当前收到 {frame_seq}。"
            )
        self._last_frame_seq = frame_seq
        return frame_seq
