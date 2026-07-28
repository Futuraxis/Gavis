"""Binding Layer 自定义异常。"""


class BindingError(Exception):
    """Binding 基础异常。"""


class InvalidBoardError(BindingError):
    """棋盘观测格式不合法。"""


class InvalidConfidenceError(BindingError):
    """置信度矩阵不合法。"""


class InvalidFrameSequenceError(BindingError):
    """帧序号没有严格递增。"""


class ImageLoadError(BindingError):
    """图片加载失败。"""


class AmbiguousObservationError(BindingError):
    """当前帧变化过多，无法可靠推断。"""


class MissingHistoryError(BindingError):
    """缺少恢复顺序所需的历史信息。"""


class InvalidActionMaskError(BindingError):
    """动作掩码不合法。"""


class VisionModelResponseError(BindingError):
    """视觉模型返回结果无法解析或字段不合法。"""
