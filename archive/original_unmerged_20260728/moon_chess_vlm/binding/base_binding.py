"""Binding 抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .schemas import Observation


class BaseBinding(ABC):
    """负责把外部输入转换为 Observation。"""

    @abstractmethod
    def parse(self, source: Any) -> Observation:
        """解析外部输入。"""
