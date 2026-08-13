"""Observation schema — pydantic model for Binding output.

When pydantic is unavailable a dataclass fallback takes over: ``Field``
and ``field_validator`` degrade gracefully and the class is rebuilt with
``@dataclass`` so ``Observation(...)`` constructs and ``model_dump()``
works (C-04 — the old fallback left ``Field`` objects as class attributes,
so construction crashed and attribute access returned ``Field`` objects).
"""

from __future__ import annotations

<<<<<<< HEAD
from copy import deepcopy
from typing import Any

try:
    from pydantic import BaseModel, Field, field_validator
except ImportError:
    # Fallback model if pydantic is not available.
    class BaseModel:
        def __init__(self, **kwargs: Any) -> None:
            for name in getattr(type(self), "__annotations__", {}):
                if name in kwargs:
                    value = kwargs[name]
                else:
                    value = deepcopy(getattr(type(self), name))
                setattr(self, name, value)

        def model_dump(self) -> dict[str, Any]:
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def Field(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
        if "default_factory" in kwargs:
            return kwargs["default_factory"]()
        return kwargs.get("default")

    def field_validator(*args: Any, **kwargs: Any) -> Any:
=======
try:
    from pydantic import BaseModel, Field, field_validator

    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover — fallback path (no pydantic)
    from dataclasses import dataclass
    from dataclasses import field as dcf

    _HAS_PYDANTIC = False

    class BaseModel:
        def model_dump(self) -> dict:
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def Field(default=..., default_factory=None, **_ignored):  # type: ignore[no-redef]  # noqa: N802
        """Degrade to a ``dataclasses.field`` (unknown kwargs dropped)."""
        if default_factory is not None:
            return dcf(default_factory=default_factory)
        return dcf(default=default)

    def field_validator(*_args, **_kwargs):  # type: ignore[no-redef]
        """No-op in the fallback — validation simply does not run."""
>>>>>>> 200af37773c4a039cd7beec5094bb780d40c0f57
        return lambda f: f


class Observation(BaseModel):
    """Unified output from all binding implementations."""

<<<<<<< HEAD
    gameId: str = Field(default="moon_demo_001", description="Game identifier")  # noqa: N815 - external JSON field
    source: str = Field(default="screen_capture", description="Source name")
    frameSeq: int = Field(default=0, description="Frame sequence number")  # noqa: N815 - external JSON field
    boardObservation: list[list[str | None]] = Field(  # noqa: N815 - external JSON field
        default_factory=lambda: [[None] * 3 for _ in range(3)],
        description="Grid of cell states (None=empty, X/O for pieces)",
=======
    # camelCase 字段名是 binding 输出 API 的一部分（前端/视觉管线直接
    # 消费 obs.gameId / obs.boardObservation），保持兼容故 noqa: N815。
    gameId: str = Field(default="moon_demo_001", description="Game identifier")  # noqa: N815
    source: str = Field(default="screen_capture", description="Source name")
    frameSeq: int = Field(default=0, description="Frame sequence number")  # noqa: N815
    boardObservation: list[list[str | None]] = Field(  # noqa: N815
        default_factory=lambda: [[None] * 3 for _ in range(3)],
        description="Grid of cell states (None=empty, 'X'/'O' for pieces)",
>>>>>>> 200af37773c4a039cd7beec5094bb780d40c0f57
    )
    confidence: list[list[float]] = Field(
        default_factory=lambda: [[0.0] * 3 for _ in range(3)],
        description="Per-cell confidence scores",
    )
<<<<<<< HEAD
    observedAt: int = Field(default=0, description="Unix timestamp in ms")  # noqa: N815 - external JSON field
=======
    observedAt: int = Field(default=0, description="Unix timestamp in ms")  # noqa: N815
>>>>>>> 200af37773c4a039cd7beec5094bb780d40c0f57

    @field_validator("boardObservation")
    @classmethod
    def _check_board_structure(cls, v: list[list[str | None]]) -> list[list[str | None]]:
        if not v or not all(len(row) == len(v) for row in v):
            raise ValueError(f"boardObservation must be square, got {v}")
        return v


if not _HAS_PYDANTIC:
    # Rebuild the class as a dataclass so the Field() objects returned by
    # the fallback become real fields with proper defaults/factories.
    Observation = dataclass(Observation)  # type: ignore[assignment,misc]
