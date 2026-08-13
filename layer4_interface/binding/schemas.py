"""Observation schema — pydantic model for Binding output."""

from __future__ import annotations

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
        return lambda f: f


class Observation(BaseModel):
    """Unified output from all binding implementations."""

    gameId: str = Field(default="moon_demo_001", description="Game identifier")  # noqa: N815 - external JSON field
    source: str = Field(default="screen_capture", description="Source name")
    frameSeq: int = Field(default=0, description="Frame sequence number")  # noqa: N815 - external JSON field
    boardObservation: list[list[str | None]] = Field(  # noqa: N815 - external JSON field
        default_factory=lambda: [[None] * 3 for _ in range(3)],
        description="Grid of cell states (None=empty, X/O for pieces)",
    )
    confidence: list[list[float]] = Field(
        default_factory=lambda: [[0.0] * 3 for _ in range(3)],
        description="Per-cell confidence scores",
    )
    observedAt: int = Field(default=0, description="Unix timestamp in ms")  # noqa: N815 - external JSON field

    @field_validator("boardObservation")
    @classmethod
    def _check_board_structure(cls, v: list[list[str | None]]) -> list[list[str | None]]:
        if not v or not all(len(row) == len(v) for row in v):
            raise ValueError(f"boardObservation must be square, got {v}")
        return v
