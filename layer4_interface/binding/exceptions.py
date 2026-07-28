"""Binding layer exceptions."""

from __future__ import annotations


class BindingError(Exception):
    """Base binding error."""


class ImageLoadError(BindingError):
    """Image could not be loaded."""


class InvalidBoardError(BindingError):
    """Board layout is invalid."""


class InvalidConfidenceError(BindingError):
    """Confidence values are invalid."""


class InvalidFrameSequenceError(BindingError):
    """Frame sequence numbers are out of order."""


class MissingHistoryError(BindingError):
    """History required but not available."""


class AmbiguousObservationError(BindingError):
    """Observation is ambiguous — needs user clarification."""


class VisionModelResponseError(BindingError):
    """Vision model returned an invalid/unexpected response."""


class InvalidActionMaskError(BindingError):
    """Action mask is invalid or inconsistent."""
