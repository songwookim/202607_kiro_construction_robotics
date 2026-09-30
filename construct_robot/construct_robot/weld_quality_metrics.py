"""Compatibility import for weld quality metrics, including private helpers."""

from .core import weld_quality_metrics as _implementation
from .core.weld_quality_metrics import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_implementation, name)
