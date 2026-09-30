"""Compatibility import for seam geometry, including older private helpers."""

from .core import seam_geometry as _implementation
from .core.seam_geometry import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_implementation, name)
