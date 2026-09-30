"""Compatibility module for weld feedback log persistence."""

import sys

from .io import weld_logging as _implementation

sys.modules[__name__] = _implementation
