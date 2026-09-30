"""Compatibility module for the Fastech protocol adapter."""

import sys

from .io import fastech_ethernet as _implementation

sys.modules[__name__] = _implementation
