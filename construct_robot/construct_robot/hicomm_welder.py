"""Compatibility module for the Hi-COMM protocol adapter."""

import sys

from .io import hicomm_welder as _implementation

sys.modules[__name__] = _implementation
