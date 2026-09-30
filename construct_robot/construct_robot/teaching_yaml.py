"""Compatibility module for teaching YAML persistence."""

import sys

from .io import teaching_yaml as _implementation

sys.modules[__name__] = _implementation
