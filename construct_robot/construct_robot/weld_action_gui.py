"""Compatibility module for the Tkinter production GUI.

Keep the historic import path as the *same module object*, so legacy patches
and diagnostics still observe the globals used by WeldActionGui methods.
"""

import sys

from .gui import weld_action_gui as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
