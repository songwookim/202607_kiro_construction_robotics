"""The numeric packages (numpy, scipy) of a node process: the record its start-up line names and the fail-fast check run before
anything else of the node is imported.

KIRO 2026-10-14: NumPy 2 in the user site and /usr/local shadowed the apt NumPy 1.21 that the apt SciPy 1.8 was built for; the server
then died at module import inside SciPy's compiled extensions ('numpy.core.multiarray failed to import', the NumPy 1.x / 2.x
banner) with a traceback that did not say what to fix. require_numeric_stack() turns exactly that state - SciPy's compiled
extensions cannot load against the loaded NumPy - into one English line with the versions, the files and the fix. A NumPy 2 with a
SciPy built for it passes: the check is the loading, not a version rule. This module imports nothing numeric at module level.
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Callable, Optional

__all__ = ["CHECK_MODULE", "numeric_record", "numeric_record_text", "numeric_problem", "require_numeric_stack"]

CHECK_MODULE = "scipy.spatial.transform"      # the first SciPy module the nodes import (ros_util, kiro_io): compiled, uses NumPy's C API


def numeric_record() -> dict:
    """{name: {'version', 'file'}} of numpy and scipy as loaded in this process (absent = not imported)."""
    out = {}
    for name in ("numpy", "scipy"):
        m = sys.modules.get(name)
        if m is not None and getattr(m, "__file__", None):
            out[name] = {"version": str(getattr(m, "__version__", "?")), "file": os.path.abspath(m.__file__)}
    return out


def numeric_record_text(rec: Optional[dict] = None) -> str:
    """'numpy 1.21.5 (/usr/lib/python3/dist-packages/numpy/__init__.py), scipy 1.8.0 (...)'."""
    rec = numeric_record() if rec is None else rec
    return ", ".join(f"{n} {d['version']} ({d['file']})" for n, d in rec.items()) or "numpy / scipy not loaded"


def numeric_problem(import_module: Callable = importlib.import_module) -> Optional[str]:
    """Import numpy, then CHECK_MODULE (SciPy's compiled extensions against the loaded NumPy) → None, or why it failed
    ('<ExceptionType>: <message>'; NumPy 2 under a SciPy built for NumPy 1 raises ImportError / AttributeError here)."""
    for mod in ("numpy", CHECK_MODULE):
        try:
            import_module(mod)
        except Exception as e:  # noqa: BLE001 - ImportError, AttributeError (_ARRAY_API not found), ...
            return f"import {mod}: {type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ''}".rstrip(": ")
    return None


def require_numeric_stack(node: str, import_module: Callable = importlib.import_module) -> dict:
    """The fail-fast check of a node's start (before its numeric imports) → numeric_record(); SystemExit with one English line (the
    loaded versions and files, the failure, the fix; exit status 1, no traceback) when SciPy cannot load against the loaded NumPy."""
    why = numeric_problem(import_module)
    if why is None:
        return numeric_record()
    raise SystemExit(f"{node}: SciPy cannot load against the loaded NumPy ({numeric_record_text()}; python {sys.executable}): {why}. "
                     "Fix: run the node with the NumPy its SciPy was built for - the apt pair numpy 1.21 / scipy 1.8 (INSTALL.md "
                     "section 6): remove the other NumPy from ~/.local and /usr/local or put /usr/lib/python3/dist-packages ahead of "
                     "them on this node's PYTHONPATH, with PYTHONNOUSERSITE=1; check with: python3 -c \"import numpy, "
                     f"{CHECK_MODULE}; print(numpy.__version__, numpy.__file__)\"")
