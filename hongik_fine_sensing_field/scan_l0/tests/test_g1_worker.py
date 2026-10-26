"""scan_l0.g1_worker: the worker process loads the same numeric packages as the server (KIRO 2026-10-14). No ROS."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
def test_the_worker_loads_the_servers_numeric_packages(tmp_path):
    """KIRO 2026-10-14: NumPy 2 earlier on the host's path than the apt NumPy 1.21 the server loaded - the worker (a fresh python -c)
    must get the server's numpy / scipy: their site directories come first on its path, and a worker that still loads others exits
    with the reason instead of computing (pickled arrays do not cross NumPy major versions)."""
    from scan_l0 import g1_worker as G
    num = G.numeric_packages()
    assert "numpy" in num and num["numpy"][0] == np.__version__
    root = os.path.dirname(os.path.dirname(num["numpy"][1]))
    pp = G.worker_pythonpath(path=["/somewhere/else", root, "/x"], numeric=num).split(os.pathsep)
    assert pp[0] == root and pp.count(root) == 1 and pp[1:] == ["/somewhere/else", "/x"]
    env = dict(os.environ, PYTHONPATH=G.worker_pythonpath(numeric=num))
    code = "from scan_l0.g1_worker import _check_numeric; _check_numeric(); print('same')"
    ok = subprocess.run([sys.executable, "-c", code], env=dict(env, **{G.NUMERIC_ENV: json.dumps(num)}), capture_output=True, text=True)
    assert ok.returncode == 0 and ok.stdout.strip() == "same"
    fake = dict(num, numpy=["2.9.9", "/usr/local/lib/python3.10/dist-packages/numpy/__init__.py"])
    bad = subprocess.run([sys.executable, "-c", code], env=dict(env, **{G.NUMERIC_ENV: json.dumps(fake)}), capture_output=True, text=True)
    assert bad.returncode == 3 and "numeric packages differ from the server's" in bad.stderr and "2.9.9" in bad.stderr
