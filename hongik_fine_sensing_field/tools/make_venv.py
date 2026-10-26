#!/usr/bin/env python3
"""Create the Python venv described by tools/python_env.yaml (see the comments there).

usage: python3 tools/make_venv.py [--env tools/python_env.yaml] [--profile system|pinned|lab] --dest DIR
                                  [--pip-from SITE_PACKAGES_DIR] [--get-pip] [--force]

Steps: (1) create the venv (system site packages per the YAML); (2) only if the profile installs packages: make sure the venv has pip
(ensurepip if the system has it; else the pip found in the system Python or in --pip-from; --get-pip downloads get-pip.py from
bootstrap.pypa.io only when asked) and install the profile's pinned packages; (3) make this tree's pure-Python packages importable
(link: pth = a .pth file with the source folders, no build tools; link: pip = pip install --no-deps -e); (4) check the requirements
(Python, numpy < 2, ...) with the venv's python and write <dest>/hongik_venv_report.txt.
The ROS nodes keep using the system Python (their entry scripts' shebang); this venv is for tools, tests and `python3 -m ...`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
TREE = HERE.parent


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def has_module(py, mod) -> bool:
    return subprocess.run([str(py), "-c", f"import {mod}"], capture_output=True).returncode == 0


def ensure_pip(vpy: Path, pip_from: str | None, get_pip: bool) -> None:
    if has_module(vpy, "pip"):
        return
    sources = []
    if pip_from:
        sources.append(pip_from)
    r = subprocess.run(["/usr/bin/python3", "-c", "import pip, os; print(os.path.dirname(os.path.dirname(pip.__file__)))"],
                       capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        sources.append(r.stdout.strip())
    for src in sources:
        env = dict(os.environ, PYTHONPATH=src)
        if subprocess.run([str(vpy), "-m", "pip", "install", "-q", "--ignore-installed", "pip"], env=env).returncode == 0 \
                and has_module(vpy, "pip"):
            return
    if get_pip:
        with tempfile.TemporaryDirectory() as td:
            gp = Path(td) / "get-pip.py"
            urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", gp)
            run([vpy, gp, "-q"])
        if has_module(vpy, "pip"):
            return
    sys.exit("make_venv: the venv has no pip and none could be bootstrapped. Install python3-venv + python3-pip (apt), or pass "
             "--pip-from <site-packages dir that contains pip>, or --get-pip (downloads get-pip.py).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=str(HERE / "python_env.yaml"))
    ap.add_argument("--profile", default="system")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--pip-from", default=None)
    ap.add_argument("--get-pip", action="store_true")
    ap.add_argument("--force", action="store_true", help="remove an existing --dest first")
    a = ap.parse_args(argv)
    spec = yaml.safe_load(open(a.env))
    if a.profile not in spec["profiles"]:
        sys.exit(f"make_venv: unknown profile {a.profile!r} (have: {', '.join(spec['profiles'])})")
    prof = spec["profiles"][a.profile]
    dest = Path(os.path.expanduser(a.dest)).resolve()
    if dest.exists():
        if not a.force:
            sys.exit(f"make_venv: {dest} exists (use --force to replace it)")
        shutil.rmtree(dest)
    py = spec.get("python", "/usr/bin/python3")
    args = [py, "-m", "venv"] + (["--system-site-packages"] if spec.get("system_site_packages", True) else [])
    with_pip = bool(prof.get("pip")) or spec.get("link") == "pip"
    if subprocess.run(args + [str(dest)], capture_output=True).returncode != 0:      # no ensurepip on this system
        if dest.exists():
            shutil.rmtree(dest)
        run(args + ["--without-pip", str(dest)])
    vpy = dest / "bin" / "python"
    if with_pip:
        ensure_pip(vpy, a.pip_from, a.get_pip)
        if prof.get("pip"):
            run([vpy, "-m", "pip", "install", "-q"] + list(prof["pip"]))
    pkgs = [TREE / p for p in spec.get("packages", [])]
    if spec.get("link", "pth") == "pip":
        run([vpy, "-m", "pip", "install", "-q", "--no-deps", "--no-build-isolation"] + sum([["-e", str(p)] for p in pkgs], []))
    else:
        site = subprocess.run([str(vpy), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                              capture_output=True, text=True, check=True).stdout.strip()
        Path(site).mkdir(parents=True, exist_ok=True)
        (Path(site) / "hongik_fine_sensing_src.pth").write_text("".join(f"{p}\n" for p in pkgs))
        print(f"linked {len(pkgs)} source package(s) via {Path(site) / 'hongik_fine_sensing_src.pth'}")
    # requirements check with the venv's python
    req = spec.get("requires", {})
    chk = r'''
import json, sys
from importlib import import_module
def ver(m):
    mod = import_module(m)
    return getattr(mod, "__version__", "?"), getattr(mod, "__file__", "?")
out = {"python": [sys.version.split()[0], sys.executable]}
for m in ("numpy", "scipy", "yaml", "scan_core", "scan_l0", "vc3dz_driver"):
    try:
        out[m] = list(ver(m))
    except Exception as e:
        out[m] = ["MISSING", repr(e)]
print(json.dumps(out))
'''
    res = json.loads(subprocess.run([str(vpy), "-c", chk], capture_output=True, text=True, check=True).stdout)
    from packaging.specifiers import SpecifierSet   # noqa: E402 (setuptools vendors packaging on Ubuntu 22.04 python3)
    bad = []
    for name, rule in req.items():
        v = res.get(name, ["MISSING"])[0]
        if v in ("MISSING", "?") or v not in SpecifierSet(rule):
            bad.append(f"{name} {v} not {rule}")
    for m in ("scan_core", "scan_l0", "vc3dz_driver"):
        if res.get(m, ["MISSING"])[0] == "MISSING":
            bad.append(f"{m} not importable: {res[m][1]}")
    lines = [f"profile {a.profile}, env {a.env}"] + [f"{k:13s} {v[0]:10s} {v[1]}" for k, v in res.items()]
    lines += [f"PROBLEM: {b}" for b in bad] or ["requirements: OK"]
    (dest / "hongik_venv_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nactivate:  source {dest}/bin/activate   (the ROS nodes still use /usr/bin/python3)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
