#!/usr/bin/env python3
"""Write colcon.meta in a ROS 2 workspace so that ament_cmake packages whose package.xml has no <export><build_type> are built as
ros.ament_cmake (e.g. KIRO's welding_interfaces). Without the export, colcon identifies such a package as ros.catkin and the workspace
setup does not register it in the ament index ('ros2 interface show' -> unknown package).

Generic rule: every package.xml under <workspace>/src (not under build/install/log, not COLCON_IGNORE'd) that declares no build type
while its CMakeLists.txt calls ament_package() gets a colcon.meta 'paths' entry {"type": "ros.ament_cmake"}. The package's own files
are not changed. Existing colcon.meta content is kept (only the entries of this rule are added or updated). Run it from anywhere:

  python3 src/<this package tree>/tools/colcon_meta_ament_cmake.py [--workspace <ws>] [--dry-run]
  colcon build            # in <ws>; colcon reads ./colcon.meta by default
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

SKIP_DIRS = {"build", "install", "log", ".git"}


def _build_type(package_xml: Path):
    txt = package_xml.read_text(encoding="utf-8", errors="replace")
    txt = re.sub(r"<!--.*?-->", "", txt, flags=re.S)
    m = re.search(r"<export>.*?<build_type[^>]*>\s*([\w.]+)\s*</build_type>.*?</export>", txt, flags=re.S)
    name = re.search(r"<name>\s*([\w-]+)\s*</name>", txt)
    return (m.group(1) if m else None), (name.group(1) if name else package_xml.parent.name)


def _calls_ament_package(cmakelists: Path) -> bool:
    if not cmakelists.exists():
        return False
    txt = re.sub(r"#[^\n]*", "", cmakelists.read_text(encoding="utf-8", errors="replace"))
    return re.search(r"\bament_package\s*\(", txt) is not None


def find_packages(src: Path) -> list:
    """[(name, directory)] of the packages the rule applies to."""
    out = []
    for root, dirs, files in os.walk(src, followlinks=True):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        here = Path(root)
        if "COLCON_IGNORE" in files or "AMENT_IGNORE" in files or "CATKIN_IGNORE" in files:
            dirs[:] = []
            continue
        if "package.xml" in files:
            build_type, name = _build_type(here / "package.xml")
            if build_type is None and _calls_ament_package(here / "CMakeLists.txt"):
                out.append((name, here))
            dirs[:] = []                                    # a package does not contain other packages
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default=".", help="workspace root (contains src/); default: the current directory")
    ap.add_argument("--dry-run", action="store_true", help="print what would be written, change nothing")
    a = ap.parse_args(argv)
    ws = Path(a.workspace).expanduser().resolve()
    src = ws / "src"
    if not src.is_dir():
        print(f"no src/ directory in {ws} - give --workspace <workspace root>", file=sys.stderr)
        return 2
    meta_path = ws / "colcon.meta"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8") or "{}")
        except ValueError as e:
            print(f"{meta_path} is not JSON ({e}) - not changed; merge the entries below by hand", file=sys.stderr)
            meta = None
    pkgs = find_packages(src)
    entries = {os.path.relpath(d, ws): {"type": "ros.ament_cmake", "name": n} for n, d in pkgs}
    for rel, e in entries.items():
        print(f"{e['name']}: {rel} -> type ros.ament_cmake (package.xml has no <export><build_type>, CMakeLists.txt calls ament_package())")
    if not entries:
        print("no package needs a build type override")
        return 0
    if meta is None or a.dry_run:
        print(json.dumps({"paths": entries}, indent=2))
        return 0 if a.dry_run else 1
    paths = dict(meta.get("paths") or {})
    paths.update(entries)
    meta["paths"] = paths
    tmp = meta_path.with_suffix(".meta.tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    tmp.replace(meta_path)
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
