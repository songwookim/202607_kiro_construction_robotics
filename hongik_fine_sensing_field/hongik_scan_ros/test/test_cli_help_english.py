"""Every command-line tool that ships in the packages answers --help in English (user 2026-10-10: everything the system prints or
asks is English; code comments may stay Korean).

Tools = every console_scripts entry point of every package (setup.py / pyproject.toml under scan_stack) + the `python -m` tools
(modules with a __main__ block). Each runs in its own process with `--help` (and `<sub> --help` for every argparse subcommand);
--help must exit 0 before any connection: argparse exits while parsing, the ROS nodes print their help before rclpy starts
(hongik_scan_ros.node_help). No robot, sensor or ROS graph is touched.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STACK = Path(__file__).resolve().parents[2]
PACKAGES = ("hongik_scan_ros", "scan_l0", "scan_core", "vc3dz_driver")
HANGUL = re.compile(r"[가-힣ㄱ-ㆎ]")
# `python -m` tools (modules with a __main__ block). A new __main__ module must be added here (and answer --help safely) -
# the test fails until it is, so --help is never run on a tool that might connect on unknown arguments.
MODULE_TOOLS = {
    "hongik_scan_ros.auto_teach_client", "hongik_scan_ros.auto_teach_server", "hongik_scan_ros.post_weld_client",
    "hongik_scan_ros.precision_sensing_client",
    "hongik_scan_ros.profiler_node", "scan_l0.calib_edge_response", "scan_l0.calib_noise_priors", "scan_core.kiro_plan",
    "vc3dz_driver.cli", "vc3dz_driver.replay", "vc3dz_driver.fake_sensor",
}


def console_scripts() -> dict:
    """{name: 'module:function'} from setup.py (entry_points console_scripts) and pyproject.toml ([project.scripts])."""
    out: dict = {}
    for pkg in PACKAGES:
        sp = STACK / pkg / "setup.py"
        if sp.exists():
            for m in re.finditer(r"""["']\s*([\w.-]+)\s*=\s*([\w.]+)\s*:\s*(\w+)\s*["']""", sp.read_text(encoding="utf-8")):
                out[m.group(1)] = f"{m.group(2)}:{m.group(3)}"
        pp = STACK / pkg / "pyproject.toml"
        if pp.exists():
            txt = pp.read_text(encoding="utf-8")
            blk = re.search(r"^\[project\.scripts\]\s*$(.*?)(?=^\[|\Z)", txt, re.M | re.S)
            for m in re.finditer(r"""^\s*([\w.-]+)\s*=\s*["']([\w.]+):(\w+)["']""", blk.group(1) if blk else "", re.M):
                out[m.group(1)] = f"{m.group(2)}:{m.group(3)}"
    return out


def main_modules() -> set:
    found = set()
    for pkg in PACKAGES:
        for f in sorted((STACK / pkg / pkg).glob("*.py")):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
                    found.add(f"{pkg}.{f.stem}")
    return found


def _run(cmd: list, timeout: float = 120.0) -> tuple:
    env = dict(os.environ, ROS_LOCALHOST_ONLY="1", PYTHONDONTWRITEBYTECODE="1", COLUMNS="140")
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, stdin=subprocess.DEVNULL)
    return p.returncode, p.stdout + p.stderr


def _entry_cmd(target: str, name: str, extra: list) -> list:
    mod, fn = target.split(":")
    code = (f"import sys, importlib; sys.argv = [{name!r}] + {extra!r}; "
            f"rc = getattr(importlib.import_module({mod!r}), {fn!r})(); sys.exit(rc or 0)")
    return [sys.executable, "-c", code]


def _subcommands(help_text: str) -> list:
    """argparse subcommands: the '{a,b,c}' that starts a line (the subparser positional; option choices follow '--option')."""
    m = re.search(r"^\s+\{([\w,-]+)\}", help_text, re.M)
    return m.group(1).split(",") if m else []


def _check(label: str, cmd_for) -> list:
    """--help of one tool (+ each subcommand) → problems (empty = fine)."""
    probs = []
    rc, out = _run(cmd_for(["--help"]))
    runs = [("--help", rc, out)]
    for sub in _subcommands(out):
        rc_s, out_s = _run(cmd_for([sub, "--help"]))
        runs.append((f"{sub} --help", rc_s, out_s))
    for what, rc_, out_ in runs:
        if rc_ != 0 or "usage:" not in out_.lower():
            probs.append(f"{label} {what}: rc {rc_}, output tail {out_[-600:]!r}")
        bad = [ln for ln in out_.splitlines() if HANGUL.search(ln)]
        if bad:
            probs.append(f"{label} {what}: Hangul in {bad[:5]}")
    return probs


def test_every_command_line_tool_answers_help_in_english():
    scripts = console_scripts()
    assert set(scripts) == {"hongik_auto_teach_client", "hongik_auto_teach_server", "hongik_profiler_node", "hongik_post_weld_scan",
                            "hongik_precision_sensing", "vc3dz", "vc3dz_replay", "vc3dz_fake_sensor"}, scripts
    mains = main_modules()
    assert mains == MODULE_TOOLS, (f"__main__ modules changed: new {sorted(mains - MODULE_TOOLS)}, gone {sorted(MODULE_TOOLS - mains)} "
                                   "- check that --help is safe (no connection) and English, then update MODULE_TOOLS")
    jobs = [(f"console_scripts {n}", (lambda extra, t=t, n=n: _entry_cmd(t, n, extra))) for n, t in sorted(scripts.items())]
    jobs += [(f"python -m {m}", (lambda extra, m=m: [sys.executable, "-m", m] + extra)) for m in sorted(MODULE_TOOLS)]
    with ThreadPoolExecutor(max_workers=4) as ex:
        problems = [p for ps in ex.map(lambda j: _check(*j), jobs) for p in ps]
    assert not problems, "\n".join(problems)
