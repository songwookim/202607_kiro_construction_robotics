"""Scope of the KIRO 1G package (static checks, no ROS graph): what was removed stays removed after merges from the lab line, and every
import of the shipped python code is declared in its package.xml (so `rosdep install` + `colcon build` work on any Ubuntu 22.04 /
ROS 2 Humble PC).

Removed for KIRO (user 2026-10-11): everything that talks to a robot control box or a robot controller (rb_cb_client, rbpodo
status / readiness, controller switching, lab launch files and payload / tool settings), the live sensor guard, external stop-request
bridges, the keyboard e-stop and the robot-specific development tools (motion check, hand teaching, box / point calibration, batch
trials), the RViz layer, the 3G / legacy routing and the runtime motion monitor (joint-state timeout / tracking-error stops, the
ESTOPPED status: the only stop is the action cancel); the lab's weld path dry run (2026-10-13: the robot executes a run's weld paths
with no arc - a lab / development tool). A merge that brings one of them back fails here - resolve it by keeping the
deletion (KIRO_MAINTENANCE.md in the development repository).
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

STACK = Path(__file__).resolve().parents[2]
PACKAGES = {"hongik_scan_ros": STACK / "hongik_scan_ros" / "hongik_scan_ros", "scan_l0": STACK / "scan_l0" / "scan_l0",
            "scan_core": STACK / "scan_core" / "scan_core", "vc3dz_driver": STACK / "vc3dz_driver" / "vc3dz_driver"}
LAUNCH = STACK / "hongik_scan_ros" / "launch"

REMOVED_MODULES = (
    "rb_cb_client", "rbpodo_msgs", "controller_manager_msgs", "visualization_msgs",
    "scan_l0.sensor_guard", "scan_l0.teach", "scan_l0.box_calib", "scan_l0.flip_calib", "scan_l0.fill_report",
    "hongik_scan_ros.estop_bridge", "hongik_scan_ros.estop_keyboard", "hongik_scan_ros.health", "hongik_scan_ros.hardware",
    "hongik_scan_ros.motion_check", "hongik_scan_ros.teach_rough", "hongik_scan_ros.point_touch", "hongik_scan_ros.box_calib",
    "hongik_scan_ros.batch_trials", "hongik_scan_ros.assembler", "hongik_scan_ros.viz", "hongik_scan_ros.tracking_lag",
    "scan_l0.weld_path_dry_run", "hongik_scan_ros.weld_path_client", "hongik_scan_msgs.action.WeldPathDryRun",
)
REMOVED_PATHS = (
    "rb_cb_client", "scan_l0/scan_l0/sensor_guard.py", "hongik_scan_ros/launch/hongik_lab_rb20_moveit.launch.py",
    "hongik_scan_ros/launch/hongik_rb20_fake_moveit.launch.py", "hongik_scan_ros/config/hongik_lab_rb20_moveit.yaml",
    "hongik_scan_ros/config/hongik_scan.rviz", "eoat_description/config", "eoat_description/urdf/rb20_1900es_u_eoat.xacro",
    "scan_l0/scan_l0/weld_path_dry_run.py", "hongik_scan_ros/hongik_scan_ros/weld_path_client.py",
    "hongik_scan_msgs/action/WeldPathDryRun.action",
)
# robot-controller concepts of the lab RB20 (control box / pendant): KIRO is a new system - its main system sends the goal and the run
# goes through KIRO's MoveIt / JTC only, so none of these names exists in the shipped code, interfaces, launch or configuration
CONTROLLER_NAMES = ("expected_mode", "expected-mode", "cb_simulation", "cb_mode", "check_mode", "robot_status", "kill_switch", "speed_bar",
                    "hardware_mode", "controller_manager", "switch_controller")
# the runtime motion monitor (user 2026-10-11): safety = MoveIt collision checking + the envelope rules before sending; the only stop is the
# action cancel (ABORTED)
MONITOR_NAMES = ("StateMonitor", "tracking_error", "joint_states_timeout", "ESTOPPED")
# third-party imports that are optional (guarded; a feature degrades or a test skips without them)
OPTIONAL = {"h5py", "matplotlib", "xacro"}
# python module -> rosdep key / ROS package name in package.xml (the python module tf2_ros is the package tf2_ros_py on Humble;
# the package tf2_ros holds only the C++ library and static_transform_publisher)
ROSDEP = {"numpy": "python3-numpy", "scipy": "python3-scipy", "yaml": "python3-yaml", "tf2_ros": "tf2_ros_py"}


def _py_files():
    for pkg, d in PACKAGES.items():
        for f in sorted(d.rglob("*.py")):
            yield pkg, f
    for f in sorted(LAUNCH.glob("*.py")):
        yield "hongik_scan_ros", f


def _imports(path: Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name, node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.module, node.lineno))
            out += [(f"{node.module}.{a.name}", node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level > 0:             # relative: inside the package itself
            pkg = next(p for p in PACKAGES if p in path.parts) if any(p in path.parts for p in PACKAGES) else "hongik_scan_ros"
            out.append((f"{pkg}.{node.module or ''}".rstrip("."), node.lineno))
            out += [(f"{pkg}.{node.module + '.' if node.module else ''}{a.name}", node.lineno) for a in node.names]
    return out


def _declared(pkg: str) -> set:
    xml = (STACK / pkg / "package.xml").read_text(encoding="utf-8")
    return set(re.findall(r"<(?:exec_)?depend>\s*([\w-]+)\s*</(?:exec_)?depend>", xml))


def test_removed_modules_are_not_imported():
    bad = [f"{f.relative_to(STACK)}:{ln} imports {m}" for _pkg, f in _py_files() for m, ln in _imports(f)
           if any(m == r or m.startswith(r + ".") for r in REMOVED_MODULES)]
    assert not bad, "\n".join(bad)


def test_removed_files_stay_removed():
    back = [p for p in REMOVED_PATHS if (STACK / p).exists()]
    assert not back, back
    assert sorted(p.name for p in LAUNCH.glob("*.py")) == ["hongik_moveit.launch.py"]
    assert sorted(p.name for p in (STACK / "hongik_scan_ros" / "config").iterdir()) == ["hongik_kiro.yaml"]


def test_every_import_is_declared_in_package_xml():
    """Third-party and ROS imports of the shipped code (module top level and inside functions) are exec dependencies of the package
    that imports them; the standard library and the package itself need nothing."""
    std = set(sys.stdlib_module_names)
    missing = []
    for pkg, f in _py_files():
        dec = _declared(pkg)
        for m, ln in _imports(f):
            top = m.split(".")[0]
            if top in std or top == pkg or top in OPTIONAL or top == "__future__":
                continue
            key = ROSDEP.get(top, top)
            if key not in dec:
                missing.append(f"{f.relative_to(STACK)}:{ln} imports {top} - add <exec_depend>{key}</exec_depend> to {pkg}/package.xml")
    assert not missing, "\n".join(sorted(set(missing)))


def test_no_machine_specific_paths_in_the_shipped_files():
    """No absolute path of the development PC (home directory, venv, lab workspace) in the shipped packages."""
    pat = re.compile(r"/home/\w+|venv_scan|rbpodo_ros2_ws_dbcm|fine_sensing_V1|Desktop/JJW")
    hits = []
    for p in sorted(STACK.rglob("*")):
        if not p.is_file() or "/tests/" in str(p) or "/test/" in str(p) or p.suffix not in (".py", ".yaml", ".xml", ".txt", ".cfg",
                                                                                             ".toml", ".xacro", ".md", ".sh"):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if pat.search(line):
                hits.append(f"{p.relative_to(STACK)}:{i}: {line.strip()[:120]}")
    assert not hits, "\n".join(hits)



def test_no_robot_controller_concepts_or_runtime_monitor_in_the_shipped_files():
    """No controller mode / arm activation / controller status / tool, payload, collision or speed-bar setting, and no runtime motion
    monitor or ESTOPPED status: those names appear nowhere in the shipped code, action definitions, launch files or configuration
    (tests excluded)."""
    pat = re.compile("|".join(re.escape(n) for n in CONTROLLER_NAMES + MONITOR_NAMES))
    hits = []
    for p in sorted(STACK.rglob("*")):
        if not p.is_file() or "/tests/" in str(p) or "/test/" in str(p) or p.suffix not in (".py", ".yaml", ".action", ".msg", ".srv",
                                                                                             ".xacro", ".urdf", ".xml"):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if pat.search(line):
                hits.append(f"{p.relative_to(STACK)}:{i}: {line.strip()[:120]}")
    assert not hits, "\n".join(hits)
