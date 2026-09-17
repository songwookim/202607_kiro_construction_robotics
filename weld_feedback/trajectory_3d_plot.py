#!/home/irs/ros2_ws/.venv/bin/python
"""Open an interactive 3D trajectory plot for weld feedback logs."""

import argparse
from pathlib import Path
import sys


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
SCRIPT_DIRECTORY = SCRIPT_DIRECTORY+"/bavel_"
WORKSPACE_DIRECTORY = SCRIPT_DIRECTORY.parent
SOURCE_PACKAGE = (
    WORKSPACE_DIRECTORY
    / "src"
    / "construct_robot_ros2"
    / "construct_robot"
)
if str(SOURCE_PACKAGE) not in sys.path:
    sys.path.insert(0, str(SOURCE_PACKAGE))

from construct_robot.weld_feedback_plot import (
    plot_all_weld_trajectories_3d,
    plot_weld_trajectory_3d,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logs",
        nargs="*",
        help="Weld .log files; defaults to every .log in this directory",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        default=True,
        help="Open only latest_weld_feedback.log",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Open one separate 3D figure per selected log",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        default=False,
        help="Save trajectory_3d PNG files without opening windows",
    )
    arguments = parser.parse_args(argv)
    logs = [Path(value).expanduser() for value in arguments.logs]
    if not logs:
        logs = (
            [SCRIPT_DIRECTORY / "latest_weld_feedback.log"]
            if arguments.latest
            else sorted(SCRIPT_DIRECTORY.glob("*.log"))
        )

    if not arguments.separate:
        try:
            output = plot_all_weld_trajectories_3d(
                logs,
                show=not arguments.no_show,
            )
        except (OSError, RuntimeError, ValueError) as error:
            print(f"FAILED combined plot: {error}", file=sys.stderr)
            return 1
        print(f"Combined 3D trajectory PNG: {output}")
        return 0

    failed = False
    for log_path in logs:
        if not log_path.is_absolute():
            log_path = (Path.cwd() / log_path).resolve()
        try:
            output = plot_weld_trajectory_3d(
                log_path,
                show=not arguments.no_show,
            )
        except (OSError, RuntimeError, ValueError) as error:
            failed = True
            print(f"FAILED {log_path}: {error}", file=sys.stderr)
            continue
        print(f"3D trajectory PNG: {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
