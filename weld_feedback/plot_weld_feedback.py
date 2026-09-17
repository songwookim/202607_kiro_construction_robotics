#!/home/irs/ros2_ws/.venv/bin/python
"""Create current/voltage PNG plots from weld feedback .log files.

Usage:
    ./plot_weld_feedback.py weld_feedback_20260818_152503_213.log
    ./plot_weld_feedback.py weld_feedback_*.log --no-show

Each output is saved beside its input with the same stem and a .png suffix.
When no log is given, latest_weld_feedback.log is used.
"""

import argparse
from pathlib import Path
import sys


WORKSPACE = Path.home() / "ros2_ws"
SOURCE_PACKAGE = (
    WORKSPACE
    / "src"
    / "construct_robot_ros2"
    / "construct_robot"
)
if str(SOURCE_PACKAGE) not in sys.path:
    sys.path.insert(0, str(SOURCE_PACKAGE))

from construct_robot.weld_feedback_plot import plot_weld_feedback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logs",
        nargs="*",
        type=Path,
        default=[Path(__file__).with_name("latest_weld_feedback.log")],
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save PNG files without opening graph windows",
    )
    arguments = parser.parse_args()
    for candidate in arguments.logs:
        path = candidate.expanduser()
        if not path.is_absolute():
            path = Path(__file__).parent / path
        if not path.is_file():
            parser.error(f"log does not exist: {path}")
        output = plot_weld_feedback(
            path,
            output=path.with_suffix(".png"),
            show=not arguments.no_show,
        )
        print(output)


if __name__ == "__main__":
    main()
