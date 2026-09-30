"""Backward-compatible import for the keyboard teaching ROS 2 node."""

from .nodes.keyboard_teaching_node import *  # noqa: F401,F403
from .nodes.keyboard_teaching_node import main


if __name__ == "__main__":
    main()
