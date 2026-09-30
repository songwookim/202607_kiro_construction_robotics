"""Backward-compatible import for the ROS 2 controller spawner."""

from .nodes.controller_spawner import *  # noqa: F401,F403
from .nodes.controller_spawner import main


if __name__ == "__main__":
    main()
