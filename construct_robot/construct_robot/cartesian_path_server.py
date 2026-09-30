"""Backward-compatible import for the Cartesian ROS 2 action node."""

from .nodes.cartesian_path_server import *  # noqa: F401,F403
from .nodes.cartesian_path_server import main


if __name__ == "__main__":
    main()
