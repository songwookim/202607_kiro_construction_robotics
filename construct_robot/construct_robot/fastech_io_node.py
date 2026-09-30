"""Backward-compatible import for the Fastech ROS 2 I/O node."""

from .nodes.fastech_io_node import *  # noqa: F401,F403
from .nodes.fastech_io_node import main


if __name__ == "__main__":
    main()
