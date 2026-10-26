"""Compatibility launch name; all sensing behavior belongs to Hongik's field launch."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    source = Path(get_package_share_directory("hongik_scan_ros")) / "launch" / "hongik_moveit.launch.py"
    return LaunchDescription([IncludeLaunchDescription(PythonLaunchDescriptionSource(str(source)))])
