"""MoveIt, ros2_control, and Cartesian motion server stack."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from construct_robot.launch_support import (
    configured_arguments,
    debugpy_prefix,
    declare_arguments,
)


MOVEIT_ARGUMENTS = (
    "left_robot_ip",
    "right_robot_ip",
    "execute_motion",
    "use_rviz",
    "rviz_config",
    "cb_simulation",
    "fake_sensor_commands",
    "use_fake_left_hardware",
    "use_fake_right_hardware",
    "use_fake_head_hardware",
)
DEBUG_ARGUMENTS = (
    "debug_cartesian_server",
    "debug_cartesian_server_port",
)


def generate_launch_description():
    moveit_share = get_package_share_directory("construct_moveit_config")
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_share, "launch", "moveit.launch.py")
        ),
        launch_arguments=configured_arguments(MOVEIT_ARGUMENTS).items(),
    )

    execute_motion = LaunchConfiguration("execute_motion")
    cartesian_server = Node(
        package="construct_robot",
        executable="cartesian_path_server",
        output="screen",
        prefix=debugpy_prefix(
            LaunchConfiguration("debug_cartesian_server"),
            LaunchConfiguration("debug_cartesian_server_port"),
        ),
        parameters=[{
            "use_moveit": True,
            "execute_motion": ParameterValue(
                execute_motion,
                value_type=bool,
            ),
            "planning_frame": "World",
        }],
    )

    return LaunchDescription(
        declare_arguments(MOVEIT_ARGUMENTS + DEBUG_ARGUMENTS)
        + [moveit, cartesian_server]
    )
