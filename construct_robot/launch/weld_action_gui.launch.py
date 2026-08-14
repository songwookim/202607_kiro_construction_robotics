"""User-facing welding application with the complete motion stack."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from construct_robot.launch_support import (
    configured_arguments,
    debugpy_prefix,
    declare_arguments,
)


MOTION_ARGUMENTS = (
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
    "debug_cartesian_server",
    "debug_cartesian_server_port",
)
GUI_ARGUMENTS = (
    "debug_gui",
    "debug_gui_port",
    "hicomm_source_ip",
    "hicomm_welder_ip",
    "hicomm_port",
)


def generate_launch_description():
    package_share = get_package_share_directory("construct_robot")
    motion_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(package_share, "launch", "motion_stack.launch.py")
        ),
        launch_arguments=configured_arguments(MOTION_ARGUMENTS).items(),
    )
    gui = Node(
        package="construct_robot",
        executable="weld_action_gui",
        output="screen",
        prefix=debugpy_prefix(
            LaunchConfiguration("debug_gui"),
            LaunchConfiguration("debug_gui_port"),
        ),
        parameters=[{
            "expected_execute_motion": ParameterValue(
                LaunchConfiguration("execute_motion"),
                value_type=bool,
            ),
            "left_robot_ip": ParameterValue(
                LaunchConfiguration("left_robot_ip"),
                value_type=str,
            ),
            "right_robot_ip": ParameterValue(
                LaunchConfiguration("right_robot_ip"),
                value_type=str,
            ),
            "use_fake_head_hardware": ParameterValue(
                LaunchConfiguration("use_fake_head_hardware"),
                value_type=bool,
            ),
            "hicomm_source_ip": ParameterValue(
                LaunchConfiguration("hicomm_source_ip"),
                value_type=str,
            ),
            "hicomm_welder_ip": ParameterValue(
                LaunchConfiguration("hicomm_welder_ip"),
                value_type=str,
            ),
            "hicomm_port": ParameterValue(
                LaunchConfiguration("hicomm_port"),
                value_type=int,
            ),
        }],
    )
    # Avoid stale Fast DDS shared-memory locks by keeping the complete launch
    # on localhost UDP. These actions must precede all included nodes.
    force_udp_transport = SetEnvironmentVariable(
        "FASTDDS_BUILTIN_TRANSPORTS",
        "UDPv4",
    )
    local_ros_graph = SetEnvironmentVariable(
        "ROS_LOCALHOST_ONLY",
        "1",
    )
    return LaunchDescription(
        [force_udp_transport, local_ros_graph]
        + declare_arguments(MOTION_ARGUMENTS + GUI_ARGUMENTS)
        + [motion_stack, gui]
    )
