"""Complete real-robot stack for the guarded welding scenario GUI."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory("construct_robot")
    right_ip = LaunchConfiguration("right_robot_ip")
    left_ip = LaunchConfiguration("left_robot_ip")
    execute_motion = LaunchConfiguration("execute_motion")
    arguments = [
        DeclareLaunchArgument("right_robot_ip", default_value="192.168.1.12"),
        DeclareLaunchArgument("left_robot_ip", default_value="192.168.1.11"),
        DeclareLaunchArgument("execute_motion", default_value="false"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("use_sudo", default_value="true"),
        DeclareLaunchArgument("start_h600_bridge", default_value="true"),
        DeclareLaunchArgument("allow_arc_output", default_value="false"),
        DeclareLaunchArgument(
            "allow_nonzero_setpoints", default_value="false"
        ),
    ]
    stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(package_share, "launch", "weld_stack.launch.py")
        ),
        launch_arguments={
            "right_robot_ip": right_ip,
            "left_robot_ip": left_ip,
            "execute_motion": execute_motion,
            "use_rviz": LaunchConfiguration("use_rviz"),
        }.items(),
    )
    h600 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(package_share, "launch", "h600_console.launch.py")
        ),
        launch_arguments={
            "start_bridge": LaunchConfiguration("start_h600_bridge"),
            "use_sudo": LaunchConfiguration("use_sudo"),
            "use_gui": "false",
            "allow_arc_output": LaunchConfiguration("allow_arc_output"),
            "allow_nonzero_setpoints": LaunchConfiguration(
                "allow_nonzero_setpoints"
            ),
        }.items(),
    )
    gui = Node(
        package="construct_robot",
        executable="welding_scenario_gui",
        output="screen",
        parameters=[{
            "expected_execute_motion": ParameterValue(
                execute_motion, value_type=bool
            ),
            "right_robot_ip": ParameterValue(right_ip, value_type=str),
            "left_robot_ip": ParameterValue(left_ip, value_type=str),
        }],
    )
    force_udp = SetEnvironmentVariable("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4")
    local_graph = SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1")
    return LaunchDescription(
        [force_udp, local_graph] + arguments + [h600, stack, gui]
    )
