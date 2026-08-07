"""User-facing weld GUI which connects both physical RB arms at startup."""

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
    argument_names = (
        "right_robot_ip",
        "left_robot_ip",
        "execute_motion",
        "use_rviz",
        "use_h600_gui",
        "use_h600_bridge",
        "rviz_config",
        "allow_arc_output",
        "allow_nonzero_setpoints",
    )
    defaults = {
        "right_robot_ip": "192.168.1.10",
        "left_robot_ip": "192.168.1.11",
        "execute_motion": "true",
        "use_rviz": "true",
        "use_h600_gui": "false",
        "use_h600_bridge": "false",
        "rviz_config": "moveit.rviz",
        "allow_arc_output": "false",
        "allow_nonzero_setpoints": "false",
    }
    arguments = [
        DeclareLaunchArgument(name, default_value=defaults[name])
        for name in argument_names
    ]
    package_share = get_package_share_directory("construct_robot")
    stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                package_share,
                "launch",
                "weld_stack.launch.py",
            )
        ),
        launch_arguments={
            name: LaunchConfiguration(name)
            for name in (
                "right_robot_ip",
                "left_robot_ip",
                "execute_motion",
                "use_rviz",
                "rviz_config",
            )
        }.items(),
    )
    h600 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                package_share,
                "launch",
                "h600_console.launch.py",
            )
        ),
        launch_arguments={
            "start_bridge": LaunchConfiguration("use_h600_bridge"),
            "use_sudo": "true",
            "use_gui": LaunchConfiguration("use_h600_gui"),
            "allow_arc_output": LaunchConfiguration("allow_arc_output"),
            "allow_nonzero_setpoints": LaunchConfiguration(
                "allow_nonzero_setpoints"
            ),
        }.items(),
    )
    gui = Node(
        package="construct_robot",
        executable="weld_action_gui",
        output="screen",
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
        }],
    )
    # Fast DDS shared-memory lock files have repeatedly left this stack with
    # discoverable endpoints but no live topics/services. Keep every process
    # in this launch on the same UDP transport, including ros2_control, MoveIt,
    # RViz and the GUI.
    """
        export ROS_LOCALHOST_ONLY=1
        export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
        export ROS_DOMAIN_ID=0
    """
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
        + arguments
        + [h600, stack, gui]
    )
