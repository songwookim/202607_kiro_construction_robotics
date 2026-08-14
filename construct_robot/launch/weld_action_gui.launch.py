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
from launch.substitutions import PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    argument_names = (
        "right_robot_ip",
        "left_robot_ip",
        "execute_motion",
        "use_rviz",
        "rviz_config",
        "debug_gui",
        "debug_gui_port",
        "debug_cartesian_server",
        "debug_cartesian_server_port",
        "use_fake_left_hardware",
        "use_fake_right_hardware",
        "use_fake_head_hardware",
        "hicomm_source_ip",
        "hicomm_welder_ip",
        "hicomm_port",
    )
    defaults = {
        "right_robot_ip": "192.168.1.12",
        "left_robot_ip": "192.168.1.11",
        "execute_motion": "true",
        "use_rviz": "true",
        "rviz_config": "moveit.rviz",
        "debug_gui": "false",
        "debug_gui_port": "5678",
        "debug_cartesian_server": "false",
        "debug_cartesian_server_port": "5679",
        "use_fake_left_hardware": "false",
        "use_fake_right_hardware": "false",
        "use_fake_head_hardware": "false",
        "hicomm_source_ip": "192.168.1.2",
        "hicomm_welder_ip": "192.168.1.10",
        "hicomm_port": "60000",
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
                "debug_cartesian_server",
                "debug_cartesian_server_port",
                "use_fake_left_hardware",
                "use_fake_right_hardware",
                "use_fake_head_hardware",
            )
        }.items(),
    )
    gui = Node(
        package="construct_robot",
        executable="weld_action_gui",
        output="screen",
        prefix=PythonExpression([
            "'python3 -m debugpy --listen 127.0.0.1:",
            LaunchConfiguration("debug_gui_port"),
            " --wait-for-client' if '",
            LaunchConfiguration("debug_gui"),
            "' == 'true' else ''",
        ]),
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
        + [stack, gui]
    )
