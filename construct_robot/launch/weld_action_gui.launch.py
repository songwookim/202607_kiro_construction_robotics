"""User-facing welding application with the complete motion stack."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

LAUNCH_ARGUMENTS = (
    ("left_robot_ip", "192.168.1.11", "Left RB control box IP address"),
    ("right_robot_ip", "192.168.1.12", "Right RB control box IP address"),
    ("execute_motion", "true", "Allow controller trajectory execution"),
    ("use_rviz", "true", "Start RViz"),
    ("rviz_config", "moveit.rviz", "RViz configuration filename"),
    ("cb_simulation", "false", "Use RB control box simulation mode"),
    ("fake_sensor_commands", "false", "Expose fake sensor commands"),
    ("use_fake_left_hardware", "false", "Use mock left-arm hardware"),
    ("use_fake_right_hardware", "false", "Use mock right-arm hardware"),
    ("use_fake_head_hardware", "false", "Use mock head hardware"),
    ("debug_gui", "false", "Wait for a debugger on the weld GUI"),
    ("debug_gui_port", "5678", "debugpy port for the weld GUI"),
    (
        "debug_cartesian_server",
        "false",
        "Wait for a debugger on the Cartesian server",
    ),
    (
        "debug_cartesian_server_port",
        "5679",
        "debugpy port for the Cartesian server",
    ),
    ("hicomm_source_ip", "192.168.1.2", "Local Hi-COMM interface IP"),
    ("hicomm_welder_ip", "192.168.1.10", "Hi-COMM controller IP"),
    ("hicomm_port", "60000", "Hi-COMM controller TCP port"),
    ("fastech_ip", "192.168.0.3", "Fastech Ezi-IO IP address"),
    ("fastech_board_id", "0", "Fastech Ezi-IO board ID"),
    ("fastech_poll_period_s", "0.01", "Fastech I/O polling period"),
    ("fastech_reconnect_period_s", "1.0", "Fastech reconnect period"),
    ("fastech_auto_connect", "true", "Connect Fastech at startup"),
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


def _debugpy_prefix(enabled_argument, port_argument):
    python_executable = EnvironmentVariable(
        "CONSTRUCT_ROBOT_PYTHON",
        default_value="/usr/bin/python3",
    )
    return PythonExpression([
        "'",
        python_executable,
        " -m debugpy --listen 127.0.0.1:",
        LaunchConfiguration(port_argument),
        " --wait-for-client' if '",
        LaunchConfiguration(enabled_argument),
        "' == 'true' else ''",
    ])


def generate_launch_description():
    declarations = [
        DeclareLaunchArgument(
            name,
            default_value=default,
            description=description,
        )
        for name, default, description in LAUNCH_ARGUMENTS
    ]

    moveit_share = get_package_share_directory("construct_moveit_config")
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_share, "launch", "moveit.launch.py")
        ),
        launch_arguments={
            name: LaunchConfiguration(name) for name in MOVEIT_ARGUMENTS
        }.items(),
    )
    execute_motion = LaunchConfiguration("execute_motion")
    cartesian_server = Node(
        package="construct_robot",
        executable="cartesian_path_server",
        output="screen",
        prefix=_debugpy_prefix(
            "debug_cartesian_server",
            "debug_cartesian_server_port",
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
    fastech_io = Node(
        package="construct_robot",
        executable="fastech_io_node",
        output="screen",
        parameters=[{
            "ip_address": ParameterValue(
                LaunchConfiguration("fastech_ip"),
                value_type=str,
            ),
            "board_id": ParameterValue(
                LaunchConfiguration("fastech_board_id"),
                value_type=int,
            ),
            "poll_period_s": ParameterValue(
                LaunchConfiguration("fastech_poll_period_s"),
                value_type=float,
            ),
            "reconnect_period_s": ParameterValue(
                LaunchConfiguration("fastech_reconnect_period_s"),
                value_type=float,
            ),
            "auto_connect": ParameterValue(
                LaunchConfiguration("fastech_auto_connect"),
                value_type=bool,
            ),
        }],
    )
    gui = Node(
        package="construct_robot",
        executable="weld_action_gui",
        output="screen",
        prefix=_debugpy_prefix("debug_gui", "debug_gui_port"),
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
            "fastech_ip": ParameterValue(
                LaunchConfiguration("fastech_ip"),
                value_type=str,
            ),
            "fastech_board_id": ParameterValue(
                LaunchConfiguration("fastech_board_id"),
                value_type=int,
            ),
            "fastech_poll_period_s": ParameterValue(
                LaunchConfiguration("fastech_poll_period_s"),
                value_type=float,
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
        + declarations
        + [moveit, cartesian_server, fastech_io, gui]
    )
