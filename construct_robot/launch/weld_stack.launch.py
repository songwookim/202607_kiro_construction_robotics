"""Internal dual-REAL-RB MoveIt/ros2_control stack."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    execute_motion = LaunchConfiguration("execute_motion")
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    right_robot_ip = LaunchConfiguration("right_robot_ip")
    left_robot_ip = LaunchConfiguration("left_robot_ip")
    cb_simulation = LaunchConfiguration("cb_simulation")
    debug_cartesian_server = LaunchConfiguration("debug_cartesian_server")
    debug_cartesian_server_port = LaunchConfiguration(
        "debug_cartesian_server_port"
    )
    use_fake_left_hardware = LaunchConfiguration("use_fake_left_hardware")
    use_fake_right_hardware = LaunchConfiguration("use_fake_right_hardware")
    use_fake_head_hardware = LaunchConfiguration("use_fake_head_hardware")
    arguments = [
        DeclareLaunchArgument("execute_motion", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value="moveit.rviz"),
        DeclareLaunchArgument(
            "right_robot_ip",
            default_value="192.168.1.12",
        ),
        DeclareLaunchArgument(
            "left_robot_ip",
            default_value="192.168.1.11",
        ),
        DeclareLaunchArgument("cb_simulation", default_value="false"),
        DeclareLaunchArgument("debug_cartesian_server", default_value="false"),
        DeclareLaunchArgument(
            "debug_cartesian_server_port", default_value="5679"
        ),
        DeclareLaunchArgument("use_fake_left_hardware", default_value="false"),
        DeclareLaunchArgument("use_fake_right_hardware", default_value="false"),
        DeclareLaunchArgument("use_fake_head_hardware", default_value="false"),
    ]
    moveit_share = get_package_share_directory("construct_moveit_config")
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_share, "launch", "moveit.launch.py")
        ),
        launch_arguments={
            "use_rviz": use_rviz,
            "rviz_config": rviz_config,
            "execute_motion": execute_motion,
            "left_robot_ip": left_robot_ip,
            "right_robot_ip": right_robot_ip,
            "cb_simulation": cb_simulation,
            "use_fake_left_hardware": use_fake_left_hardware,
            "use_fake_right_hardware": use_fake_right_hardware,
            "use_fake_head_hardware": use_fake_head_hardware,
        }.items(),
    )
    server = Node(
        package="construct_robot",
        executable="cartesian_path_server",
        output="screen",
        prefix=PythonExpression([
            "'python3 -m debugpy --listen 127.0.0.1:",
            debug_cartesian_server_port,
            " --wait-for-client' if '",
            debug_cartesian_server,
            "' == 'true' else ''",
        ]),
        parameters=[{
            "use_moveit": True,
            "execute_motion": ParameterValue(
                execute_motion,
                value_type=bool,
            ),
            "planning_frame": "World",
        }],
    )
    return LaunchDescription(arguments + [moveit, server])
