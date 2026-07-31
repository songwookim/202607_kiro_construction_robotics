import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import AndSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    port = LaunchConfiguration("port")
    execute_motion = LaunchConfiguration("execute_motion")
    use_rviz = LaunchConfiguration("use_rviz")
    sync_rviz_goal = LaunchConfiguration("sync_rviz_goal_to_current")
    use_fake_left_hardware = LaunchConfiguration("use_fake_left_hardware")
    use_fake_right_hardware = LaunchConfiguration("use_fake_right_hardware")
    right_robot_ip = LaunchConfiguration("right_robot_ip")
    cb_simulation = LaunchConfiguration("cb_simulation")

    arguments = [
        DeclareLaunchArgument("port", default_value="8080"),
        DeclareLaunchArgument("execute_motion", default_value="false"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument(
            "sync_rviz_goal_to_current",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "use_fake_left_hardware",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "use_fake_right_hardware",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "right_robot_ip",
            default_value="192.168.1.10",
        ),
        DeclareLaunchArgument("cb_simulation", default_value="false"),
    ]

    moveit_share = get_package_share_directory("construct_moveit_config")
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_share, "launch", "moveit.launch.py")
        ),
        launch_arguments={
            "use_rviz": use_rviz,
            "use_fake_left_hardware": use_fake_left_hardware,
            "use_fake_right_hardware": use_fake_right_hardware,
            "use_initial_left_positions": "true",
            "use_initial_right_positions": "true",
            "right_robot_ip": right_robot_ip,
            "cb_simulation": cb_simulation,
        }.items(),
    )
    action_server = Node(
        package="construct_robot",
        executable="cartesian_path_server",
        output="screen",
        parameters=[
            {
                "use_moveit": True,
                "execute_motion": ParameterValue(
                    execute_motion,
                    value_type=bool,
                ),
                "planning_frame": "World",
            }
        ],
    )
    viewer = Node(
        package="construct_robot",
        executable="viser_viewer",
        output="screen",
        arguments=["--port", port],
    )
    rviz_goal_sync = Node(
        package="construct_robot",
        executable="rviz_goal_state_sync",
        output="screen",
        condition=IfCondition(AndSubstitution(use_rviz, sync_rviz_goal)),
    )

    return LaunchDescription(
        arguments + [moveit, action_server, viewer, rviz_goal_sync]
    )
