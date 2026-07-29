import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    port = LaunchConfiguration("port")
    execute_motion = LaunchConfiguration("execute_motion")
    use_fake_left_hardware = LaunchConfiguration("use_fake_left_hardware")
    use_fake_right_hardware = LaunchConfiguration("use_fake_right_hardware")

    arguments = [
        DeclareLaunchArgument("port", default_value="8080"),
        DeclareLaunchArgument("execute_motion", default_value="false"),
        DeclareLaunchArgument(
            "use_fake_left_hardware",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "use_fake_right_hardware",
            default_value="true",
        ),
    ]

    moveit_share = get_package_share_directory("construct_moveit_config")
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_share, "launch", "moveit.launch.py")
        ),
        launch_arguments={
            "use_rviz": "true",
            "use_fake_left_hardware": use_fake_left_hardware,
            "use_fake_right_hardware": use_fake_right_hardware,
            "use_initial_left_positions": "true",
            "use_initial_right_positions": "false",
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

    return LaunchDescription(arguments + [moveit, action_server, viewer])
