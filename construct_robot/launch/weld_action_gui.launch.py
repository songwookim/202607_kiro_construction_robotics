import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    moveit_share = get_package_share_directory("construct_moveit_config")
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_share, "launch", "moveit.launch.py")),
        launch_arguments={
            "use_fake_left_hardware": "true",
            "use_fake_right_hardware": "false",
            "use_initial_left_positions": "true",
            "use_initial_right_positions": "true",
        }.items())
    server = Node(
        package="construct_robot",
        executable="cartesian_path_server",
        output="screen",
        parameters=[{
            "use_moveit": True,
            "execute_motion": True,
            "planning_frame": "World",
        }])
    gui = TimerAction(
        period=3.0,
        actions=[Node(
            package="construct_robot",
            executable="weld_action_gui",
            output="screen")])
    return LaunchDescription([moveit, server, gui])
