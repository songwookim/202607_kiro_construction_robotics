import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import AndSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    h600_port = LaunchConfiguration("h600_port")
    execute_motion = LaunchConfiguration("execute_motion")
    use_rviz = LaunchConfiguration("use_rviz")
    use_gui = LaunchConfiguration("use_gui")
    sync_rviz_goal = LaunchConfiguration("sync_rviz_goal_to_current")
    use_viser = LaunchConfiguration("use_viser")
    viser_port = LaunchConfiguration("viser_port")
    use_h600_gui = LaunchConfiguration("use_h600_gui")
    right_robot_ip = LaunchConfiguration("right_robot_ip")
    cb_simulation = LaunchConfiguration("cb_simulation")
    allow_arc_output = LaunchConfiguration("allow_arc_output")
    allow_nonzero_setpoints = LaunchConfiguration(
        "allow_nonzero_setpoints"
    )
    arguments = [
        DeclareLaunchArgument("h600_port", default_value="1502"),
        DeclareLaunchArgument("execute_motion", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("use_gui", default_value="true"),
        DeclareLaunchArgument(
            "sync_rviz_goal_to_current",
            default_value="true",
        ),
        DeclareLaunchArgument("use_viser", default_value="true"),
        DeclareLaunchArgument("viser_port", default_value="8080"),
        DeclareLaunchArgument("use_h600_gui", default_value="false"),
        DeclareLaunchArgument(
            "right_robot_ip",
            default_value="192.168.1.10",
        ),
        DeclareLaunchArgument("cb_simulation", default_value="false"),
        DeclareLaunchArgument("allow_arc_output", default_value="false"),
        DeclareLaunchArgument(
            "allow_nonzero_setpoints",
            default_value="false",
        ),
    ]
    moveit_share = get_package_share_directory("construct_moveit_config")
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_share, "launch", "moveit.launch.py")),
        launch_arguments={
            "use_fake_left_hardware": "true",
            "use_fake_right_hardware": "false",
            "use_rviz": use_rviz,
            "use_initial_left_positions": "true",
            "use_initial_right_positions": "true",
            "right_robot_ip": right_robot_ip,
            "cb_simulation": cb_simulation,
        }.items())
    server = Node(
        package="construct_robot",
        executable="cartesian_path_server",
        output="screen",
        parameters=[{
            "use_moveit": True,
            "execute_motion": ParameterValue(
                execute_motion,
                value_type=bool,
            ),
            "planning_frame": "World",
            "use_h600_modbus": True,
        }])
    h600 = Node(
        package="construct_robot",
        executable="h600_modbus_bridge",
        output="screen",
        parameters=[{
            "port": ParameterValue(h600_port, value_type=int),
            "allow_arc_output": ParameterValue(
                allow_arc_output,
                value_type=bool,
            ),
            "allow_nonzero_setpoints": ParameterValue(
                allow_nonzero_setpoints,
                value_type=bool,
            ),
        }],
    )
    gui = TimerAction(
        period=3.0,
        condition=IfCondition(use_gui),
        actions=[Node(
            package="construct_robot",
            executable="weld_action_gui",
            output="screen",
            parameters=[{
                "expected_execute_motion": ParameterValue(
                    execute_motion,
                    value_type=bool,
                ),
                "expected_robot_connected": True,
                "expected_right_robot_ip": right_robot_ip,
            }],
        )],
    )
    viewer = Node(
        package="construct_robot",
        executable="viser_viewer",
        output="screen",
        condition=IfCondition(use_viser),
        arguments=["--port", viser_port],
    )
    h600_gui = Node(
        package="construct_robot",
        executable="h600_modbus_gui",
        output="screen",
        condition=IfCondition(use_h600_gui),
    )
    rviz_goal_sync = Node(
        package="construct_robot",
        executable="rviz_goal_state_sync",
        output="screen",
        condition=IfCondition(AndSubstitution(use_rviz, sync_rviz_goal)),
    )
    return LaunchDescription(
        arguments
        + [
            moveit,
            h600,
            server,
            viewer,
            h600_gui,
            rviz_goal_sync,
            gui,
        ]
    )
