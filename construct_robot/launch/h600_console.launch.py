from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    start_bridge = LaunchConfiguration("start_bridge")
    port = LaunchConfiguration("port")
    allow_arc = LaunchConfiguration("allow_arc_output")
    allow_nonzero = LaunchConfiguration("allow_nonzero_setpoints")
    arguments = [
        DeclareLaunchArgument("start_bridge", default_value="true"),
        DeclareLaunchArgument("port", default_value="502"),
        DeclareLaunchArgument("allow_arc_output", default_value="false"),
        DeclareLaunchArgument(
            "allow_nonzero_setpoints",
            default_value="false",
        ),
    ]
    bridge = Node(
        package="construct_robot",
        executable="h600_modbus_bridge",
        output="screen",
        condition=IfCondition(start_bridge),
        parameters=[{
            "port": ParameterValue(port, value_type=int),
            "allow_arc_output": ParameterValue(
                allow_arc,
                value_type=bool,
            ),
            "allow_nonzero_setpoints": ParameterValue(
                allow_nonzero,
                value_type=bool,
            ),
        }],
    )
    gui = Node(
        package="construct_robot",
        executable="h600_modbus_gui",
        output="screen",
    )
    return LaunchDescription(arguments + [bridge, gui])
