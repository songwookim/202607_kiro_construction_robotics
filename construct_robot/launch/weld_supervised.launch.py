from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    initial_connected = LaunchConfiguration("initial_connected")
    right_robot_ip = LaunchConfiguration("right_robot_ip")
    execute_motion = LaunchConfiguration("execute_motion")
    use_rviz = LaunchConfiguration("use_rviz")
    use_viser = LaunchConfiguration("use_viser")
    use_h600_gui = LaunchConfiguration("use_h600_gui")
    h600_port = LaunchConfiguration("h600_port")
    arguments = [
        DeclareLaunchArgument(
            "initial_connected",
            default_value="false",
        ),
        DeclareLaunchArgument(
            "right_robot_ip",
            default_value="192.168.1.10",
        ),
        DeclareLaunchArgument("execute_motion", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("use_viser", default_value="true"),
        DeclareLaunchArgument("use_h600_gui", default_value="false"),
        DeclareLaunchArgument("h600_port", default_value="1502"),
    ]
    supervisor = Node(
        package="construct_robot",
        executable="weld_stack_supervisor",
        output="screen",
        parameters=[{
            "initial_connected": ParameterValue(
                initial_connected,
                value_type=bool,
            ),
            "right_robot_ip": right_robot_ip,
            "execute_motion": ParameterValue(
                execute_motion,
                value_type=bool,
            ),
            "use_rviz": ParameterValue(use_rviz, value_type=bool),
            "use_viser": ParameterValue(use_viser, value_type=bool),
            "use_h600_gui": ParameterValue(
                use_h600_gui,
                value_type=bool,
            ),
            "h600_port": ParameterValue(h600_port, value_type=int),
        }],
    )
    gui = Node(
        package="construct_robot",
        executable="weld_action_gui",
        output="screen",
        parameters=[{
            "expected_execute_motion": ParameterValue(
                execute_motion,
                value_type=bool,
            ),
            "expected_robot_connected": ParameterValue(
                initial_connected,
                value_type=bool,
            ),
            "expected_right_robot_ip": right_robot_ip,
        }],
    )
    return LaunchDescription(arguments + [supervisor, gui])
