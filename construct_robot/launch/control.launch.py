import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    left_robot_ip = LaunchConfiguration("left_robot_ip")
    right_robot_ip = LaunchConfiguration("right_robot_ip")
    use_fake_left_hardware = LaunchConfiguration("use_fake_left_hardware")
    use_fake_right_hardware = LaunchConfiguration("use_fake_right_hardware")
    fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")
    cb_simulation = LaunchConfiguration("cb_simulation")

    description_share = get_package_share_directory("construct_description")
    moveit_share = get_package_share_directory("construct_moveit_config")
    xacro_file = os.path.join(
        description_share,
        "urdf_0528",
        "construct_robot_0528.control.urdf.xacro",
    )
    controllers_file = os.path.join(
        moveit_share,
        "config",
        "ros2_controllers.yaml",
    )
    robot_description = {
        "robot_description": Command([
            FindExecutable(name="xacro"),
            " ",
            xacro_file,
            " left_robot_ip:=", left_robot_ip,
            " right_robot_ip:=", right_robot_ip,
            " use_fake_left_hardware:=", use_fake_left_hardware,
            " use_fake_right_hardware:=", use_fake_right_hardware,
            " fake_sensor_commands:=", fake_sensor_commands,
            " cb_simulation:=", cb_simulation,
        ])
    }

    arguments = [
        DeclareLaunchArgument(
            "left_robot_ip", default_value="192.168.1.11",
            description="Left RB Control Box IP"),
        DeclareLaunchArgument(
            "right_robot_ip", default_value="192.168.1.12",
            description="Right RB Control Box IP"),
        DeclareLaunchArgument(
            "use_fake_left_hardware", default_value="true"),
        DeclareLaunchArgument(
            "use_fake_right_hardware", default_value="true"),
        DeclareLaunchArgument(
            "fake_sensor_commands", default_value="false"),
        DeclareLaunchArgument(
            "cb_simulation", default_value="false"),
    ]

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )
    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        output="screen",
        parameters=[robot_description, controllers_file],
    )
    controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            "joint_state_broadcaster",
            "right_manipulator_controller",
            "left_manipulator_controller",
            "robot_head_controller",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", "300",
            "--service-call-timeout", "60",
            "--switch-timeout", "60",
        ],
    )
    world_alias = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_alias_static_transform_publisher",
        arguments=["--frame-id", "World", "--child-frame-id", "world"],
    )
    link0_alias = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="link0_alias_static_transform_publisher",
        arguments=["--frame-id", "World", "--child-frame-id", "link0"],
    )
    return LaunchDescription(
        arguments + [
            world_alias,
            link0_alias,
            robot_state_publisher,
            control_node,
            controller_spawner,
        ])
