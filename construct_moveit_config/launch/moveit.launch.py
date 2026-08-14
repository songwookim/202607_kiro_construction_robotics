import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


right_robot_ip = LaunchConfiguration("right_robot_ip")
left_robot_ip = LaunchConfiguration("left_robot_ip")
fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")
use_fake_left_hardware = LaunchConfiguration("use_fake_left_hardware")
use_fake_right_hardware = LaunchConfiguration("use_fake_right_hardware")
use_fake_head_hardware = LaunchConfiguration("use_fake_head_hardware")
cb_simulation = LaunchConfiguration("cb_simulation")
use_rviz = LaunchConfiguration("use_rviz")
execute_motion = LaunchConfiguration("execute_motion")


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "rviz_config",
            default_value="moveit.rviz",
            description="RViz configuration file",
        ),
        DeclareLaunchArgument(
            "left_robot_ip",
            default_value="192.168.1.11",
            description="Left RB Cobot Control Box IP address",
        ),
        DeclareLaunchArgument(
            "right_robot_ip",
            default_value="192.168.1.12",
            description="Right RB Cobot Control Box IP address",
        ),
        DeclareLaunchArgument(
            "fake_sensor_commands",
            default_value="false",
            description="True when use fake sensor commands",
        ),
        DeclareLaunchArgument("use_fake_left_hardware", default_value="false"),
        DeclareLaunchArgument(
            "use_fake_right_hardware",
            default_value="false",
        ),
        DeclareLaunchArgument("use_fake_head_hardware", default_value="false"),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Start RViz",
        ),
        DeclareLaunchArgument(
            "execute_motion",
            default_value="true",
            description="Allow MoveIt trajectory execution on controllers",
        ),
        DeclareLaunchArgument(
            "cb_simulation",
            default_value="false",
            description="Use the RB Control Box simulation mode",
        ),
    ]
    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )


def launch_setup(_context):
    execution_enabled = execute_motion.perform(_context).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    mappings = {
        "left_robot_ip": left_robot_ip,
        "right_robot_ip": right_robot_ip,
        "fake_sensor_commands": fake_sensor_commands,
        "cb_simulation": cb_simulation,
        "use_fake_left_hardware": use_fake_left_hardware,
        "use_fake_right_hardware": use_fake_right_hardware,
        "use_fake_head_hardware": use_fake_head_hardware,
    }

    moveit_builder = (
        MoveItConfigsBuilder(
            "construct_robot_0528",
            package_name="construct_moveit_config",
        )
        .robot_description(
            file_path="config/construct_robot_0528.urdf.xacro",
            mappings=mappings,
        )
        .robot_description_semantic(
            file_path="config/construct_robot_0528.srdf"
        )
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .planning_pipelines(
            pipelines=(
                ["ompl", "chomp", "pilz_industrial_motion_planner"]
                if execution_enabled
                else ["ompl"]
            )
        )
    )
    if execution_enabled:
        moveit_builder = moveit_builder.trajectory_execution(
            file_path="config/moveit_controllers.yaml"
        )
    else:
        moveit_builder = moveit_builder.trajectory_execution(
            file_path="config/plan_only_controllers.yaml",
            moveit_manage_controllers=False,
        )
    moveit_config = moveit_builder.to_moveit_configs()
    # Start the actual move_group node/action server
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                "allow_trajectory_execution": ParameterValue(
                    execute_motion,
                    value_type=bool,
                )
            },
        ],
    )

    rviz_base = LaunchConfiguration("rviz_config")
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("construct_moveit_config"), "config", rviz_base]
    )

    # RViz
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        condition=IfCondition(use_rviz),
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    # Compatibility aliases for planning-scene publishers which still use the
    # lowercase legacy frame names. The URDF root/planning frame is "World".
    world_alias_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_alias_static_transform_publisher",
        output="log",
        arguments=["--frame-id", "World", "--child-frame-id", "world"],
    )
    link0_alias_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="link0_alias_static_transform_publisher",
        output="log",
        arguments=["--frame-id", "World", "--child-frame-id", "link0"],
    )

    # Publish TF
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    # ros2_control: left/right RB hardware plus a MoveIt-only fake head.
    ros2_controllers_path = os.path.join(
        get_package_share_directory("construct_moveit_config"),
        "config",
        "ros2_controllers.yaml",
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[moveit_config.robot_description, ros2_controllers_path],
        output="both",
    )

    # This package's controller spawner applies one configurable timeout to
    # load, configure, and activate calls. Humble's stock executable does not
    # apply --service-call-timeout consistently to all three operations.
    controllers = ["joint_state_broadcaster"]
    if execution_enabled:
        controllers.extend(
            [
                "right_manipulator_controller",
                "left_manipulator_controller",
                "robot_head_controller",
            ]
        )

    controllers_spawner = Node(
        package="construct_robot",
        executable="controller_spawner",
        arguments=[
            *controllers,
            "--controller-manager",
            "/controller_manager",
            "--response-timeout",
            "90",
        ],
    )

    start_move_group_after_controllers = RegisterEventHandler(
        OnProcessExit(
            target_action=controllers_spawner,
            on_exit=[run_move_group_node],
        )
    )

    nodes_to_start = [
        rviz_node,
        world_alias_tf,
        link0_alias_tf,
        robot_state_publisher,
        ros2_control_node,
        start_move_group_after_controllers,
        controllers_spawner,
    ]

    return nodes_to_start
