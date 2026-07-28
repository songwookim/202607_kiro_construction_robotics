import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


right_robot_ip = LaunchConfiguration("right_robot_ip")
left_robot_ip = LaunchConfiguration("left_robot_ip")
use_fake_left_hardware = LaunchConfiguration("use_fake_left_hardware")
use_fake_right_hardware = LaunchConfiguration("use_fake_right_hardware")
use_initial_left_positions = LaunchConfiguration("use_initial_left_positions")
use_initial_right_positions = LaunchConfiguration("use_initial_right_positions")
fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")
cb_simulation = LaunchConfiguration("cb_simulation")
use_rviz = LaunchConfiguration("use_rviz")

def generate_launch_description():

    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "rviz_config",
            default_value="moveit.rviz",
            description="RViz configuration file",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "left_robot_ip",
            default_value="192.168.1.11",
            description="Left RB Cobot Control Box IP address",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "right_robot_ip",
            default_value="192.168.1.10",
            description="Right RB Cobot Control Box IP address",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_fake_left_hardware",
            default_value="true",
            description="Use fake hardware for the left manipulator",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_fake_right_hardware",
            default_value="false",
            description="Use fake hardware for the right manipulator",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_initial_left_positions",
            default_value="true",
            description="Initialize fake left manipulator from initial_positions.yaml",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_initial_right_positions",
            default_value="false",
            description="Initialize fake right manipulator from initial_positions.yaml",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "fake_sensor_commands",
            default_value="false",
            description="True when use fake sensor commands",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Start RViz",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "cb_simulation",
            default_value="false",
            description="Use the RB Control Box simulation mode",
        )
    )
    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )


def launch_setup(context, *args, **kwargs):
    mappings = {
        "left_robot_ip": left_robot_ip,
        "right_robot_ip": right_robot_ip,
        "use_fake_left_hardware": use_fake_left_hardware,
        "use_fake_right_hardware": use_fake_right_hardware,
        "use_initial_left_positions": use_initial_left_positions,
        "use_initial_right_positions": use_initial_right_positions,
        "fake_sensor_commands": fake_sensor_commands,
        "cb_simulation": cb_simulation,
    }

    moveit_config = (
        # MoveItConfigsBuilder("construct_robot")
        MoveItConfigsBuilder('construct_robot_0528', package_name='construct_moveit_config')
        .robot_description(file_path="config/construct_robot_0528.urdf.xacro",
                            mappings=mappings)
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_scene_monitor(
            publish_robot_description=True, publish_robot_description_semantic=True
        )
        .planning_pipelines(
            pipelines=["ompl", "chomp", "pilz_industrial_motion_planner"]
        )
        .to_moveit_configs()
    )
    # Start the actual move_group node/action server
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
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
        get_package_share_directory("construct_robot_bringup"),
        "config",
        "controllers.yaml",
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[moveit_config.robot_description, ros2_controllers_path],
        output="both",
    )

    # Use one spawner so configure/activate service calls are serialized.
    # Separate concurrent spawners can time out while the real RB hardware
    # occupies controller_manager's service callback.
    controllers_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "right_manipulator_controller",
            "left_manipulator_controller",
            "robot_head_controller",
            "--controller-manager-timeout",
            "300",
            "--service-call-timeout",
            "60",
            "--switch-timeout",
            "60",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    nodes_to_start = [
        rviz_node,
        world_alias_tf,
        link0_alias_tf,
        robot_state_publisher,
        run_move_group_node,
        ros2_control_node,
        controllers_spawner,
    ]

    return nodes_to_start
