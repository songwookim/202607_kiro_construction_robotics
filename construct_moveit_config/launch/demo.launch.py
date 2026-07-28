from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from launch.action import Action
from typing import cast

def generate_launch_description():
    use_rviz = LaunchConfiguration('use_rviz')
    moveit_config = (
        MoveItConfigsBuilder("construct_robot_0528", package_name="construct_moveit_0528")
        .robot_description_kinematics(file_path='config/kinematics.yaml')
        .to_moveit_configs()
    )
    demo_launch = generate_demo_launch(moveit_config)

    def _add_demo_entities(context):
        use_rviz_value = LaunchConfiguration('use_rviz').perform(context).strip().lower()
        enable_rviz = use_rviz_value in {'1', 'true', 'yes', 'on'}
        actions = []
        for entity in demo_launch.entities:
            executable_name = str(getattr(entity, 'node_executable', getattr(entity, 'executable', '')))
            if isinstance(entity, Node) and executable_name == 'rviz2' and not enable_rviz:
                continue
            actions.append(cast(Action, entity))
        return actions

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0',
                   'helios_link', 'helios2_ray_frame'],
        parameters=[{'use_sim_time': True}]
    )

    static_tf2 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['-0.06', '0', '0', '0', '0', '0',
                   'zed2i_link', 'zed2i_left_camera_frame'],
        parameters=[{'use_sim_time': True}]
    )
    static_tf3 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0.06', '0', '0', '0', '0', '0',
                   'zed2i_link', 'zed2i_right_camera_frame'],
        parameters=[{'use_sim_time': True}]
    )
    demo_launch.entities.append(static_tf)
    demo_launch.entities.append(static_tf2)  # 추가
    demo_launch.entities.append(static_tf3)  # 추가

    launch_description = LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true'),
    ])
    launch_description.add_action(OpaqueFunction(function=_add_demo_entities))
    return launch_description

