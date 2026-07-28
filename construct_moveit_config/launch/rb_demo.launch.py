from launch import LaunchDescription
from launch.action import Action
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch
from typing import cast


def generate_launch_description():
    ros2_control_name = LaunchConfiguration('ros2_control_name')
    hardware_plugin = LaunchConfiguration('hardware_plugin')
    joint_commands_topic = LaunchConfiguration('joint_commands_topic')
    joint_states_topic = LaunchConfiguration('joint_states_topic')
    rbpodo_ip = LaunchConfiguration('rbpodo_ip')
    rbpodo_port = LaunchConfiguration('rbpodo_port')
    rbpodo_operation_mode = LaunchConfiguration('rbpodo_operation_mode')
    rbpodo_speed_bar = LaunchConfiguration('rbpodo_speed_bar')
    use_rbpodo_bridge = LaunchConfiguration('use_rbpodo_bridge')
    use_degrees_for_rbpodo = LaunchConfiguration('use_degrees_for_rbpodo')
    enable_test_commands = LaunchConfiguration('enable_test_commands')
    startup_pose_source = LaunchConfiguration('startup_pose_source')
    startup_enforce_until_reached = LaunchConfiguration('startup_enforce_until_reached')
    startup_reach_tolerance_deg = LaunchConfiguration('startup_reach_tolerance_deg')
    use_viser_gui = LaunchConfiguration('use_viser_gui')
    viser_host = LaunchConfiguration('viser_host')
    viser_port = LaunchConfiguration('viser_port')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_joint_plot = LaunchConfiguration('use_joint_plot')
    debug_bridge = LaunchConfiguration('debug_bridge')
    debug_bridge_wait = LaunchConfiguration('debug_bridge_wait')
    debug_plot = LaunchConfiguration('debug_plot')
    debug_plot_wait = LaunchConfiguration('debug_plot_wait')
    use_passive_jitter_monitor = LaunchConfiguration('use_passive_jitter_monitor')

    controlled_joint_names = [
        f'right_manipulator_joint{i}' for i in range(1, 7)
    ]

    def debugpy_prefix(enabled, wait, port):
        return PythonExpression([
            "'python3 -m debugpy --listen 127.0.0.1:",
            str(port),
            "' + (' --wait-for-client' if '",
            wait,
            "' == 'true' else '') if '",
            enabled,
            "' == 'true' else ''",
        ])

    moveit_config = (
        MoveItConfigsBuilder('construct_robot_0528', package_name='construct_moveit_0528')
        .robot_description(
            file_path='config/construct_robot_0528.urdf.xacro',
            mappings={
                'ros2_control_name': ros2_control_name,
                'hardware_plugin': hardware_plugin,
                'joint_commands_topic': joint_commands_topic,
                'joint_states_topic': joint_states_topic,
            },
        )
        .to_moveit_configs()
    )
    demo_launch = generate_demo_launch(moveit_config)

    launch_description = LaunchDescription([
        DeclareLaunchArgument('ros2_control_name', default_value='RBBridgeSystem'),
        DeclareLaunchArgument(
            'hardware_plugin',
            default_value='topic_based_ros2_control/TopicBasedSystem',
        ),
        DeclareLaunchArgument('joint_commands_topic', default_value='/rb_joint_commands'),
        DeclareLaunchArgument('joint_states_topic', default_value='/rb_joint_states'),
        DeclareLaunchArgument('rbpodo_ip', default_value='192.168.1.10'),
        DeclareLaunchArgument('rbpodo_port', default_value='5000'),
        DeclareLaunchArgument('rbpodo_operation_mode', default_value='preserve'),
        DeclareLaunchArgument('rbpodo_speed_bar', default_value='0.10'),
        DeclareLaunchArgument('use_rbpodo_bridge', default_value='true'),
        DeclareLaunchArgument('use_degrees_for_rbpodo', default_value='true'),
        DeclareLaunchArgument('enable_test_commands', default_value='false'),
        DeclareLaunchArgument('startup_pose_source', default_value='initial_positions'), # initial_positions, rviz, or rbpodo
        DeclareLaunchArgument('startup_enforce_until_reached', default_value='true'),
        DeclareLaunchArgument('startup_reach_tolerance_deg', default_value='2.0'),
        DeclareLaunchArgument('use_viser_gui', default_value='false'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('viser_host', default_value='0.0.0.0'),
        DeclareLaunchArgument('viser_port', default_value='8080'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('use_joint_plot', default_value='true'),
        DeclareLaunchArgument('debug_bridge', default_value='false'),
        DeclareLaunchArgument('debug_bridge_wait', default_value='false'),
        DeclareLaunchArgument('debug_plot', default_value='false'),
        DeclareLaunchArgument('debug_plot_wait', default_value='false'),
        DeclareLaunchArgument('use_passive_jitter_monitor', default_value='true'),
    ])

    for entity in demo_launch.entities:
        launch_description.add_action(cast(Action, entity))

    launch_description.add_action(
        Node(
            package='construct_robot',
            executable='rbpodo_bridge.py',
            name='rbpodo_bridge',
            output='screen',
            condition=IfCondition(use_rbpodo_bridge),
            prefix=debugpy_prefix(debug_bridge, debug_bridge_wait, 5678),
            parameters=[
                {
                    'command_topic': joint_commands_topic,
                    'state_topic': joint_states_topic,
                    'rbpodo_ip': rbpodo_ip,
                    'rbpodo_port': rbpodo_port,
                    'rbpodo_operation_mode': rbpodo_operation_mode,
                    'rbpodo_speed_bar': rbpodo_speed_bar,
                    'use_degrees_for_rbpodo': use_degrees_for_rbpodo,
                    # Validated RB Servo J defaults. These intentionally stay
                    # internal so normal launches need no tuning arguments.
                    'command_mode': 'servo_j',
                    'servo_command_hz': 100.0,
                    'servo_t1': 0.01,
                    'servo_t2': 0.10,
                    'servo_gain': 1.0,
                    'servo_alpha': 0.5,
                    'servo_stop_t1': 0.10,
                    'servo_stop_t2': 0.15,
                    'command_watchdog_sec': 0.25,
                    'servo_idle_stop_sec': 0.50,
                    'servo_command_buffer_sec': 0.08,
                    'command_tolerance_deg': 0.001,
                    'enable_test_commands': enable_test_commands,
                    'startup_pose_source': startup_pose_source,
                    'startup_enforce_until_reached': startup_enforce_until_reached,
                    'startup_reach_tolerance_deg': startup_reach_tolerance_deg,
                    'controlled_joint_names': controlled_joint_names,
                }
            ],
        )
    )

    launch_description.add_action(
        Node(
            package='construct_robot',
            executable='rb_joint_passive_jitter_monitor.py',
            name='rb_joint_passive_jitter_monitor',
            output='screen',
            condition=IfCondition(use_passive_jitter_monitor),
            arguments=[
                '--ip', rbpodo_ip,
                '--port', '5001',
                '--window-sec', '30.0',
                '--filter-cutoff-hz', '8.0',
                '--update-hz', '10.0',
            ],
        )
    )

    # Debug-only live monitor. It does not participate in the control path.
    launch_description.add_action(
        Node(
            package='construct_robot',
            executable='rb_joint_realtime_plot.py',
            name='rb_joint_realtime_plot',
            output='screen',
            condition=IfCondition(use_joint_plot),
            prefix=debugpy_prefix(debug_plot, debug_plot_wait, 5679),
            arguments=[
                '--command-topic', joint_commands_topic,
                '--state-topic', joint_states_topic,
                '--ang-topic', '/rb_joint_ang_states',
                '--ref-topic', '/rb_joint_ref_states',
                '--diagnostics-topic', '/rbpodo_bridge/diagnostics',
                '--update-hz', '10.0',
            ],
        )
    )

    # Viser already has matching defaults; only launch-specific wiring remains.
    launch_description.add_action(
        Node(
            package='construct_robot',
            executable='viser_pose_sync_gui.py',
            name='viser_pose_sync_gui',
            output='screen',
            condition=IfCondition(use_viser_gui),
            parameters=[
                {
                    'command_topic': joint_commands_topic,
                    'rviz_joint_topic': '/joint_states',
                    'rb_joint_topic': joint_states_topic,
                    'startup_pose_source': startup_pose_source,
                    'viser_host': viser_host,
                    'viser_port': viser_port,
                    'controlled_joint_names': controlled_joint_names,
                }
            ],
        )
    )

    # launch_description.add_action(
    #     Node(
    #         package='tf2_ros',
    #         executable='static_transform_publisher',
    #         arguments=['0', '0', '0', '0', '0', '0', 'helios_link', 'helios2_ray_frame'],
    #         parameters=[{'use_sim_time': use_sim_time}],
    #     )
    # )
    # launch_description.add_action(
    #     Node(
    #         package='tf2_ros',
    #         executable='static_transform_publisher',
    #         arguments=['-0.06', '0', '0', '0', '0', '0', 'zed2i_link', 'zed2i_left_camera_frame'],
    #         parameters=[{'use_sim_time': use_sim_time}],
    #     )
    # )
    # launch_description.add_action(
    #     Node(
    #         package='tf2_ros',
    #         executable='static_transform_publisher',
    #         arguments=['0.06', '0', '0', '0', '0', '0', 'zed2i_link', 'zed2i_right_camera_frame'],
    #         parameters=[{'use_sim_time': use_sim_time}],
    #     )
    # )

    return launch_description
