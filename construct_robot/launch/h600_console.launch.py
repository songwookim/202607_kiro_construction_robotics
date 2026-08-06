import os
import shlex
import subprocess

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def sudo_preflight(_context):
    """Authenticate before any GUI can hide or interleave the sudo prompt."""
    try:
        subprocess.run(["sudo", "-v"], check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "sudo authentication failed; H600 TCP/502 bridge was not started"
        ) from error


def generate_launch_description():
    start_bridge = LaunchConfiguration("start_bridge")
    use_gui = LaunchConfiguration("use_gui")
    use_sudo = LaunchConfiguration("use_sudo")
    host = LaunchConfiguration("host")
    allow_arc = LaunchConfiguration("allow_arc_output")
    allow_nonzero = LaunchConfiguration("allow_nonzero_setpoints")
    # sudo intentionally removes Python/loader paths. Restore only the ROS
    # runtime variables after privilege elevation so the installed entry point
    # and generated construct_msgs interfaces remain importable.
    sudo_environment_names = (
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "AMENT_PREFIX_PATH",
        "CMAKE_PREFIX_PATH",
        "ROS_DOMAIN_ID",
        "ROS_LOCALHOST_ONLY",
        "RMW_IMPLEMENTATION",
        "CYCLONEDDS_URI",
        "FASTRTPS_DEFAULT_PROFILES_FILE",
    )
    sudo_assignments = [
        f"{name}={shlex.quote(os.environ[name])}"
        for name in sudo_environment_names
        if name in os.environ
    ]
    # A root bridge and user GUI cannot share Fast DDS SHM files reliably.
    # Force both participants onto loopback/network UDP transport.
    dds_environment = {"FASTDDS_BUILTIN_TRANSPORTS": "UDPv4"}
    sudo_assignments.append("FASTDDS_BUILTIN_TRANSPORTS=UDPv4")
    sudo_prefix = " ".join(
        ["sudo", "-n", "-E", "/usr/bin/env"] + sudo_assignments
    )
    privileged_condition = IfCondition(PythonExpression([
        "'", start_bridge, "' == 'true' and '",
        use_sudo, "' == 'true'",
    ]))
    sudo_auth = OpaqueFunction(
        function=sudo_preflight,
        condition=privileged_condition,
    )
    arguments = [
        DeclareLaunchArgument("start_bridge", default_value="true"),
        DeclareLaunchArgument("use_gui", default_value="true"),
        DeclareLaunchArgument(
            "use_sudo",
            default_value="true",
            description=(
                "Run only the bridge through sudo so it can bind TCP/502"
            ),
        ),
        DeclareLaunchArgument("host", default_value="0.0.0.0"),
        DeclareLaunchArgument("allow_arc_output", default_value="false"),
        DeclareLaunchArgument(
            "allow_nonzero_setpoints",
            default_value="false",
        ),
    ]
    privileged_bridge = Node(
        package="construct_robot",
        executable="h600_modbus_bridge",
        output="screen",
        prefix=sudo_prefix,
        additional_env=dds_environment,
        condition=privileged_condition,
        parameters=[{
            "host": host,
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
    regular_bridge = Node(
        package="construct_robot",
        executable="h600_modbus_bridge",
        output="screen",
        additional_env=dds_environment,
        condition=IfCondition(PythonExpression([
            "'", start_bridge, "' == 'true' and '",
            use_sudo, "' != 'true'",
        ])),
        parameters=[{
            "host": host,
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
        additional_env=dds_environment,
        condition=IfCondition(use_gui),
    )
    return LaunchDescription(
        arguments + [sudo_auth, privileged_bridge, regular_bridge, gui]
    )
