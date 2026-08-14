"""Shared definitions used by the construct_robot launch entry points."""

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PythonExpression,
)


@dataclass(frozen=True)
class LaunchArgumentSpec:
    default: str
    description: str


ARGUMENT_SPECS = {
    "left_robot_ip": LaunchArgumentSpec(
        "192.168.1.11", "Left RB control box IP address"
    ),
    "right_robot_ip": LaunchArgumentSpec(
        "192.168.1.12", "Right RB control box IP address"
    ),
    "execute_motion": LaunchArgumentSpec(
        "true", "Allow planned trajectories to execute on controllers"
    ),
    "use_rviz": LaunchArgumentSpec("true", "Start RViz"),
    "rviz_config": LaunchArgumentSpec(
        "moveit.rviz", "RViz configuration filename"
    ),
    "cb_simulation": LaunchArgumentSpec(
        "false", "Use RB control box simulation mode"
    ),
    "fake_sensor_commands": LaunchArgumentSpec(
        "false", "Expose fake sensor command interfaces"
    ),
    "use_fake_left_hardware": LaunchArgumentSpec(
        "false", "Use mock hardware for the left arm"
    ),
    "use_fake_right_hardware": LaunchArgumentSpec(
        "false", "Use mock hardware for the right arm"
    ),
    "use_fake_head_hardware": LaunchArgumentSpec(
        "false", "Use mock hardware for the head"
    ),
    "debug_gui": LaunchArgumentSpec(
        "false", "Wait for a Python debugger on the weld GUI"
    ),
    "debug_gui_port": LaunchArgumentSpec(
        "5678", "debugpy port for the weld GUI"
    ),
    "debug_cartesian_server": LaunchArgumentSpec(
        "false", "Wait for a Python debugger on the Cartesian server"
    ),
    "debug_cartesian_server_port": LaunchArgumentSpec(
        "5679", "debugpy port for the Cartesian server"
    ),
    "hicomm_source_ip": LaunchArgumentSpec(
        "192.168.1.2", "Local Hi-COMM network interface IP address"
    ),
    "hicomm_welder_ip": LaunchArgumentSpec(
        "192.168.1.10", "Hi-COMM welding controller IP address"
    ),
    "hicomm_port": LaunchArgumentSpec(
        "60000", "Hi-COMM welding controller TCP port"
    ),
}


def declare_arguments(
    names: Iterable[str],
    *,
    default_overrides: Optional[Mapping[str, str]] = None,
):
    """Create consistently described launch arguments in the given order."""
    overrides = default_overrides or {}
    declarations = []
    for name in names:
        spec = ARGUMENT_SPECS[name]
        declarations.append(
            DeclareLaunchArgument(
                name,
                default_value=overrides.get(name, spec.default),
                description=spec.description,
            )
        )
    return declarations


def configured_arguments(names: Iterable[str]):
    """Map launch argument names to substitutions for an included launch."""
    return {name: LaunchConfiguration(name) for name in names}


def debugpy_prefix(enabled, port):
    """Run a Python node under debugpy only when its debug flag is true."""
    python_executable = EnvironmentVariable(
        "CONSTRUCT_ROBOT_PYTHON",
        default_value="/usr/bin/python3",
    )
    return PythonExpression([
        "'",
        python_executable,
        " -m debugpy --listen 127.0.0.1:",
        port,
        " --wait-for-client' if '",
        enabled,
        "' == 'true' else ''",
    ])
