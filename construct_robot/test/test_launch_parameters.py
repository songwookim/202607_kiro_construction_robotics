"""Inspect launch parameter construction only; never execute launch actions."""

import ast
import importlib.util
from pathlib import Path

from launch import LaunchContext


def test_launch_parameter_names_values_and_types_are_preserved():
    path = Path(__file__).parents[1] / "launch" / "weld_action_gui.launch.py"
    spec = importlib.util.spec_from_file_location("weld_launch_parameters_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    context.launch_configurations.update({
        name: default for name, default, _description in module.LAUNCH_ARGUMENTS
    })
    context.launch_configurations.update({
        "execute_motion": "false", "fastech_ip": "192.168.0.99",
        "fastech_auto_connect": "false", "hicomm_port": "60123",
    })
    # Extract the two declaration tables without calling generate_launch_description.
    tables = []
    value_types = {"str": str, "int": int, "float": float, "bool": bool}
    for call in ast.walk(ast.parse(path.read_text())):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "_launch_parameters":
            tables.append([
                (ast.literal_eval(row.elts[0]), ast.literal_eval(row.elts[1]),
                 value_types[row.elts[2].id]) for row in call.args
            ])
    expected = [
        {"ip_address": "192.168.0.99", "board_id": 0, "poll_period_s": 0.01,
         "touch_input_channel": 4, "reconnect_period_s": 1.0, "auto_connect": False},
        {"expected_execute_motion": False, "left_robot_ip": "192.168.1.11",
         "right_robot_ip": "192.168.1.12", "use_fake_head_hardware": False,
         "keyboard_teaching_backend": "servo", "hicomm_source_ip": "192.168.1.2",
         "hicomm_welder_ip": "192.168.1.10", "hicomm_port": 60123,
         "fastech_ip": "192.168.0.99", "fastech_board_id": 0,
         "fastech_poll_period_s": 0.01,
         "wide_sensing_result_topic": "/wide_sensing/output/result"},
    ]
    assert len(tables) == len(expected)
    for table, values in zip(tables, expected):
        actual = {name: parameter.evaluate(context)
                  for name, parameter in module._launch_parameters(*table).items()}
        assert actual == values
        assert {name: type(value) for name, value in actual.items()} == {
            name: type(value) for name, value in values.items()
        }
