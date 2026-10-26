"""Hongik-owned field launch, site arguments and numerical environment; no hardware."""

import importlib.util
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest


def launch_module():
    from ament_index_python.packages import get_package_share_directory
    source = Path(get_package_share_directory("hongik_scan_ros")) / "launch/hongik_moveit.launch.py"
    spec = importlib.util.spec_from_file_location("hongik_launch_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_site_arguments_forwarded_without_a_runtime_patch():
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument, OpaqueFunction
    from ament_index_python.packages import get_package_share_directory

    captured = []
    with patch("launch_ros.actions.Node", side_effect=lambda **kw: captured.append(kw) or kw):
        description = launch_module().generate_launch_description()
        context = LaunchContext()
        requested = dict(platform_config=str(Path(get_package_share_directory("hongik_scan_ros")) / "config/hongik_kiro.yaml"),
                         job_file="/site/job.yaml", out_root="/site/runs", kiro_confirm_mode="each")
        context.launch_configurations.update(requested)
        context.launch_configurations["with_profiler_node"] = "true"
        for entity in description.entities:
            if isinstance(entity, DeclareLaunchArgument):
                entity.execute(context)
        next(e for e in description.entities if isinstance(e, OpaqueFunction)).execute(context)
    server = next(n for n in captured if n.get("executable") == "hongik_auto_teach_server")
    assert server.get("name") is None  # preserve the server's own name and the internal adapter's distinct name
    for name, value in requested.items():
        assert server["parameters"][0][name] == value
    assert server["additional_env"]["PYTHONNOUSERSITE"] == "1"
    assert server["additional_env"]["PYTHONPATH"] == os.pathsep.join(dict.fromkeys([p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p] + ["/usr/lib/python3/dist-packages"]))
    assert "additional_env" not in next(n for n in captured if n.get("package") == "tf2_ros")
    profiler = next(n for n in captured if n.get("executable") == "hongik_profiler_node")
    assert profiler["additional_env"] == server["additional_env"]
    assert "prefix" not in server


def test_compatibility_launch_includes_hongik_owned_launch():
    from launch.actions import IncludeLaunchDescription
    source = Path(__file__).resolve().parents[1] / "launch/hongik_precision_sensing.launch.py"
    spec = importlib.util.spec_from_file_location("compat_sensing_launch", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.generate_launch_description()
    assert len(description.entities) == 1
    assert isinstance(description.entities[0], IncludeLaunchDescription)


def test_numeric_site_preserves_ros_order_handles_duplicates_and_empty(monkeypatch):
    module = launch_module()
    monkeypatch.setenv("PYTHONPATH", "/ros:/workspace")
    assert module.numeric_env("") == {}
    assert module.numeric_env("/apt")["PYTHONPATH"] == "/ros:/workspace:/apt"
    monkeypatch.setenv("PYTHONPATH", "/ros:/apt:/workspace")
    assert module.numeric_env("/apt")["PYTHONPATH"] == "/ros:/apt:/workspace"


def test_estimator_environment_is_a_child_copy_and_appends_system_path():
    from construct_robot.nodes.hongik_scan_estimate import hongik_env

    original = {"PYTHONPATH": "/opt/ros/humble/lib/python3.10/site-packages:/site/workspace",
                "PYTHONNOUSERSITE": "0", "OTHER_SETTING": "keep"}
    child = hongik_env(original)
    assert original["PYTHONNOUSERSITE"] == "0"
    assert original["PYTHONPATH"].endswith("/site/workspace")
    assert child["PYTHONNOUSERSITE"] == "1"
    assert child["PYTHONPATH"] == original["PYTHONPATH"] + ":/usr/lib/python3/dist-packages"
    assert child["OTHER_SETTING"] == "keep"
