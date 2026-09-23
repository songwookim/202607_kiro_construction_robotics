from types import SimpleNamespace

import pytest
import yaml
from geometry_msgs.msg import Pose

from construct_robot.task_teaching_panel import (
    TaskTeachingPanel, atomic_yaml, decode, encode, validate_task_group,
)


def test_pose_task_roundtrip(tmp_path):
    pose = Pose()
    pose.position.z = 0.42
    pose.orientation.w = 1.0
    steps = [{"type": "motion", "planning_group": "left_manipulator", "points": [pose]}]
    path = tmp_path / "task.yaml"
    atomic_yaml(path, {"steps": encode(steps)})
    restored = decode(yaml.safe_load(path.read_text())["steps"])
    assert restored[0]["points"][0] == pose
    atomic_yaml(path, {"steps": []})
    assert yaml.safe_load(path.read_text()) == {"steps": []}


@pytest.mark.parametrize("step", [
    {"type": "motion", "planning_group": "right_manipulator"},
    {"type": "digital_weld"}, {"type": "digital_output"},
    {"type": "head_motion"}, {"type": "planned_trajectory"},
])
def test_left_task_rejects_other_hardware(step):
    with pytest.raises(ValueError):
        validate_task_group([step], "left_manipulator")


def test_unsafe_serialization_rejected():
    with pytest.raises(ValueError):
        encode(float("nan"))
    with pytest.raises(ValueError):
        decode({"ros_type": "Unknown", "fields": {}})
    with pytest.raises(ValueError):
        TaskTeachingPanel.safe_name("../other")


def test_continuous_path_preserves_order_and_arm(tmp_path):
    panel = object.__new__(TaskTeachingPanel)
    panel.speed = SimpleNamespace(get=lambda: "5")
    panel.category = SimpleNamespace(get=lambda: "Left · Spray path")
    panel.order = SimpleNamespace(get=lambda *args: ("a", "b", "c"))
    panel.base = lambda: tmp_path
    poses = []
    for value in (0.1, 0.2, 0.3):
        pose = Pose()
        pose.position.z = value
        pose.orientation.w = 1.0
        poses.append(pose)
    panel.load_pose = lambda path: ("left_manipulator", [f"left_manipulator_joint{i}" for i in range(1, 7)],
                                    [0.0] * 6, poses["abc".index(path.stem)])
    result = []
    panel.replace_builder = result.extend
    panel.build_path()
    assert [s["type"] for s in result] == ["named_pose", "motion"]
    assert result[0]["use_joint_planning"] is True
    assert result[0]["parallel_slot"] != result[1]["parallel_slot"]
    assert result[1]["points"] == poses
    assert result[1]["tcp_speed_m_s"] == 0.005
    validate_task_group(result, "left_manipulator")


def test_cleaner_failure_turns_off_only_owned_output():
    from unittest.mock import Mock
    from construct_robot.weld_action_gui import WeldActionGui
    gui = object.__new__(WeldActionGui)
    gui.sequence_stop_requested = False
    gui.fake_arc_enabled = SimpleNamespace(get=lambda: False)
    gui.post = Mock()
    gui.error = Mock()
    gui._set_sequence_status = Mock()
    gui._sequence_finished = Mock()
    gui._run_sequence_step = Mock(return_value=(False, "test failure"))
    gui._set_fastech_output_sync = Mock(return_value=(True, "OFF"))
    gui._finish_weld_feedback_record = Mock()
    gui.hicomm_client = SimpleNamespace(inhibit_outputs=Mock(), clear_outputs=Mock(), latest_status=Mock())
    gui.node = SimpleNamespace(cancel_active_motion=Mock())
    gui._sequence_worker([{"type": "digital_output", "port": 7, "value": True,
                           "task_cleaner_output": True}], [0], True)
    gui._set_fastech_output_sync.assert_called_once_with(7, False)
