from types import SimpleNamespace
from unittest.mock import Mock

from construct_robot.weld_action_gui import WeldActionGui


def test_sequence_failure_keeps_touch_output_unchanged():
    gui = object.__new__(WeldActionGui)
    gui.sequence_stop_requested = False
    gui.fake_arc_enabled = SimpleNamespace(get=lambda: False)
    gui.post = Mock()
    gui._set_sequence_status = Mock()
    gui._sequence_finished = Mock()
    gui._run_sequence_step = Mock(return_value=(False, "test failure"))
    gui._set_fastech_output_sync = Mock()
    gui._finish_weld_feedback_record = Mock()
    gui.hicomm_client = SimpleNamespace(
        inhibit_outputs=Mock(), clear_outputs=Mock(), latest_status=Mock()
    )
    gui.node = SimpleNamespace(cancel_active_motion=Mock())

    gui._sequence_worker([{"type": "sleep", "seconds": 0}], [0], True)

    gui.hicomm_client.inhibit_outputs.assert_called_once()
    gui.hicomm_client.clear_outputs.assert_called_once()
    gui.node.cancel_active_motion.assert_called_once()
    gui._set_fastech_output_sync.assert_not_called()
