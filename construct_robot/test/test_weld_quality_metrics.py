import math

import pytest

from construct_robot.weld_quality_metrics import (
    analyze_weld_quality, event_timeline, format_quality_summary,
)


def _sample(t, current, voltage, *, state=1, wcr=True, wfs=6.0):
    return {
        "elapsed_s": t, "feedback_current_a": current,
        "feedback_voltage_v": voltage, "output_state": state,
        "wcr_detected": wcr, "wire_feed_m_min": wfs,
        "arc_ack": state == 1, "gas_ack": state == 1,
        "forward_ack": state == 1,
    }


def _tcp(t, along, x, y=0.0):
    return {
        "elapsed_s": t, "x_m": x, "y_m": y, "z_m": 0.0,
        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        "along_mm": along, "remaining_mm": 10.0-along,
        "speed_m_s": 0.01, "raw_speed_m_s": 0.01,
    }


def test_quality_integrates_arc_energy_using_seam_length_not_weave_path(tmp_path):
    samples = [
        *(_sample(t, 100, 20) for t in (0, .25, .5, .75, 1, 1.25)),
        _sample(1.5, 0, 0, state=0, wcr=False),
    ]
    tcp = [_tcp(0.0, 0, 0), _tcp(.25, 2.5, .0025, .0015),
           _tcp(.5, 5, .005, .003), _tcp(.75, 7.5, .0075, .0015),
           _tcp(1.0, 10, .010), _tcp(1.05, 10, .010),
           _tcp(1.10, 10, .010), _tcp(1.15, 10, .010)]
    frame = bytearray(55)
    frame[0] = 1
    frame[14:16] = (110).to_bytes(2, "little")
    frame[16] = 17
    document = {
        "started_unix_time": 1000.0, "samples": samples,
        "tcp_trajectory": tcp,
        "commanded": {"current_a": 100, "voltage": 20,
                      "hot_start_current_a": 110,
                      "hot_start_percent": 10,
                      "hot_start_hold_adjustment": 2},
        "rx_welding_setting_echo": {"current_a": 100,
                                     "voltage_v": 20,
                                     "hot_start_current_a": 0,
                                     "hot_start_hold_adjustment": -15},
        "tx_frames": [{"elapsed_s": .01, "unix_time": 1000.01,
                       "raw_hex": bytes(frame).hex(" ").upper()}],
        "execution_conditions": {
            "weld_tcp_speed_mm_s": 10,
            "weld_weave_enabled": False,
            "seam_start_xyz": (0.0, 0.0, 0.0),
            "seam_goal_xyz": (.010, 0.0, 0.0),
        },
        "production_metrics": {"weld_motion_start_elapsed_s": 0.0,
                               "weld_motion_complete_elapsed_s": 1.0,
                               "arc_on_time_s": 1.5,
                               "net_weld_arc_time_s": 1.0,
                               "wire_consumable_mm": 150},
        "arc_off_control": {"command_elapsed_s": 1.0},
    }
    quality = analyze_weld_quality(document)
    assert quality["production"]["arc_energy_J"] == pytest.approx(3000.0)
    # The 3 mm lateral detour changes TCP distance, not the denominator.
    assert quality["production"]["arc_energy_J_per_mm"] == pytest.approx(300.0)
    assert quality["motion"]["actual_seam_length_mm"] == pytest.approx(10.0)
    assert quality["motion"]["final_endpoint_error_mm"] == pytest.approx(0.0)
    assert quality["production"]["wire_consumed_arc_active_mm"] == pytest.approx(150.0)
    assert quality["production"]["wire_consumed_weld_motion_mm"] == pytest.approx(100.0)
    assert quality["hot_start"]["encoded_tx_current_a"] == 110
    assert quality["hot_start"]["encoded_tx_hold_raw"] == 17
    assert quality["hot_start"]["status"] == "MISMATCH"
    assert quality["crater"]["detected"] is False
    assert quality["geometry"]["ctwd_mm"] is None
    assert "WELD SUMMARY" in "\n".join(format_quality_summary({
        **document, "quality_metrics": quality,
    }))
    from construct_robot.weld_action_gui import format_weld_feedback_log
    from construct_robot.weld_feedback_plot import parse_weld_trajectory_log
    document.update({
        "result": "completed", "started": "2026-09-17 00:00:00",
        "ended": "2026-09-17 00:00:02", "elapsed_seconds": 2.0,
        "quality_metrics": quality,
        "feedback": {
            "rx_samples": len(samples), "welding_samples": 6,
            "wcr_seen": True,
            **{key: {"min": 0, "average": 1, "max": 2}
               for key in ("current_a", "voltage_v", "wire_feed_m_min")},
        },
    })
    path = tmp_path / "new.log"
    path.write_text(format_weld_feedback_log(document), encoding="utf-8")
    rendered = path.read_text()
    assert "[tx_frames]" in rendered
    assert "signed_weave_offset_mm" in rendered
    assert "ARC_ON_CMD.elapsed_s=" in rendered
    assert len(parse_weld_trajectory_log(path)["actual"]) == len(tcp)


def test_crater_requires_observed_rx_state_and_timeline_uses_send_time():
    samples = [
        _sample(.0, 100, 20), _sample(.1, 80, 18, state=2),
        _sample(.3, 0, 0, state=0, wcr=False),
        _sample(.4, 0, 0, state=0, wcr=False),
    ]
    frame_on = bytes([1] + [0]*54)
    frame_off = bytes(55)
    frames = [
        {"elapsed_s": .001, "raw_hex": frame_on.hex(" ")},
        {"elapsed_s": .2, "raw_hex": frame_off.hex(" ")},
    ]
    timeline = event_timeline(samples, started_unix_time=1000.0,
                              tx_frames=frames)
    assert timeline["ARC_ON_CMD"]["elapsed_s"] == pytest.approx(.001)
    assert timeline["ARC_OFF_CMD"]["elapsed_s"] == pytest.approx(.2)
    assert timeline["CRATER_ENTER"]["elapsed_s"] == pytest.approx(.1)
    assert timeline["CURRENT_EXTINCT"]["elapsed_s"] == pytest.approx(.3)
    assert math.isclose(timeline["ARC_ON_CMD"]["unix_time"], 1000.001)
