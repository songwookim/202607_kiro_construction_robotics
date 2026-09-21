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


def test_software_crater_requires_feedback_plateau_not_native_state2():
    samples = [_sample(t, 66, 25, state=1) for t in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)]
    samples.append(_sample(1.6, 0, 0, state=0, wcr=False))
    document = {
        "started_unix_time": 1000.0,
        "samples": samples,
        "tcp_trajectory": [],
        "tx_frames": [],
        "commanded": {"current_a": 220, "voltage": 26.0,
                      "software_crater_enabled": True, "software_crater_hold_s": 0.5,
                      "expect_native_crater": False},
        "execution_conditions": {},
        "arc_off_control": {"command_elapsed_s": 1.55},
        "software_crater_control": {
            "main_current_a": 220, "main_voltage_v": 26.0,
            "target_current_a": 66, "target_voltage_v": 25.0,
            "ratio_percent": 30, "requested_hold_s": 0.5,
            "hold_start_elapsed_s": 1.0, "hold_end_elapsed_s": 1.5,
            "actual_hold_s": 0.5, "tx_status": "SENT", "main_restored": True,
        },
    }
    quality = analyze_weld_quality(document)
    assert quality["software_crater"]["status"] == "OBSERVED"
    assert quality["software_crater"]["actual_current"]["average"] == 66
    assert quality["crater"]["detected"] is False
    document["samples"] = [_sample(t, 180, 25, state=1) for t in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)]
    assert analyze_weld_quality(document)["software_crater"]["status"] == "NOT_OBSERVED"


def test_custom_hot_start_timeline_is_between_arc_and_weld_motion():
    samples = [_sample(0.10, 100, 20), _sample(0.30, 100, 20)]
    events = event_timeline(
        samples, started_unix_time=1000.0,
        custom_hot_start={"hold_start_elapsed_s": 0.20,
                          "hold_end_elapsed_s": 0.35},
        motion_start=0.38,
    )
    assert events["ARC_RECOGNIZED"]["elapsed_s"] == pytest.approx(0.10)
    assert events["WCR_ON"]["elapsed_s"] == pytest.approx(0.10)
    assert events["CUSTOM_HOT_START_BEGIN"]["elapsed_s"] == pytest.approx(0.20)
    assert events["CUSTOM_HOT_START_END"]["elapsed_s"] == pytest.approx(0.35)
    assert events["WELD_MOTION_START"]["elapsed_s"] == pytest.approx(0.38)


def test_steady_main_excludes_custom_hold_and_motion_transient():
    document = {
        "samples": [
            _sample(.2, 240, 28), _sample(.4, 240, 28),
            _sample(.6, 240, 28), _sample(.8, 240, 28),
            _sample(1.0, 190, 24), _sample(1.2, 205, 25),
            _sample(1.4, 200, 25), _sample(1.6, 200, 25),
            _sample(1.8, 66, 25),
        ],
        "commanded": {"current_a": 200, "voltage": 25},
        "custom_hot_start": {
            "enabled": True, "requested_hold_s": .5,
            "hold_start_elapsed_s": .3, "hold_end_elapsed_s": .8,
            "actual_hold_s": .5, "max_tcp_drift_mm": .054,
            "status": "COMPLETED",
        },
        "production_metrics": {
            "weld_motion_start_elapsed_s": 1.0,
            "weld_motion_complete_elapsed_s": 1.7,
        },
        "software_crater_control": {"command_elapsed_s": 1.75},
        "arc_off_control": {"command_elapsed_s": 1.9},
    }
    quality = analyze_weld_quality(document)
    assert quality["electrical"]["steady_window_start_s"] == pytest.approx(1.3)
    assert quality["electrical"]["current_a"]["steady_main"]["average"] == 200
    custom = quality["custom_hot_start"]
    assert custom["sample_count"] == 3
    assert custom["current_a"]["average"] == 240
    assert custom["voltage_v"]["average"] == 28
    assert "Current      : avg/min/max 240.00" in "\n".join(format_quality_summary({
        **document, "quality_metrics": quality,
    }))


def test_disabled_native_hot_start_has_not_applicable_metrics():
    document = {
        "samples": [_sample(.2, 200, 25)],
        "commanded": {
            "current_a": 200, "voltage": 25,
            "hot_start_enabled": False,
            "hot_start_current_a": 0, "hot_start_percent": 5,
        },
    }
    quality = analyze_weld_quality(document)
    hot = quality["hot_start"]
    assert hot["status"] == "DISABLED"
    assert hot["feedback_effect"] == "NOT_APPLICABLE"
    assert hot["requested_delta_a"] is None
    assert hot["startup_plateau_current_avg_a"] is None
    assert hot["detected_duration_s"] is None
    summary = "\n".join(format_quality_summary({**document, "quality_metrics": quality}))
    assert "Requested    : NOT_REQUESTED" in summary
    assert "TX Current   : 0 A" in summary
    assert "RX Echo      : N/A" in summary


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
    assert quality["hot_start"]["tx_status"] == "SENT"
    assert quality["hot_start"]["rx_status"] == "NOT_ECHOED"
    assert quality["hot_start"]["feedback_status"] == "NOT_OBSERVED"
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
    assert timeline["CRATER_ENTER"]["elapsed_s"] is None
    assert timeline["CURRENT_EXTINCT"]["elapsed_s"] == pytest.approx(.3)
    assert math.isclose(timeline["ARC_ON_CMD"]["unix_time"], 1000.001)
    samples.insert(2, _sample(.21, 80, 18, state=1))
    samples.insert(3, _sample(.25, 75, 17, state=2))
    timeline = event_timeline(samples, started_unix_time=1000.0,
                              tx_frames=frames,
                              arc_off_control={"sequence_clear_elapsed_s": .4})
    assert timeline["CRATER_ENTER"]["elapsed_s"] == pytest.approx(.25)
    assert timeline["CRATER_EXIT"]["elapsed_s"] == pytest.approx(.3)
    assert timeline["SEQUENCE_CLEAR"]["elapsed_s"] == pytest.approx(.4)


def test_weave_peaks_and_dwell_use_actual_offset_not_requested_amplitude():
    points = []
    def add(along, offset, speed=0.005):
        t = len(points)*.02
        row = _tcp(t, along, along*.001, offset*.001)
        row["raw_speed_m_s"] = speed
        points.append(row)
    add(0, 0)
    for start in (0, 4):
        add(start+.5, .5)
        add(start+1, 1)
        for _ in range(5):
            add(start+1, 1, 0.012)
        add(start+2, 0)
        add(start+3, -1)
        for _ in range(5):
            add(start+3, -1, 0.012)
        add(start+4, 0)
    planned = [(0, 0), (1, 1), (2, 0), (3, -1),
               (4, 0), (5, 1), (6, 0), (7, -1), (8, 0)]
    quality = analyze_weld_quality({
        "tcp_trajectory": points, "samples": [],
        "commanded": {}, "production_metrics": {
            "weld_motion_start_elapsed_s": 0.0,
            "weld_motion_complete_elapsed_s": points[-1]["elapsed_s"],
        },
        "execution_conditions": {
            "weld_weave_enabled": True, "weld_weave_pattern": "sine",
            "weld_weave_axis": "world_y", "weld_weave_amplitude_mm": 2.0,
            "weld_weave_cycles": 2, "weld_weave_actual_pitch_mm": 4.0,
            "weld_weave_left_dwell_s": .10, "weld_weave_right_dwell_s": .10,
            "seam_start_xyz": (0.0, 0.0, 0.0),
            "seam_goal_xyz": (.008, 0.0, 0.0),
            "planned_weave_waypoints_xyz": [(x*.001, y*.001, 0.0)
                                             for x, y in planned],
        },
    })
    weave = quality["weave"]
    assert weave["measured_cycle_count"] == 2
    assert weave["measured_left_amplitude_avg_mm"] == pytest.approx(1.0)
    assert weave["measured_right_amplitude_avg_mm"] == pytest.approx(1.0)
    assert weave["measured_full_width_avg_mm"] == pytest.approx(2.0)
    assert weave["left_dwell_measured_avg_s"] == pytest.approx(.10)
    assert weave["right_dwell_measured_avg_s"] == pytest.approx(.10)
    assert weave["valid_dwell_peaks"] == {"left": 2, "right": 2}
    assert weave["dwell_detection_method"] == "peak_offset_plateau"
    assert weave["dwell_measurement_confidence"] == "HIGH"
    assert all(detail["sample_count"] == 6 for detail in weave["dwell_peak_details"])
    assert weave["low_speed_samples_per_peak"] == {"left": 0.0, "right": 0.0}
    assert weave["tcp_sample_count"] == len(points)


def test_dwell_plateau_ignores_noisy_world_tcp_speed_and_requires_four_samples():
    from construct_robot.weld_quality_metrics import _contiguous_dwell

    rows = []
    for index, (offset, along) in enumerate((
        (0.4, 0.6), (0.75, 0.9), (1.01, 1.00), (0.98, 1.02),
        (1.00, 1.01), (1.02, 1.03), (0.99, 1.02), (0.6, 1.4),
    )):
        row = {"elapsed_s": index * 0.02, "signed_weave_offset_mm": offset,
               "along_mm": along, "raw_speed_m_s": 0.012}
        rows.append(row)
    detail = _contiguous_dwell(rows, rows[4])
    assert detail["valid"]
    assert detail["sample_count"] == 5
    assert detail["dwell_duration_s"] == pytest.approx(0.08)
    assert detail["dwell_start_time"] == pytest.approx(0.04)
    assert detail["dwell_end_time"] == pytest.approx(0.12)
    assert detail["offset_range_mm"] == pytest.approx(0.04)
    assert detail["seam_progress_range_mm"] == pytest.approx(0.03)
    sparse = _contiguous_dwell(rows[2:5], rows[4])
    assert sparse["sample_count"] == 3
    assert sparse["dwell_duration_s"] is None


def test_teaching_snapshot_uses_gui_provenance_without_ui_attribute():
    from geometry_msgs.msg import Pose
    from construct_robot.weld_action_gui import WeldActionGui
    gui = WeldActionGui.__new__(WeldActionGui)
    pose = Pose()
    pose.orientation.w = 1.0
    gui.taught_robot_poses = {
        "robot_start": ("right_manipulator",
                        tuple(f"right_manipulator_joint{i}" for i in range(1, 7)),
                        (0.0,)*6, pose),
    }
    gui.teaching_capture_provenance = {
        "robot_start": {"capture_source": "measured_joint_fk"},
    }
    snapshot = gui._teaching_snapshot_document()
    assert snapshot["robot_start"]["capture_provenance"]["capture_source"] == "measured_joint_fk"


def test_pre_off_crater_state_continuing_after_off_is_not_new_native_crater():
    samples = [_sample(.0, 100, 20, state=1),
               _sample(.1, 80, 18, state=2),
               _sample(.25, 75, 17, state=2),
               _sample(.3, 0, 0, state=0, wcr=False)]
    frames = [{"elapsed_s": .01, "raw_hex": bytes([1]+[0]*54).hex(" ")},
              {"elapsed_s": .2, "raw_hex": bytes(55).hex(" ")}]
    timeline = event_timeline(samples, tx_frames=frames)
    assert timeline["CRATER_ENTER"]["elapsed_s"] is None
    assert timeline["CRATER_EXIT"]["elapsed_s"] is None
