"""GUI weld-parameter table rows: current setting vs last weld."""

from construct_robot.core.weld_parameter_table import build_rows, fmt
from construct_robot.io.weld_logging import read_weld_log_sections

LOG = """WELD FEEDBACK LOG
result=completed
ended=2026-10-02 17:28:11

[commanded]
current_a=245
voltage=26.0
material=FE-SOLID
software_crater_enabled=True

[execution_conditions]
weld_tcp_speed_mm_s=5.0
weld_lead_in_mm=0.0
weld_lead_out_mm=2.0

[quality_metrics]
electrical.current_a.steady_main.average=249.76
electrical.current_a.steady_main.std=9.22
motion.actual_seam_speed.average=4.57
motion.actual_seam_speed.std=2.70

[software_crater_control]
enabled=True
actual_hold_s=0.7515
status=NOT_OBSERVED
target_current_a=74
target_voltage_v=24.5
"""


def rows_by_label(settings, sections):
    return {row[1]: row for row in build_rows(settings, sections)}


def test_log_sections_include_header_and_key_values(tmp_path):
    path = tmp_path / "latest_weld_feedback.log"
    path.write_text(LOG, encoding="utf-8")
    sections = read_weld_log_sections(path)
    assert sections["header"]["result"] == "completed"
    assert sections["commanded"]["current_a"] == "245"
    assert read_weld_log_sections(tmp_path / "missing.log") == {}


def test_rows_compare_now_last_and_measured(tmp_path):
    path = tmp_path / "latest.log"
    path.write_text(LOG, encoding="utf-8")
    sections = read_weld_log_sections(path)
    settings = {"current_a": 250, "voltage": 26.0, "material": "FE-SOLID",
                "weld_tcp_speed_mm_s": 5.0, "weld_lead_in_mm": 0.0, "weld_lead_out_mm": 2.0,
                "software_crater_enabled": True}
    rows = rows_by_label(settings, sections)
    assert rows["전류"][3:7] == ("250", "245", "249.8 ± 9.2", True)
    assert rows["전압"][3:5] == ("26.0", "26.0") and rows["전압"][6] is False
    assert rows["용접 속도"][5] == "4.6 ± 2.7"
    assert rows["Lead-in / Lead-out"][3:5] == ("0.0 / 2.0", "0.0 / 2.0")
    assert rows["소프트웨어 크레이터"][3:6] == ("ON", "ON", "0.75 s · NOT_OBSERVED")
    assert rows["결과"][5] == "completed"


def test_without_a_previous_weld_only_current_settings_show():
    rows = rows_by_label({"current_a": 245}, {})
    assert rows["전류"][3:7] == ("245", "-", "-", False)


def test_format_rules():
    assert fmt(None) == "-" and fmt("N/A") == "-"
    assert fmt(True) == "ON" and fmt("False") == "OFF"
    assert fmt("26", 1) == "26.0" and fmt("CO2 100%") == "CO2 100%"


def test_touch_pair_weave_follows_the_line_between_wall_and_floor_touches():
    import math
    from geometry_msgs.msg import Pose
    from construct_robot.core.seam_geometry import touch_pair_weave_direction

    def pose(x, y, z):
        p = Pose(); p.position.x, p.position.y, p.position.z = x, y, z; p.orientation.w = 1.0
        return p

    seam = (0.0, 0.0, 1.0)  # vertical (3G)
    # Wall at +Y side, floor at +X side of the joint; 2 mm along-seam offset ignored.
    touches = {"start_wall": pose(0.0, 0.004, 0.0), "start_floor": pose(0.004, 0.0, 0.002),
               "goal_wall": pose(0.0, 0.004, 0.05), "goal_floor": pose(0.004, 0.0, 0.05)}
    weave = touch_pair_weave_direction(touches, seam)
    assert weave[2] == 0.0 or abs(weave[2]) < 1e-12
    assert math.isclose(abs(weave[0]), math.sqrt(0.5), abs_tol=1e-9)
    assert math.isclose(abs(weave[1]), math.sqrt(0.5), abs_tol=1e-9)
    # Sign follows the reference so left/right dwell keep their meaning.
    flipped = touch_pair_weave_direction(touches, seam, orientation_reference=(-1.0, 1.0, 0.0))
    assert flipped[0] < 0.0 < flipped[1]
    assert touch_pair_weave_direction({"start_wall": touches["start_wall"]}, seam) is None


def test_real_3g_touches_give_a_weave_across_the_two_touched_points():
    from pathlib import Path
    import yaml
    from geometry_msgs.msg import Pose
    from construct_robot.core.seam_geometry import touch_pair_weave_direction
    path = (Path(__file__).resolve().parents[2] / "construct_description" / "config"
            / "right_manipulator_seam_touch_points.yaml")
    data = yaml.safe_load(path.read_text())["touches"]
    touches = {}
    for name, entry in data.items():
        p = Pose()
        p.position.x, p.position.y, p.position.z = (entry["contact_tcp"]["position_m"][k] for k in "xyz")
        touches[name] = p
    weave = touch_pair_weave_direction(touches, (0.0, 0.0, 1.0))
    assert weave is not None and abs(weave[2]) < 1e-9
