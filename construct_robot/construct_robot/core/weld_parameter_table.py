"""Rows of the GUI weld-parameter table: current setting vs last weld.

Each row compares the value set in the GUI now, the value the last weld was
actually commanded with, and what was measured during that weld.  Settings
and the last weld's values both use the saved feedback-log keys
(``[commanded]`` / ``[execution_conditions]``), so one row spec serves both.
"""

import math


def _number(raw):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def fmt(raw, digits=1):
    """Format a raw value: numbers to ``digits``, booleans ON/OFF, else text."""
    if raw is None or raw == "" or raw == "N/A":
        return "-"
    if isinstance(raw, bool) or str(raw).strip().lower() in ("true", "false"):
        return "ON" if str(raw).strip().lower() == "true" else "OFF"
    value = _number(raw)
    if value is None:
        return str(raw)
    if digits == 0:
        return f"{value:.0f}"
    return f"{value:.{digits}f}"


def _stat(sections, section, prefix, digits=1):
    """'avg ± std' from <prefix>.average/.std in a log section."""
    values = sections.get(section, {})
    average = _number(values.get(f"{prefix}.average"))
    if average is None:
        return "-"
    std = _number(values.get(f"{prefix}.std"))
    text = f"{average:.{digits}f}"
    return f"{text} ± {std:.{digits}f}" if std is not None else text


def _value(section, key, digits=1):
    return lambda sections: fmt(sections.get(section, {}).get(key), digits)


def _hold(section):
    def measured(sections):
        values = sections.get(section, {})
        if str(values.get("enabled", "")).lower() != "true":
            return "-"
        hold = fmt(values.get("actual_hold_s"), 2)
        status = values.get("status", "")
        return f"{hold} s · {status}" if status else f"{hold} s"
    return measured


# (group, label, unit, setting key or None, digits, measured(sections) or None)
# Setting keys are looked up in the settings dict (GUI now) and in the last
# weld's [commanded] then [execution_conditions] sections.
ROWS = (
    ("전기", "전류", "A", "current_a", 0,
     lambda s: _stat(s, "quality_metrics", "electrical.current_a.steady_main")),
    ("전기", "전압", "V", "voltage", 1,
     lambda s: _stat(s, "quality_metrics", "electrical.voltage_v.steady_main")),
    ("전기", "와이어 송급", "m/min", None, 1,
     lambda s: _stat(s, "quality_metrics", "electrical.wfs_m_min")),
    ("전기", "용접기 RX 에코 (전류 / 전압)", "A / V", None, 1,
     lambda s: "{} / {}".format(
         fmt(s.get("quality_metrics", {}).get("electrical.current_a.rx_echo"), 0),
         fmt(s.get("quality_metrics", {}).get("electrical.voltage_v.rx_echo"), 1))),
    ("전기", "재질", "", "material", 1, None),
    ("전기", "와이어 직경", "mm", "diameter_mm", 1, None),
    ("전기", "모드", "", "mode", 1, None),
    ("전기", "가스", "", "gas", 1, None),
    ("전기", "시너직", "", "synergic", 1, None),
    ("전기", "보정", "", "correction", 1, None),
    ("모션", "용접 속도", "mm/s", "weld_tcp_speed_mm_s", 1,
     lambda s: _stat(s, "quality_metrics", "motion.actual_seam_speed")),
    ("모션", "비드 길이", "mm", None, 1,
     _value("quality_metrics", "motion.actual_seam_length_mm")),
    ("모션", "끝점 오차", "mm", None, 2,
     _value("quality_metrics", "motion.final_endpoint_error_mm", 2)),
    ("모션", "Lead-in / Lead-out", "mm", ("weld_lead_in_mm", "weld_lead_out_mm"), 1, None),
    ("모션", "ARC OFF lead", "ms", "weld_arc_off_delay_ms", 0, None),
    ("위빙", "위빙 사용", "", "weld_weave_enabled", 1, None),
    ("위빙", "패턴", "", "weld_weave_pattern", 1, None),
    ("위빙", "위빙 기준 (터치 보정 시)", "", "weld_weave_reference", 1, None),
    ("위빙", "횡방향 축 (manual·터치 없음)", "", "weld_weave_axis", 1, None),
    ("위빙", "진폭 (±)", "mm", "weld_weave_amplitude_mm", 2,
     lambda s: "L {} / R {}".format(
         fmt(s.get("quality_metrics", {}).get("weave.measured_left_amplitude_avg_mm"), 2),
         fmt(s.get("quality_metrics", {}).get("weave.measured_right_amplitude_avg_mm"), 2))),
    ("위빙", "피치", "mm", "weld_weave_pitch_mm", 2,
     _value("quality_metrics", "weave.measured_pitch_avg_mm", 2)),
    ("위빙", "드웰 L / R", "s", ("weld_weave_left_dwell_s", "weld_weave_right_dwell_s"), 2,
     lambda s: "{} / {}".format(
         fmt(s.get("quality_metrics", {}).get("weave.left_dwell_measured_avg_s"), 2),
         fmt(s.get("quality_metrics", {}).get("weave.right_dwell_measured_avg_s"), 2))),
    ("핫스타트", "용접기 핫스타트", "", "hot_start_enabled", 1, None),
    ("핫스타트", "핫스타트 부스트 / hold adj", "% / -",
     ("hot_start_percent", "hot_start_hold_adjustment"), 0, None),
    ("핫스타트", "Custom 핫스타트", "", "custom_hot_start_enabled", 1,
     _hold("custom_hot_start")),
    ("핫스타트", "Custom 부스트 / Hold", "% / s",
     ("custom_hot_start_percent", "custom_hot_start_hold_s"), 2,
     lambda s: _stat(s, "custom_hot_start", "current_a", 0) + " A"
     if s.get("custom_hot_start", {}).get("current_a.average") else "-"),
    ("크레이터", "소프트웨어 크레이터", "", "software_crater_enabled", 1,
     _hold("software_crater_control")),
    ("크레이터", "전류 비율 / 전압 / Hold", "% / V / s",
     ("software_crater_ratio_percent", "software_crater_voltage_v",
      "software_crater_hold_s"), 2,
     lambda s: "{} A / {} V".format(
         fmt(s.get("software_crater_control", {}).get("target_current_a"), 0),
         fmt(s.get("software_crater_control", {}).get("target_voltage_v"), 1))
     if s.get("software_crater_control", {}) else "-"),
    ("크레이터", "패널 네이티브 크레이터 관찰", "", "expect_native_crater", 1, None),
    ("결과", "결과", "", None, 1, _value("header", "result")),
    ("결과", "용접 시각", "", None, 1, _value("header", "ended")),
    ("결과", "아크 시간 / 용접 모션 시간", "s", None, 2,
     lambda s: "{} / {}".format(
         fmt(s.get("production_metrics", {}).get("arc_on_time_s"), 2),
         fmt(s.get("production_metrics", {}).get("weld_motion_duration_s"), 2))),
    ("결과", "와이어 소모", "mm", None, 0,
     _value("production_metrics", "wire_consumable_mm", 0)),
)


def _lookup(source, key):
    for section in ("commanded", "execution_conditions"):
        if key in source.get(section, {}):
            return source[section][key]
    return None


def _setting_text(key, digits, lookup):
    if key is None:
        return ""
    keys = key if isinstance(key, tuple) else (key,)
    return " / ".join(fmt(lookup(name), digits) for name in keys)


def build_rows(settings, last_sections):
    """Return [(group, label, unit, now, last commanded, last measured, changed)].

    ``settings`` is a flat {log key: value} of the GUI's current settings;
    ``last_sections`` is read_weld_log_sections() of the last weld log ({} if
    none).  ``changed`` marks settings that differ from the last weld.
    """
    rows = []
    for group, label, unit, key, digits, measured in ROWS:
        now = _setting_text(key, digits, settings.get)
        last = _setting_text(key, digits, lambda name: _lookup(last_sections, name))
        result = measured(last_sections) if measured and last_sections else ("-" if measured else "")
        changed = bool(key) and bool(last_sections) and last not in ("", "-") and now != last
        rows.append((group, label, unit, now, last, result, changed))
    return rows
