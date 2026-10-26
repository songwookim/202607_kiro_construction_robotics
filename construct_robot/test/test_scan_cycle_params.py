"""4-PASS PARAMS: defaults = the former fixed cycle, validation, save / load, and the cycle builder using them."""

import pytest

from construct_robot.application import scan_cycle_params as scp
from construct_robot.application.scan_start_touch import START_TOUCH
from construct_robot.application.scan_weld_cycle import PASS4_WEAVE, PASS_PROFILES
from test_scan_weld_cycle import cycle, cycle_log_directory  # noqa: F401  (autouse fixture)


def test_defaults_are_the_former_fixed_cycle():
    p = scp.default_params()
    assert [(r["current_a"], r["voltage_v"], r["speed_mm_s"]) for r in p["passes"].values()] == \
        [tuple(float(v) for v in row) for row in PASS_PROFILES]
    assert [r["weave"] for r in p["passes"].values()] == [False, False, False, True]
    assert {k: p["passes"][4][k] for k in PASS4_WEAVE} == PASS4_WEAVE
    assert scp.touch_policy(p) == pytest.approx(START_TOUCH)
    assert p["cleaner"] == "cut_only"


@pytest.mark.parametrize("change, field", [
    ({"passes": {2: {"current_a": "abc"}}}, "P2 current"),
    ({"passes": {3: {"weave_pattern": "zigzag"}}}, "P3 weave pattern"),
    ({"start_touch": {"speed_mm_s": 0.6}}, "speed_mm_s"),           # the runtime refuses > 0.5 mm/s
    ({"start_touch": {"max_descent_mm": 12}}, "max_descent_mm"),
    ({"cleaner": "none"}, "cleaner"),
])
def test_out_of_range_values_name_their_field(change, field):
    with pytest.raises(ValueError, match=field):
        scp.validate_params(change)


def test_save_and_load_round_trip(tmp_path):
    p = scp.default_params()
    p["passes"][2].update(current_a="215", weave=True, weave_amplitude_mm="2.5")
    p["start_touch"]["speed_mm_s"] = "0.3"
    path = scp.save_params(p, tmp_path / "params.yaml")
    loaded = scp.load_params(path)
    assert loaded["passes"][2]["current_a"] == 215.0 and loaded["passes"][2]["weave"] is True
    assert loaded["passes"][2]["weave_amplitude_mm"] == 2.5 and loaded["start_touch"]["speed_mm_s"] == 0.3
    assert scp.load_params(tmp_path / "missing.yaml") == scp.default_params()
    assert "P2 215A/26.5V/7mm/s sine ±2.5/4mm" in scp.summary(loaded)


def test_the_cycle_takes_per_pass_recipe_weave_and_traverse_from_the_params(monkeypatch):
    import construct_robot.application.scan_weld_cycle as swc
    captured = {}
    original = swc.build_scan_weld_cycle

    def build(*args, **kwargs):
        captured["rows"] = original(*args, params=params, **kwargs)
        return captured["rows"]
    params = scp.default_params()
    params["passes"][1].update(current_a=180, voltage_v=23.0, speed_mm_s=4.5, start_touch=True)
    params["passes"][2].update(weave=True, weave_amplitude_mm=2.0, weave_pitch_mm=3.0, start_touch=False)
    params["passes"][4].update(weave=False)
    params["start_touch"].update(speed_mm_s=0.3, max_descent_mm=6.0)
    monkeypatch.setattr(swc, "build_scan_weld_cycle", build)
    import test_scan_weld_cycle
    monkeypatch.setattr(test_scan_weld_cycle, "build_scan_weld_cycle", build)
    cycle()
    welds = {r["scan_cycle_pass"]: r for r in captured["rows"] if r["type"] == "scan_weld_pass"}
    p1, p2, p4 = welds[1], welds[2], welds[4]
    assert (p1["settings"]["current_a"], p1["settings"]["voltage"], p1["weld_tcp_speed_mm_s"]) == (180, 23.0, 4.5)
    assert p1["start_touch_probe"] == pytest.approx({"speed_m_s": 0.0003, "max_descent_m": 0.006})
    assert p2["start_touch_probe"] is None
    assert p2["weld_weave_enabled"] and p2["scenario"].weave_enabled
    assert (p2["scenario"].weave_amplitude_mm, p2["scenario"].weave_pitch_mm) == (2.0, 3.0)
    assert not p4["weld_weave_enabled"] and not p4["scenario"].weave_enabled


def test_no_params_keeps_the_fixed_cycle():
    welds = [r for r in cycle() if r["type"] == "scan_weld_pass"]
    assert [(w["settings"]["current_a"], w["weld_tcp_speed_mm_s"]) for w in welds] == \
        [(int(c), s) for c, _v, s in PASS_PROFILES]
    assert [w["weld_weave_enabled"] for w in welds] == [False, False, False, True]
    assert all(w["start_touch_probe"] == START_TOUCH for w in welds)


def test_panel_reads_the_table_and_builds_with_it(monkeypatch, tmp_path):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    from tkinter import ttk
    from unittest.mock import Mock
    from construct_robot.gui.scan_cycle_params_panel import ScanCycleParamsPanel
    monkeypatch.setattr(scp, "params_path", lambda: tmp_path / "params.yaml")
    host = Mock()
    host.build_scan_weld_cycle.return_value = True
    try:
        panel = ScanCycleParamsPanel(host, ttk.Frame(root))
        panel.vars[3]["current_a"].set("205")
        panel.vars[3]["weave"].set(True)
        panel.touch_vars["speed_mm_s"].set("0.4")
        assert panel.build(rescan_after_passes=False)
        params = host.build_scan_weld_cycle.call_args.kwargs["params"]
        assert params["passes"][3]["current_a"] == 205.0 and params["passes"][3]["weave"] is True
        assert params["start_touch"]["speed_mm_s"] == 0.4
        panel.touch_vars["speed_mm_s"].set("0.9")
        assert panel.build() is False and host.error.called            # refused, nothing built
        panel.touch_vars["speed_mm_s"].set("0.4")
        panel.save()
        assert scp.load_params()["passes"][3]["current_a"] == 205.0
    finally:
        root.destroy()


@pytest.mark.parametrize("first", [1, 2, 3, 4])
def test_no_scan_can_start_at_a_later_pass(tmp_path, first):
    from construct_robot.application.scan_weld_cycle import (build_scan_weld_cycle, expand_scan_weld_cycle,
                                                             validate_scan_weld_cycle)
    import test_scan_weld_cycle as t
    folder = tmp_path / "saved"
    folder.mkdir()
    for k in range(1, 5):
        (folder / f"pass_{k}.yaml").write_text(f"pass: {k}\n")
    rows = t.cycle(rescan_after_passes=False, saved_folder=folder) if first == 1 else None
    if rows is None:
        original = t.build_scan_weld_cycle
        t.build_scan_weld_cycle = lambda *a, **kw: original(*a, start_pass=first, **kw)
        try:
            rows = t.cycle(rescan_after_passes=False, saved_folder=folder)
        finally:
            t.build_scan_weld_cycle = original
    welds = [r["scan_cycle_pass"] for r in rows if r["type"] == "scan_weld_pass"]
    cleans = sorted({r["scan_cycle_pass"] for r in rows if r["scan_cycle_phase"] == "clean"})
    assert welds == list(range(first, 5)) and cleans == list(range(first, 4))
    validate_scan_weld_cycle(rows)
    expanded = expand_scan_weld_cycle(rows)
    assert sorted({s["scan_cycle_pass"] for s in expanded if s.get("scan_cycle_phase") == "weld"}) == list(range(first, 5))
    if first > 1:                                    # dropping a later pass is still refused
        with pytest.raises(ValueError, match=f"P{first}"):
            validate_scan_weld_cycle([r for r in rows if r.get("scan_cycle_pass") != 4])


def test_a_later_start_pass_needs_no_scan():
    import test_scan_weld_cycle as t
    original = t.build_scan_weld_cycle
    t.build_scan_weld_cycle = lambda *a, **kw: original(*a, start_pass=2, **kw)
    try:
        with pytest.raises(ValueError, match="only for No Scan"):
            t.cycle(rescan_after_passes=False)
    finally:
        t.build_scan_weld_cycle = original


def test_hot_start_and_crater_per_pass_override_the_recipe_only_when_switched_on(monkeypatch):
    import construct_robot.application.scan_weld_cycle as swc
    import test_scan_weld_cycle
    params = scp.default_params()
    params["arc"]["passes"][2].update(hot_start_enabled=False, custom_hot_start_percent=35,
                                      custom_hot_start_hold_s=0.3, software_crater_enabled=True,
                                      software_crater_ratio_percent=25, software_crater_voltage_v=21,
                                      software_crater_hold_s=0.8)
    original = swc.build_scan_weld_cycle
    for use in (False, True):
        params["arc"]["use_table"] = use
        monkeypatch.setattr(test_scan_weld_cycle, "build_scan_weld_cycle",
                            lambda *a, **kw: original(*a, params=params, **kw))
        welds = {r["scan_cycle_pass"]: r for r in test_scan_weld_cycle.cycle() if r["type"] == "scan_weld_pass"}
        p2, p1 = welds[2]["settings"], welds[1]["settings"]
        if use:
            assert p2["hot_start_enabled"] is False and p2["custom_hot_start_percent"] == 35.0
            assert (p2["software_crater_ratio_percent"], p2["software_crater_voltage_v"], p2["software_crater_hold_s"]) == (25.0, 21.0, 0.8)
            assert p1["custom_hot_start_percent"] == scp.default_params()["arc"]["passes"][1]["custom_hot_start_percent"]
        else:                                     # the Build-time recipe of the cycle() helper is kept
            assert p2["custom_hot_start_percent"] == 10.0 and p2["software_crater_ratio_percent"] == 30
    assert "per-pass table" in scp.summary(params) and "crater 25%/21V/0.8s" in scp.summary(params)


@pytest.mark.parametrize("field,value", [("software_crater_ratio_percent", 55), ("hot_start_hold_adjustment", 20),
                                         ("custom_hot_start_hold_s", 0), ("software_crater_hold_s", 0)])
def test_hot_start_and_crater_ranges_are_the_recipe_ranges(field, value):
    params = scp.default_params()
    params["arc"]["passes"][3][field] = value
    with pytest.raises(ValueError, match=f"P3 {field}"):
        scp.validate_params(params)


def test_panel_copies_hot_start_and_crater_from_the_welding_tab(monkeypatch, tmp_path):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    from tkinter import ttk
    from unittest.mock import Mock
    from construct_robot.core.weld_config import DEFAULT_DIGITAL_WELD_SETTINGS
    from construct_robot.gui.scan_cycle_params_panel import ScanCycleParamsPanel
    monkeypatch.setattr(scp, "params_path", lambda: tmp_path / "params.yaml")
    host = Mock()
    host._digital_weld_settings.return_value = dict(DEFAULT_DIGITAL_WELD_SETTINGS, software_crater_enabled=True,
                                                    software_crater_ratio_percent=28.0, hot_start_percent=15.0)
    try:
        panel = ScanCycleParamsPanel(host, ttk.Frame(root))
        panel.copy_arc_from_recipe()
        panel.arc_use_table.set(True)
        panel.arc_vars[4]["software_crater_voltage_v"].set("22")
        params = panel.params()
        assert params["arc"]["use_table"] is True
        assert all(params["arc"]["passes"][k]["software_crater_ratio_percent"] == 28.0 for k in scp.PASSES)
        assert params["arc"]["passes"][1]["hot_start_percent"] == 15.0
        assert params["arc"]["passes"][4]["software_crater_voltage_v"] == 22.0
    finally:
        root.destroy()


def test_software_crater_goes_up_to_50_percent():
    from construct_robot.core.weld_config import DEFAULT_DIGITAL_WELD_SETTINGS, validate_digital_weld_settings
    ok = validate_digital_weld_settings(dict(DEFAULT_DIGITAL_WELD_SETTINGS, software_crater_enabled=True,
                                             software_crater_ratio_percent=50.0))
    assert ok["software_crater_ratio_percent"] == 50.0
    with pytest.raises(ValueError, match="20..50"):
        validate_digital_weld_settings(dict(DEFAULT_DIGITAL_WELD_SETTINGS, software_crater_ratio_percent=51.0))
    params = scp.default_params()
    params["arc"]["passes"][4]["software_crater_ratio_percent"] = 50
    assert scp.validate_params(params)["arc"]["passes"][4]["software_crater_ratio_percent"] == 50.0
