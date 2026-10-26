"""Post-weld scan action and CLI (user 2026-10-13) without a ROS graph: the CLI arguments (hongik_post_weld_scan), the goal they
build, the server's goal → inputs mapping (refusals before anything is sent), and the server's execute path on a stub - refusal
while another run holds the slot, a refused input run, a dry run (no profiler release, nothing sent) and a scan result."""
from __future__ import annotations

import os
import sys
import threading
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
STACK = os.path.dirname(PKG)
for p in (PKG, os.path.join(STACK, "scan_l0"), os.path.join(STACK, "scan_l0", "tests"), os.path.join(STACK, "scan_core"),
          os.path.join(STACK, "vc3dz_driver")):
    if p not in sys.path:
        sys.path.insert(0, p)

_act = pytest.importorskip("hongik_scan_msgs.action")
if not hasattr(_act, "PostWeldScan"):
    pytest.skip("hongik_scan_msgs without PostWeldScan (source the g1-postweld overlay)", allow_module_level=True)
PostWeldScan = _act.PostWeldScan

from hongik_scan_ros import post_weld_client as cl  # noqa: E402
from test_post_weld_units import make_run  # noqa: E402  (scan_l0/tests: a minimal finished-run folder)


# ---- CLI ---------------------------------------------------------------------------------------------------------------------
def test_cli_arguments_and_goal(tmp_path):
    ap = cl.build_parser()
    a = ap.parse_args([])
    assert (a.run, a.measure_speed, a.confirm, a.dry_run) == ("", None, "auto", False) and not hasattr(a, "expected_mode")
    g = cl.goal_from_args(a)
    assert (g.run_dir, g.measure_speed_mps, g.confirm_mode, g.dry_run, g.after_pass) == ("", 0.0, "auto", False, 0)
    assert not hasattr(g, "expected_mode")                                     # no robot-controller concepts in this package
    a = ap.parse_args(["--run", "k30_1_a", "--measure-speed", "30", "--confirm", "each", "--dry-run"])
    g = cl.goal_from_args(a, cwd=tmp_path)
    assert g.run_dir == "k30_1_a" and g.measure_speed_mps == pytest.approx(0.030) and g.confirm_mode == "each" and g.dry_run
    assert cl.goal_from_args(ap.parse_args(["--measure-speed", "15"])).measure_speed_mps == pytest.approx(0.015)
    # any speed [mm/s] (2026-10-13): the calibrated range is the server's check (the run-time speed check of the 1G core); the client
    # refuses only what is not a speed > 0 - before anything is sent
    assert [cl.goal_from_args(ap.parse_args(["--measure-speed", v])).measure_speed_mps for v in ("20", "22.5", "45", "60", "70")] \
        == pytest.approx([0.020, 0.0225, 0.045, 0.060, 0.070])
    for bad in ("0", "-5", "nan"):
        with pytest.raises(ValueError, match="--measure-speed must be a speed > 0"):
            cl.goal_from_args(ap.parse_args(["--measure-speed", bad]))
        out = []
        assert cl.main(["--measure-speed", bad], say=out.append) == 2
        assert out == [f"REFUSED: --measure-speed must be a speed > 0 mm/s, got {float(bad)!r}"], out
    assert cl.goal_from_args(ap.parse_args(["--after-pass", "2"])).after_pass == 2                  # the pass just completed
    # an existing local folder goes as its absolute path (the server runs on this PC); a name stays a name under the server's out_root
    (tmp_path / "runs" / "k15_2_b").mkdir(parents=True)
    g = cl.goal_from_args(ap.parse_args(["--run", "runs/k15_2_b"]), cwd=tmp_path)
    assert g.run_dir == str((tmp_path / "runs" / "k15_2_b").resolve())
    for bad in (["--measure-speed", "fast"], ["--confirm", "none"], ["--confirm", "large"],
                ["--expected-mode", "real"], ["--after-pass", "0"], ["--after-pass", "-1"], ["--after-pass", "cap"]):
        with pytest.raises(SystemExit):
            ap.parse_args(bad)
    line = cl.goal_line(cl.goal_from_args(ap.parse_args(["--dry-run"])))
    assert "(the latest on the server)" in line and "the run's measurement speed" in line and "DRY RUN (nothing sent)" in line
    assert "final (after every pass)" in line and "after pass 3" in cl.goal_line(cl.goal_from_args(ap.parse_args(["--after-pass", "3"])))
    res = SimpleNamespace(status="OK", message="post-weld scan OK: run k30_1_a, ...", out_dir="/x/k30_1_a/post_weld_scan_1",
                          run_dir="/x/k30_1_a")
    assert cl.result_lines(res) == ["RESULT: OK - post-weld scan OK: run k30_1_a, ...", "  input run: /x/k30_1_a",
                                    "  output: /x/k30_1_a/post_weld_scan_1 (raw.npz, run_log.json)"]
    assert "robot was commanded to stop" in cl.result_lines(SimpleNamespace(status="ABORTED", message="m", out_dir="", run_dir=""))[1]
    assert cl.OK_STATUSES == ("OK", "DRY_RUN") and cl.ACTION == "/hongik_fine_sensing/post_weld_scan"


# ---- the server's goal → inputs --------------------------------------------------------------------------------------------
def _goal(**kw):
    g = PostWeldScan.Goal()
    for k, v in kw.items():
        setattr(g, k, v)
    return g


def test_goal_inputs_and_refusals_before_anything_is_sent(tmp_path):
    from hongik_scan_ros.auto_teach_server import post_weld_inputs
    from scan_l0.post_weld import InputRunError
    root = tmp_path / "out_root"
    root.mkdir()
    d = make_run(root, "k30_1_a")
    kw = dict(out_root=str(root), confirm_default="each")
    inp = post_weld_inputs(_goal(), **kw)
    assert str(inp["run_dir"]) == d and inp["measure_speed_mps"] is None and inp["confirm"] == "each" and "expected_mode" not in inp
    inp = post_weld_inputs(_goal(run_dir="k30_1_a", measure_speed_mps=0.015, confirm_mode="auto", dry_run=True), **kw)
    assert inp["measure_speed_mps"] == pytest.approx(0.015) and inp["confirm"] == "auto" and inp["dry_run"] is True
    assert inp["after_pass"] == 0
    assert post_weld_inputs(_goal(after_pass=3), **kw)["after_pass"] == 3
    for g, match in ((_goal(measure_speed_mps=-0.01), "measure_speed_mps"), (_goal(confirm_mode="maybe"), "confirm_mode"),
                     (_goal(run_dir="missing"), "does not exist")):
        with pytest.raises(ValueError, match=match):
            post_weld_inputs(g, **kw)
    (tmp_path / "empty").mkdir()
    with pytest.raises(InputRunError, match="no finished run"):
        post_weld_inputs(_goal(), out_root=str(tmp_path / "empty"), confirm_default="each")


# ---- the server's execute path on a stub ----------------------------------------------------------------------------------
class _GH:
    def __init__(self, goal):
        self.request = goal
        self.is_cancel_requested = False
        self.ended = None

    def abort(self):
        self.ended = "abort"

    def succeed(self):
        self.ended = "succeed"

    def canceled(self):
        self.ended = "canceled"


def _stub(tmp_path, *, busy=None, run_result=None):
    from hongik_scan_ros.auto_teach_server import AutoTeachServer
    st = SimpleNamespace(out_root=str(tmp_path / "out_root"), confirm_mode="each", said=[], calls=[],
                         _cancel=threading.Event())
    st._say = lambda line, level="info": st.said.append((level, line))
    st._claim = lambda kind, gh: busy
    st._release_slot = lambda: st.calls.append("release_slot")
    st._not_ready = lambda: []
    st._release_profiler = lambda: (st.calls.append("release_profiler"), (True, "no profiler node"))[1]

    def run(run_, inp):
        st.calls.append(("run", str(run_.run_dir), inp["dry_run"]) + ((inp["after_pass"],) if inp["after_pass"] else ()))
        return run_result
    st._run_post_weld = run
    st._execute_post_weld_claimed = lambda gh, res: AutoTeachServer._execute_post_weld_claimed(st, gh, res)
    return AutoTeachServer, st


def test_server_refuses_while_another_run_holds_the_slot(tmp_path):
    srv, st = _stub(tmp_path, busy="auto_teach goal running, phase pass")
    gh = _GH(_goal())
    res = srv._execute_post_weld(st, gh)
    assert res.status == "REFUSED" and res.message == "busy: another sensing run is active (auto_teach goal running, phase pass; nothing sent)"
    assert gh.ended == "abort" and st.calls == []                     # the slot is the other goal's: not released, nothing run
    assert st.said[-1][0] == "warn" and st.said[-1][1].startswith("post_weld_scan result: REFUSED: busy")


def test_server_refuses_a_bad_input_run_and_runs_a_dry_run_without_the_profiler(tmp_path):
    from scan_l0.post_weld import PostWeldResult
    (tmp_path / "out_root").mkdir()
    srv, st = _stub(tmp_path)
    gh = _GH(_goal(run_dir="missing"))
    res = srv._execute_post_weld(st, gh)
    assert res.status == "REFUSED" and "does not exist" in res.message and res.message.endswith("(nothing sent)")
    assert gh.ended == "abort" and st.calls == ["release_slot"]
    d = make_run(tmp_path / "out_root", "k30_1_a")
    srv, st = _stub(tmp_path)
    gh = _GH(_goal(after_pass=5))                                         # the run's weld plan has 4 passes
    res = srv._execute_post_weld(st, gh)
    assert res.status == "REFUSED" and "after_pass 5: the run's weld plan has passes 1..4" in res.message and gh.ended == "abort"
    assert st.calls == ["release_slot"]
    srv, st = _stub(tmp_path, run_result=PostWeldResult(status="DRY_RUN", reason="dry run", summary="post-weld scan DRY_RUN: run k30_1_a"))
    gh = _GH(_goal(dry_run=True))
    res = srv._execute_post_weld(st, gh)
    assert (res.status, res.message, res.out_dir, res.run_dir) == ("DRY_RUN", "post-weld scan DRY_RUN: run k30_1_a", "", d)
    assert gh.ended == "succeed" and st.calls == [("run", d, True), "release_slot"]          # no profiler release in a dry run
    srv, st = _stub(tmp_path, run_result=PostWeldResult(status="OK", out_dir=d + "/post_weld_scan_x", summary="post-weld scan OK"))
    gh = _GH(_goal(after_pass=2))
    res = srv._execute_post_weld(st, gh)
    assert res.status == "OK" and res.out_dir == d + "/post_weld_scan_x" and gh.ended == "succeed"
    assert st.calls == ["release_profiler", ("run", d, False, 2), "release_slot"]
    srv, st = _stub(tmp_path, run_result=PostWeldResult(status="REFUSED", summary="post-weld scan REFUSED: no direction"))
    gh = _GH(_goal())
    assert srv._execute_post_weld(st, gh).status == "REFUSED" and gh.ended == "abort"
