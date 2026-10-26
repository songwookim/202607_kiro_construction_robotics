"""hongik_precision_sensing (precision_sensing_client) unit tests, no ROS graph: the two points (--a / --b, --points FILE, the saved
points and the reuse question with its two answers), the frame check against the server's ~/status, the goal message, the end
summary and the result file (the full result message, the server's text; round trip in test_kiro_io)."""
from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
STACK = os.path.dirname(PKG)
for p in (PKG, os.path.join(STACK, "scan_l0"), os.path.join(STACK, "scan_core"), os.path.join(STACK, "vc3dz_driver")):
    if p not in sys.path:
        sys.path.insert(0, p)

from hongik_scan_ros import kiro_io as K  # noqa: E402
from hongik_scan_ros import precision_sensing_client as C  # noqa: E402


def _args(argv):
    return C.build_parser().parse_args(C.fold_coordinate_args(argv))


class _Answers:
    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


# --------------------------------------------------------------------------------------
# points
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["0.69,0.015,-0.03", "0.69, 0.015, -0.03", "0.69 0.015 -0.03", " 0.69,0.015 -0.03 "])
def test_a_point_is_three_numbers_with_commas_or_spaces(text):
    assert C.parse_point(text) == (0.69, 0.015, -0.03)


@pytest.mark.parametrize("text", ["0.69,0.015", "0.69,0.015,-0.03,1", "x,y,z", "0.69,nan,0", "0.69,inf,0", ""])
def test_anything_else_is_not_a_point(text):
    with pytest.raises(ValueError, match="a point is X,Y,Z in metres"):
        C.parse_point(text)


@pytest.mark.parametrize("argv", [["--a", "-0.5,0.1,0.2", "--b", "1,-2,3"], ["--a=-0.5,0.1,0.2", "--b", "1 -2 3"],
                                  ["--a", "-0.5 0.1 0.2", "--b", "1,-2,3"], ["--a", "-0.5, 0.1, 0.2", "--b=1, -2, 3"]])
def test_command_line_points_with_a_leading_minus(argv, tmp_path):
    pts = C.choose_points(_args(argv), ask=None, say=None, store=tmp_path / "none.yaml")
    assert (pts.a, pts.b, pts.frame, pts.source) == ((-0.5, 0.1, 0.2), (1.0, -2.0, 3.0), None, "--a / --b")


@pytest.mark.parametrize("argv, why", [(["--a", "1,2,3"], "--a and --b go together"),
                                       (["--b", "1,2,3"], "--a and --b go together"),
                                       (["--a", "1,2,3", "--b", "4,5,6", "--points", "p.yaml"], "exclude each other"),
                                       (["--a", "1,2", "--b", "4,5,6"], "a point is X,Y,Z")])
def test_incomplete_or_conflicting_points_are_refused(argv, why, tmp_path):
    with pytest.raises(ValueError, match=why):
        C.choose_points(_args(argv), ask=None, say=None, store=tmp_path / "none.yaml")
    said = []
    assert C.main(argv, ask=_Answers(), say=said.append, store=tmp_path / "none.yaml") == 2
    assert said[0].startswith("REFUSED: ") and why in said[0]


def test_points_file(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("a: [0.69, 0.015, -0.03]\nb: [1.01, -0.01, -0.03]\nframe: World\n", encoding="utf-8")
    pts = C.choose_points(_args(["--points", str(f)]), ask=None, say=None, store=tmp_path / "none.yaml")
    assert (pts.a, pts.b, pts.frame, pts.source) == ((0.69, 0.015, -0.03), (1.01, -0.01, -0.03), "World", f"points file {f}")
    f.write_text("a: '0.69, 0.015, -0.03'\nb: [1, 0, 0]\n", encoding="utf-8")          # 'X,Y,Z' text and integers, no frame
    pts = C.load_points_file(f)
    assert (pts.a, pts.b, pts.frame) == ((0.69, 0.015, -0.03), (1.0, 0.0, 0.0), None)


@pytest.mark.parametrize("text, why", [("a: [1, 2, 3]\n", "missing b"), ("a: [1, 2, 3]\nb: [1, 2, 3]\nc: 1\n", "unknown key(s) c"),
                                       ("a: [1, 2]\nb: [1, 2, 3]\n", ": a: [x, y, z] in metres expected"),
                                       ("a: [1, 2, true]\nb: [1, 2, 3]\n", ": a: [x, y, z] in metres expected"),
                                       ("- 1\n- 2\n", "a mapping"), ("a: [1, 2, 3\n", "not YAML")])
def test_a_points_file_that_does_not_give_two_points_is_refused(tmp_path, text, why):
    f = tmp_path / "p.yaml"
    f.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=f"points file {f}"):
        C.load_points_file(f)
    with pytest.raises(ValueError) as e:
        C.load_points_file(f)
    assert why in str(e.value)
    with pytest.raises(ValueError, match="No such file"):
        C.load_points_file(tmp_path / "missing.yaml")


def test_saved_points_are_a_points_file_with_the_servers_frame(tmp_path):
    store = tmp_path / "home" / ".hongik" / "precision_sensing_points.yaml"
    pts = C.Points((0.1 + 0.2, -1.0 / 3.0, 1e-17), (1.0, 2.0, 3.0), None, "--a / --b")
    assert C.save_points(store, pts, "World") == store
    text = store.read_text(encoding="utf-8")
    assert text.startswith("# hongik_precision_sensing: the points of the last accepted goal (") and "frame: World" in text
    back = C.load_points_file(store)
    assert (back.a, back.b, back.frame) == (pts.a, pts.b, "World")                     # exact floats


# --------------------------------------------------------------------------------------
# no points given: the saved points and the reuse question (use / do not use)
# --------------------------------------------------------------------------------------
def test_reuse_question_use(tmp_path):
    store = tmp_path / "s.yaml"
    C.save_points(store, C.Points((0.69, 0.015, -0.03), (1.01, -0.01, -0.03), None, "x"), "link0")
    ask, said = _Answers("maybe", "u"), []
    pts = C.choose_points(_args([]), ask=ask, say=said.append, store=store)
    assert ask.prompts == ["Reuse the previous points (A [0.6900, 0.0150, -0.0300], B [1.0100, -0.0100, -0.0300] m, link0)? "
                           "[u]se / [d]o not use: "] * 2
    assert said == ["  answer u (use the previous points) or d (do not use them)"]
    assert (pts.a, pts.b, pts.frame, pts.source) == ((0.69, 0.015, -0.03), (1.01, -0.01, -0.03), "link0",
                                                     f"the previous points, {store}")
    assert C.choose_points(_args([]), ask=_Answers("use"), say=said.append, store=store).a == pts.a


@pytest.mark.parametrize("answer", ["d", "do not use", "D"])
def test_reuse_question_do_not_use_starts_nothing(tmp_path, answer):
    store = tmp_path / "s.yaml"
    C.save_points(store, C.Points((0.69, 0.015, -0.03), (1.01, -0.01, -0.03), None, "x"), "link0")
    assert C.choose_points(_args([]), ask=_Answers(answer), say=print, store=store) is None
    said = []
    assert C.main([], ask=_Answers(answer), say=said.append, store=store) == 2           # before any ROS: nothing sent
    assert said == ["not started: give the two points: --a X,Y,Z --b X,Y,Z or --points FILE"]


def test_no_points_and_none_saved(tmp_path):
    said, ask = [], _Answers()
    assert C.main([], ask=ask, say=said.append, store=tmp_path / "none.yaml") == 2
    assert not ask.prompts and said == [f"no points given and none saved ({tmp_path / 'none.yaml'})",
                                        "not started: give the two points: --a X,Y,Z --b X,Y,Z or --points FILE"]
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [1, 2, 3]\n", encoding="utf-8")
    said = []
    assert C.main([], ask=ask, say=said.append, store=bad) == 2 and not ask.prompts
    assert said[0].startswith("the saved points are not usable: points file ") and said[0].endswith("missing b")


def test_an_unfinished_answer_starts_nothing(tmp_path):
    store = tmp_path / "s.yaml"
    C.save_points(store, C.Points((0.69, 0.015, -0.03), (1.01, -0.01, -0.03), None, "x"), "link0")
    said = []
    assert C.main([], ask=_Answers(), say=said.append, store=store) == 2                 # EOF at the question
    assert said == ["\nnot started: the input was not completed"]


# --------------------------------------------------------------------------------------
# the server's frame, the goal
# --------------------------------------------------------------------------------------
def test_the_servers_base_frame_from_its_status():
    st = ("adapter=moveit_jtc base_frame=World running=False goal= phase=idle pass=0 ready=True not_ready='' run_stopped=False "
          "stop_reason='' awaiting_confirm=False last=")
    assert C.status_base_frame(st) == "World"
    assert C.status_base_frame(st.replace("base_frame=World", "base_frame=link0")) == "link0"
    assert C.status_base_frame("adapter=moveit_jtc running=False last=base_frame=x") is None and C.status_base_frame(None) is None


def test_points_in_another_frame_are_refused():
    pts = C.Points((1, 2, 3), (4, 5, 6), "World", "points file p.yaml")
    C.check_frame(pts, "World")
    C.check_frame(C.Points((1, 2, 3), (4, 5, 6), None, "--a / --b"), "link0")
    with pytest.raises(ValueError, match="points file p.yaml: the points are in World, the server reads goal points in link0 - give "
                                         "them in link0 \\(the client does not transform points\\)"):
        C.check_frame(pts, "link0")


def test_goal_is_1g_with_the_points_and_identity_orientations():
    act = pytest.importorskip("welding_interfaces.action")
    g = C.goal_from_points(C.Points((0.69, 0.015, -0.03), (1.01, -0.01, -0.03), None, "--a / --b"))
    assert isinstance(g, act.PrecisionSensing.Goal) and g.welding_position == 1 == act.PrecisionSensing.Goal.POSITION_1G
    for pose, P in ((g.pose_a, (0.69, 0.015, -0.03)), (g.pose_b, (1.01, -0.01, -0.03))):
        assert (pose.position.x, pose.position.y, pose.position.z) == P
        assert (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w) == (0.0, 0.0, 0.0, 1.0)
    inp = K.goal_to_run_inputs(g, g1_pipeline="core")                                    # what the server makes of it
    assert (inp.position, inp.start_m, inp.end_m) == ("1G", (0.69, 0.015, -0.03), (1.01, -0.01, -0.03))


# --------------------------------------------------------------------------------------
# the end: run folder, summary, result file
# --------------------------------------------------------------------------------------
def _result(code=0, message="PASS: every quantity within its target", data_file="", n_paths=4, n_wp=41):
    act = pytest.importorskip("welding_interfaces.action")
    from geometry_msgs.msg import Pose
    from welding_interfaces.msg import WeldingPath
    r = act.PrecisionSensing.Result()
    r.error_code, r.error_message, r.data_file_path = code, message, data_file
    for k in range(n_paths):
        w = WeldingPath()
        poses = []
        for i in range(n_wp):
            ps = Pose()
            ps.position.x, ps.position.y, ps.position.z = 0.7 + 0.005 * i, 0.001 * k, -0.06 + 1e-3 / 3.0
            ps.orientation.x, ps.orientation.y, ps.orientation.z, ps.orientation.w = 0.0, 0.0, math.sin(0.3), math.cos(0.3)
            poses.append(ps)
        w.waypoints = poses
        r.paths.append(w)
    return r


def test_run_folder_from_the_servers_line_else_the_data_file():
    res = SimpleNamespace(data_file_path="/runs/a/precision_sensing.npz")
    assert C.run_folder(res, "/runs/b/precision_sensing_result.yaml") == "/runs/b"
    assert C.run_folder(res, None) == "/runs/a"
    assert C.run_folder(SimpleNamespace(data_file_path=""), None) is None


def test_summary_and_result_file_of_a_success(tmp_path):
    run = tmp_path / "20261013-120000_kiro_1G"
    (run / "kiro").mkdir(parents=True)
    for name in ("pass_10.yaml", "pass_2.yaml", "pass_1.yaml", "pass_1_info.yaml", "README.md"):
        (run / "kiro" / name).write_text("x", encoding="utf-8")
    res = _result(data_file=str(run / K.NPZ_NAME))
    lines = C.summary_lines(res, str(run))
    assert lines == ["RESULT: SUCCESS (0) - PASS: every quantity within its target", "  paths: 4 (waypoints per pass: 41, 41, 41, 41)",
                     f"  run folder: {run}", "  KIRO-format passes: kiro/pass_1.yaml, kiro/pass_2.yaml, kiro/pass_10.yaml",
                     f"  data file: {run / K.NPZ_NAME}"]
    assert C.save_result(res, str(run)) == f"  result file: {run / K.RESULT_YAML_NAME}"
    text = (run / K.RESULT_YAML_NAME).read_text(encoding="utf-8")
    assert text == K.result_yaml_text(res)                                               # the server's text for the same message
    assert sorted(os.listdir(run)) == ["kiro", K.RESULT_YAML_NAME]                       # nothing else written


def test_summary_and_result_file_without_a_run_folder(tmp_path):
    res = _result(code=K.ERROR_INVALID_GOAL, message="REFUSED: welding_position 2 (2G) not supported (nothing sent)", n_paths=0)
    assert C.summary_lines(res, None) == ["RESULT: ERROR_INVALID_GOAL (5) - REFUSED: welding_position 2 (2G) not supported "
                                          "(nothing sent)", "  paths: 0", "  run folder: none (the goal ended before the run made one)",
                                          "  data file: none"]
    assert C.save_result(res, None) == "  result file: not saved (no run folder)"
    elsewhere = str(tmp_path / "on_the_server_pc")
    assert C.save_result(res, elsewhere) == f"  result file: not saved (the run folder {elsewhere} is not on this PC)"
    assert not os.path.exists(elsewhere)
