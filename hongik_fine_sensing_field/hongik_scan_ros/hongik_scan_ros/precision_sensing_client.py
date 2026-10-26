"""hongik_precision_sensing - send one PrecisionSensing goal (welding_interfaces, 1G) to the server's
/hongik_fine_sensing/precision_sensing action from a terminal: the operator's way to give the two rough points.

A and B = approximate points near the two ends of the weld line, on the seam inside the specimen span: positions [m] in the frame
the server reads goal points in (its base frame = MoveIt's planning frame; World on KIRO's PC). Orientations are sent as identity
(the server does not use them). The run decides which end it calls S (the end it visits first; the npz meta ends_from_goal names
the point).

  --a X,Y,Z --b X,Y,Z   the two points (also 'X Y Z' in quotes)
  --points FILE         the two points from a YAML file {a: [x, y, z], b: [x, y, z], frame: <name>}; frame is optional - when
                        given it must be the server's base frame (the client does not transform points)
  (neither)             the previous goal's points (~/.hongik/precision_sensing_points.yaml, a points file):
                        "Reuse the previous points (A [..], B [..])? [u]se / [d]o not use"
  --server-timeout S    wait for the action server [s] (default 30)
  --log-topic NAME      server log topic ('' = find it, 'none' = the action feedback only: elapsed time)

The points of every accepted goal are saved to ~/.hongik/precision_sensing_points.yaml. Progress: every line of the run.
Ctrl-C = cancel the goal = the server stops our trajectory and ends the run (no reset afterwards). At the end: a summary (result code,
message, paths, run folder, kiro/pass_<k>.yaml, data file) and the full result message saved as
<run folder>/precision_sensing_result.yaml (the server writes the same file for every goal, KIRO's main system's included).
No dry run: the PrecisionSensing goal has no field for one. The client sends no motion command.
Exit code: 0 = SUCCESS, 1 = the goal ended with an error code (or was not sent to a server), 2 = not started (points).

  ros2 run hongik_scan_ros hongik_precision_sensing --a 0.69,0.015,-0.03 --b 1.01,-0.01,-0.03
  ros2 run hongik_scan_ros hongik_precision_sensing --points points.yaml
  ros2 run hongik_scan_ros hongik_precision_sensing
"""
# (한국어 설명, 코드 주석) KIRO PrecisionSensing 액션의 운영자 클라이언트 (사용자 2026-10-13): 두 점은 좌표로만 (--a/--b, --points, 이전 점
# 재사용 질문 use / do not use). 서버가 센서·로봇을 가지므로 클라이언트는 목표·취소만 보낸다 (모션 명령·제어기 통신 없음). 결과 메시지 전체를
# <run>/precision_sensing_result.yaml 로 (서버도 같은 글을 쓴다 — KIRO 메인 시스템의 목표도). 문구는 모두 영어.
from __future__ import annotations

import argparse
import datetime as _dt
import math
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import kiro_io as K
from .auto_teach_client import (CANCEL_ANSWER_S, LOG_DISCOVERY_S, LOG_DRAIN_S, ProgressPrinter, _cancel_goal, _wait,
                                log_topic_candidates)

__all__ = ["ACTION", "POINTS_STORE", "Points", "parse_point", "fold_coordinate_args", "build_parser", "load_points_file",
           "save_points", "reuse_prompt", "ask_reuse_points", "choose_points", "status_base_frame", "check_frame", "goal_from_points",
           "RESULT_FILE_PREFIX", "run_folder", "kiro_pass_files", "summary_lines", "save_result", "main"]

ACTION = "/hongik_fine_sensing/precision_sensing"
POINTS_STORE = "~/.hongik/precision_sensing_points.yaml"
POINTS_KEYS = ("a", "b", "frame")
RESULT_FILE_PREFIX = "precision_sensing: result file "          # the server's line naming the run's result file
RESULT_LINE_PREFIX = "precision_sensing result: "                # the server's last line of a goal
ELAPSED_STEP_S = 10                                               # feedback-only progress: one elapsed-time line per this many seconds


@dataclass(frozen=True)
class Points:
    a: tuple                          # (x, y, z) m
    b: tuple
    frame: Optional[str]              # None = not stated (the server's base frame)
    source: str                       # where they came from (printed with the goal)


def _xyz_text(p) -> str:
    return "[" + ", ".join(f"{float(v):.4f}" for v in p) + "]"


def parse_point(text) -> tuple:
    """'X,Y,Z' or 'X Y Z' [m] → (x, y, z). ValueError unless exactly three finite numbers."""
    parts = [t for t in re.split(r"[,\s]+", str(text).strip()) if t]
    try:
        v = tuple(float(t) for t in parts)
    except ValueError:
        v = ()
    if len(v) != 3 or not all(math.isfinite(x) for x in v):
        raise ValueError(f"a point is X,Y,Z in metres (three finite numbers), got {str(text)!r}")
    return v


def _is_point(text: str) -> bool:
    try:
        parse_point(text)
        return True
    except ValueError:
        return False


def fold_coordinate_args(argv) -> list:
    """'--a -0.5,0.1,0.2' → '--a=-0.5,0.1,0.2': argparse reads a token that starts with '-' and is not a plain number as an option."""
    out, i, argv = [], 0, list(argv)
    while i < len(argv):
        t = argv[i]
        if t in ("--a", "--b") and i + 1 < len(argv) and argv[i + 1].startswith("-") and _is_point(argv[i + 1]):
            out.append(f"{t}={argv[i + 1]}")
            i += 2
            continue
        out.append(t)
        i += 1
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hongik_precision_sensing", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", metavar="X,Y,Z", help="point A near one end of the weld line [m], the server's base frame")
    ap.add_argument("--b", metavar="X,Y,Z", help="point B near the other end of the weld line [m], the server's base frame")
    ap.add_argument("--points", metavar="FILE", help="the two points from a YAML file {a: [x, y, z], b: [x, y, z], frame: <name>}")
    ap.add_argument("--action", default=ACTION)
    ap.add_argument("--server-timeout", type=float, default=30.0, metavar="S", help="wait for the action server [s]")
    ap.add_argument("--log-topic", default="", metavar="NAME",
                    help="server log topic (std_msgs/String); '' = find it, 'none' = use the action feedback only (elapsed time)")
    return ap


def _point_value(v, what: str) -> tuple:
    if isinstance(v, str):
        return parse_point(v)
    if isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
        return parse_point(" ".join(repr(float(x)) for x in v))
    raise ValueError(f"{what}: [x, y, z] in metres expected, got {v!r}")


def load_points_file(path) -> Points:
    """A points file {a: [x, y, z], b: [x, y, z], frame: <name>} (frame optional) → Points. ValueError with the reason."""
    import yaml
    p = Path(path).expanduser()
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except OSError as e:
        raise ValueError(f"points file {p}: {e.strerror or e}") from None
    except yaml.YAMLError as e:
        raise ValueError(f"points file {p}: not YAML ({type(e).__name__})") from None
    if not isinstance(doc, dict):
        raise ValueError(f"points file {p}: a mapping {{a: [x, y, z], b: [x, y, z], frame: <name>}} expected")
    unknown = sorted(str(k) for k in doc if k not in POINTS_KEYS)
    if unknown:
        raise ValueError(f"points file {p}: unknown key(s) {', '.join(unknown)} (keys: a, b, frame)")
    missing = [k for k in ("a", "b") if k not in doc]
    if missing:
        raise ValueError(f"points file {p}: missing {' and '.join(missing)}")
    frame = doc.get("frame")
    frame = str(frame).strip() if frame is not None else ""
    return Points(_point_value(doc["a"], f"points file {p}: a"), _point_value(doc["b"], f"points file {p}: b"), frame or None,
                  f"points file {p}")


def save_points(path, pts: Points, frame: str) -> Path:
    """The goal's points → a points file (written atomically) with the frame the server read them in."""
    import yaml
    from scan_core.g1_types import atomic_write_bytes
    p = Path(path).expanduser()
    stamp = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    text = (f"# hongik_precision_sensing: the points of the last accepted goal ({stamp}); a points file (--points)\n"
            + yaml.safe_dump({"a": [float(v) for v in pts.a], "b": [float(v) for v in pts.b], "frame": str(frame)}, sort_keys=False,
                             default_flow_style=None))
    atomic_write_bytes(p, text.encode("utf-8"))
    return p


def reuse_prompt(pts: Points) -> str:
    return (f"Reuse the previous points (A {_xyz_text(pts.a)}, B {_xyz_text(pts.b)} m"
            + (f", {pts.frame}" if pts.frame else "") + ")? [u]se / [d]o not use: ")


def ask_reuse_points(ask, say, pts: Points) -> bool:
    """The reuse question → True (use) / False (do not use); any other answer asks again."""
    while True:
        ans = str(ask(reuse_prompt(pts))).strip().lower()
        if ans in ("u", "use"):
            return True
        if ans in ("d", "do not use", "dont use", "don't use"):
            return False
        say("  answer u (use the previous points) or d (do not use them)")


def choose_points(a, *, ask, say, store) -> Optional[Points]:
    """--a/--b, --points, or (neither) the saved points after the reuse question → Points, or None (no points: not started).
    ValueError for inputs that do not make two points."""
    given = [n for n in ("a", "b") if getattr(a, n)]
    if a.points and given:
        raise ValueError("--points and --a / --b exclude each other")
    if len(given) == 1:
        raise ValueError("--a and --b go together (both points)")
    if given:
        return Points(parse_point(a.a), parse_point(a.b), None, "--a / --b")
    if a.points:
        return load_points_file(a.points)
    store = Path(store).expanduser()
    if not store.exists():
        say(f"no points given and none saved ({store})")
        return None
    try:
        prev = load_points_file(store)
    except ValueError as e:
        say(f"the saved points are not usable: {e}")
        return None
    if not ask_reuse_points(ask, say, prev):
        return None
    return Points(prev.a, prev.b, prev.frame, f"the previous points, {store}")


def status_base_frame(text) -> Optional[str]:
    """The server's ~/status message → its base frame (base_frame=<name>), None if not named."""
    m = re.search(r"(?:^|\s)base_frame=(\S+)", str(text or ""))
    return m.group(1) if m else None


def check_frame(pts: Points, server_frame: str) -> None:
    """ValueError when the points name a frame other than the server's (the client does not transform points)."""
    if pts.frame and pts.frame != server_frame:
        raise ValueError(f"{pts.source}: the points are in {pts.frame}, the server reads goal points in {server_frame} - give them in "
                         f"{server_frame} (the client does not transform points)")


def goal_from_points(pts: Points):
    """Points → PrecisionSensing.Goal: welding_position 1 (1G), pose_a / pose_b = the points, orientation identity (not used)."""
    from welding_interfaces.action import PrecisionSensing
    g = PrecisionSensing.Goal()
    g.welding_position = int(PrecisionSensing.Goal.POSITION_1G)
    for pose, P in ((g.pose_a, pts.a), (g.pose_b, pts.b)):
        pose.position.x, pose.position.y, pose.position.z = (float(v) for v in P)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = 0.0, 0.0, 0.0, 1.0
    return g


# ---- the end of a goal ------------------------------------------------------------------------------------------------------
def run_folder(res, result_file: Optional[str]) -> Optional[str]:
    """The goal's run folder: the folder of the server's result file (its 'precision_sensing: result file' line), else the folder of
    data_file_path, else None (the goal ended before the run made a folder)."""
    if result_file:
        return str(Path(result_file).parent)
    if str(res.data_file_path or ""):
        return str(Path(str(res.data_file_path)).parent)
    return None


def kiro_pass_files(run_dir: Optional[str]) -> list:
    """kiro/pass_<k>.yaml of the run folder (KIRO-format straight lines), in pass order; [] without them."""
    d = Path(run_dir) / "kiro" if run_dir else None
    if d is None or not d.is_dir():
        return []
    found = [(int(m.group(1)), f) for f in d.iterdir() if (m := re.fullmatch(r"pass_(\d+)\.yaml", f.name))]
    return [f for _k, f in sorted(found)]


def summary_lines(res, run_dir: Optional[str]) -> list:
    """The result → the summary lines (code name, message, paths, run folder, KIRO-format pass files, data file)."""
    code = int(res.error_code)
    out = [f"RESULT: {K.CODE_NAMES.get(code, '?')} ({code}) - {res.error_message}"]
    out.append(f"  paths: {len(res.paths)}")
    for p in res.paths:                                  # pass_number + the start / goal wire tip of each pass
        a, b = p.start.position, p.goal.position
        out.append(f"    pass {p.pass_number}: start ({a.x:.4f}, {a.y:.4f}, {a.z:.4f}) -> goal ({b.x:.4f}, {b.y:.4f}, {b.z:.4f}) m")
    out.append(f"  run folder: {run_dir}" if run_dir else "  run folder: none (the goal ended before the run made one)")
    files = kiro_pass_files(run_dir)
    if files:
        out.append("  KIRO-format passes: " + ", ".join(f"kiro/{f.name}" for f in files))
    out.append(f"  data file: {res.data_file_path}" if str(res.data_file_path or "") else "  data file: none")
    return out


def save_result(res, run_dir: Optional[str]) -> str:
    """The full result message → <run folder>/precision_sensing_result.yaml (the server's text for the same message) → the line."""
    if not run_dir:
        return "  result file: not saved (no run folder)"
    if not Path(run_dir).is_dir():
        return f"  result file: not saved (the run folder {run_dir} is not on this PC)"
    try:
        return f"  result file: {K.write_result_yaml(run_dir, res)}"
    except OSError as e:
        return f"  result file: not saved ({type(e).__name__}: {e})"


def main(argv=None, *, ask=None, say=None, store=None) -> int:
    """ask / say: input / output (default input / print); store: the saved points file (default POINTS_STORE)."""
    ask = ask or input
    say = say or (lambda s_: print(s_, flush=True))
    a = build_parser().parse_args(fold_coordinate_args(sys.argv[1:] if argv is None else argv))
    store = Path(store or POINTS_STORE).expanduser()
    try:
        pts = choose_points(a, ask=ask, say=say, store=store)
    except ValueError as e:
        say(f"REFUSED: {e}")
        return 2
    except (EOFError, KeyboardInterrupt):
        say("\nnot started: the input was not completed")
        return 2
    if pts is None:
        say("not started: give the two points: --a X,Y,Z --b X,Y,Z or --points FILE")
        return 2

    import rclpy
    from rclpy.action import ActionClient
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    from welding_interfaces.action import PrecisionSensing
    from .ros_util import call_service

    goal = goal_from_points(pts)
    if not rclpy.ok():
        # Ctrl-C = KeyboardInterrupt with the context alive (the cancel must still reach the server; auto_teach_client 2026-10-08)
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("hongik_precision_sensing")
    client = ActionClient(node, PrecisionSensing, a.action)
    status_name = a.action.rsplit("/", 1)[0] + "/hongik_auto_teach_server/status"
    status_cli = node.create_client(Trigger, status_name)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    rc = 1
    try:
        try:
            if not client.wait_for_server(timeout_sec=a.server_timeout):
                say(f"REFUSED: action server {a.action} not available (is hongik_moveit.launch.py running with this package?)")
                return 1
            st = call_service(status_cli, Trigger.Request(), timeout_s=10.0, wait_service_s=a.server_timeout)
            frame = status_base_frame(st.message) if st is not None else None
            if frame is None:
                say(f"REFUSED: the server's {status_name} did not name its base frame (the frame of the goal points) - nothing sent")
                return 1
            try:
                check_frame(pts, frame)
            except ValueError as e:
                say(f"REFUSED: {e} - nothing sent")
                return 2
            topic = None
            if str(a.log_topic).strip().lower() != "none":
                cands = [a.log_topic] if a.log_topic else log_topic_candidates(a.action)
                t_end = time.monotonic() + LOG_DISCOVERY_S
                while topic is None:
                    topic = next((t for t in cands if node.count_publishers(t) > 0), None)
                    if topic is not None or time.monotonic() >= t_end:
                        break
                    time.sleep(0.05)
        except KeyboardInterrupt:
            say("\nCtrl-C: not started (nothing sent)")
            return 1
        seen = {"result_file": None, "last": ""}

        def out(line: str) -> None:
            if line.startswith(RESULT_FILE_PREFIX):
                seen["result_file"] = line[len(RESULT_FILE_PREFIX):].strip()
            seen["last"] = line
            say(line)
        pr = ProgressPrinter(out, use_log=topic is not None)
        if topic is not None:
            qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1000, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE)
            node.create_subscription(String, topic, lambda m: pr.on_log(m.data), qos)

        def on_fb(fb):
            t = float(fb.feedback.elapsed_time)
            pr.on_feedback(f"elapsed {int(t // ELAPSED_STEP_S) * ELAPSED_STEP_S} s")
        say(f"goal: 1G, A {_xyz_text(pts.a)} B {_xyz_text(pts.b)} m in {frame} ({pts.source}) (Ctrl-C = cancel = stop the run); "
            "progress " + (f"from {topic}" if topic is not None else "from the action feedback"))
        fut = client.send_goal_async(goal, feedback_callback=on_fb)
        interrupted = False
        try:
            while not fut.done():
                time.sleep(0.05)
        except KeyboardInterrupt:                               # Ctrl-C while sending: cancel the goal once it is accepted
            interrupted = True
            if not _wait(fut, CANCEL_ANSWER_S):
                say("\nCtrl-C: no answer to the goal (nothing to cancel)")
                return 1
        gh = fut.result()
        if gh is None or not gh.accepted:
            say("REFUSED: the goal was not accepted by the server (see the server log)")
            return 1
        try:
            say(f"points saved: {save_points(store, pts, frame)}")
        except OSError as e:
            say(f"points not saved ({store}): {type(e).__name__}: {e}")
        res_fut = gh.get_result_async()
        if not interrupted:
            try:
                while not res_fut.done():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                interrupted = True
        if interrupted and not _cancel_goal(gh, res_fut):          # Ctrl-C = cancel; the cancelled run's result is summarized
            return 1
        if pr.use_log:
            # the log topic and the result arrive separately: wait briefly for the server's result line (the result file line
            # comes before it)
            t_end = time.monotonic() + LOG_DRAIN_S
            while not str(seen["last"]).startswith(RESULT_LINE_PREFIX) and time.monotonic() < t_end:
                time.sleep(0.05)
        res = res_fut.result().result
        run_dir = run_folder(res, seen["result_file"])
        for line in summary_lines(res, run_dir):
            say(line)
        say(save_result(res, run_dir))
        rc = 0 if int(res.error_code) == K.SUCCESS else 1
    finally:
        executor.shutdown()
        spin.join(timeout=5.0)
        client.destroy()
        node.destroy_node()
        rclpy.try_shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
