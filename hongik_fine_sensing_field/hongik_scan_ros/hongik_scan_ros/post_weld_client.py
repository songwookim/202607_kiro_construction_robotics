"""hongik_post_weld_scan - after the welding (or after any pass of it), scan the whole weld once - raw data only - through the
server's /hongik_fine_sensing/post_weld_scan action (hongik_scan_msgs/action/PostWeldScan).

The server takes a finished 1G run (its measured seam ends, frame and scan settings; default: the latest run under its out_root) and
runs ONE straight scan from 20 mm beyond the measured S end to 20 mm beyond the measured E end along the measured seam, like a
sensing pass of that run: its world, its pass standoff and sensor orientation, the measurement speed, the run's sensor settings;
positioning moves through MoveIt at the platform limits from wherever the arm is. Every frame is recorded as it comes - no profile
checks, no estimation, no weld plan, no KIRO files. Output: <run>/post_weld_scan_p<k>|final_<YYYYmmdd-HHMMSS>/ (raw.npz +
run_log.json). Ctrl-C = cancel the goal = the server stops the robot.

  --run DIR              the run folder (absolute, or a folder name under the server's out_root); default = the latest run
  --measure-speed V      the scan line's speed [mm/s]; default = the run's measurement speed (its passes). Any speed inside the
                         range the noise priors were calibrated over (the calibration file's g1 noise_priors speed_range_mps,
                         eoat_kiro_01: 15..60 mm/s) - the server checks it like a run's measurement speed chosen at run time and
                         refuses others before any motion
  --confirm auto|each    auto (default): ask only for a joint-space detour; each: confirm every move (~/confirm_move)
  --after-pass K         the pass just completed (1..n of the run's weld plan); default = final (after every pass) - names the
                         output folder (post_weld_scan_p<K>_… / post_weld_scan_final_…) and goes into the run log; the scan is the
                         same after any pass
  --dry-run              plan and print the line and the moves; nothing is sent, the sensor is not opened, no folder written

  ros2 run hongik_scan_ros hongik_post_weld_scan
  ros2 run hongik_scan_ros hongik_post_weld_scan --run k30_1_a --measure-speed 30 --dry-run
  ros2 run hongik_scan_ros hongik_post_weld_scan --run k30_1_a --measure-speed 45
  ros2 run hongik_scan_ros hongik_post_weld_scan --after-pass 2
"""
# (한국어 설명, 코드 주석) 용접 뒤 데이터 수집 스캔 (사용자 2026-10-13) 의 터미널 클라이언트. 서버가 센서·로봇을 가지므로 스캔은 서버 안에서
# (한 번에 한 목표). 진행 줄은 서버 로그 토픽 (~/log) 을 하나도 빠짐없이 (auto_teach_client 의 ProgressPrinter 그대로). 문구는 모두 영어.
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from .auto_teach_client import (CANCEL_ANSWER_S, LOG_DISCOVERY_S, LOG_DRAIN_S, ProgressPrinter, _cancel_goal, _wait,
                                log_topic_candidates, measure_speed_mps)

__all__ = ["ACTION", "CONFIRM_CHOICES", "build_parser", "goal_from_args", "result_lines", "main", "OK_STATUSES"]

ACTION = "/hongik_fine_sensing/post_weld_scan"
CONFIRM_CHOICES = ("auto", "each")
OK_STATUSES = ("OK", "DRY_RUN")                   # exit code 0


def _pass_number(txt: str) -> int:
    """--after-pass: a pass number >= 1."""
    try:
        k = int(txt)
    except ValueError:
        raise argparse.ArgumentTypeError(f"a pass number >= 1, got {txt!r}")
    if k < 1:
        raise argparse.ArgumentTypeError(f"a pass number >= 1, got {k} (leave it out for the final scan)")
    return k


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hongik_post_weld_scan", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="", help="the finished run folder (absolute path or a folder name under the server's out_root); "
                                              "default = the latest run on the server")
    ap.add_argument("--measure-speed", type=float, default=None, metavar="MM_S",
                    help="the scan line's measurement speed [mm/s]: any speed inside the calibration's noise-prior speed range "
                         "(eoat_kiro_01: 15..60 mm/s; the server refuses others before any motion); default = the run's measurement "
                         "speed (its passes; positioning moves stay at the platform limits)")
    ap.add_argument("--confirm", default="auto", choices=CONFIRM_CHOICES,
                    help="auto (default): only a joint-space move that is not a straight joint line asks; each: every move asks "
                         "(answer with ~/confirm_move)")
    ap.add_argument("--after-pass", type=_pass_number, default=0, metavar="K",
                    help="the pass just completed (1..n of the run's weld plan); default = final (after every pass)")
    ap.add_argument("--direction", default="", choices=("", "S->E", "E->S"),
                    help="scan direction along the run's measured seam (its S / E); default = the nearer end (the torch tip's distance "
                         "from the current pose)")
    ap.add_argument("--tool-orientation", default="pass", choices=("pass", "least-rotation"),
                    help="pass (default): the sensor / torch orientation of the run's sensing passes - the same view as the passes; "
                         "least-rotation: of the two torch sides the one nearer the current pose (e.g. to keep the torch cable's side "
                         "after a weld)")
    ap.add_argument("--dry-run", action="store_true", help="plan and print the line and the moves; nothing is sent")
    ap.add_argument("--action", default=ACTION)
    ap.add_argument("--server-timeout", type=float, default=30.0)
    ap.add_argument("--log-topic", default="",
                    help="server log topic (std_msgs/String); '' = find it, 'none' = use the action feedback only")
    return ap


def goal_from_args(a, *, cwd: Optional[Path] = None):
    """Parsed arguments → PostWeldScan.Goal. --run: an existing local folder is sent as its absolute path (the server runs on this PC),
    anything else as given (a folder name under the server's out_root; the server refuses what it cannot find). --measure-speed:
    ValueError for a speed that is not finite and > 0 (auto_teach_client.measure_speed_mps); the calibrated range is the server's
    check (the calibration file is on the server)."""
    from hongik_scan_msgs.action import PostWeldScan
    g = PostWeldScan.Goal()
    run = str(a.run or "").strip()
    if run:
        p = Path(run).expanduser()
        if not p.is_absolute() and cwd is not None:
            p = Path(cwd) / p
        g.run_dir = str(p.resolve()) if p.is_dir() else run
    else:
        g.run_dir = ""
    g.measure_speed_mps = measure_speed_mps(a.measure_speed)
    g.confirm_mode = str(a.confirm)
    g.dry_run = bool(a.dry_run)
    g.after_pass = int(getattr(a, "after_pass", 0) or 0)
    g.direction = str(getattr(a, "direction", "") or "")
    g.tool_orientation = str(getattr(a, "tool_orientation", "pass") or "pass").replace("-", "_")
    return g


def goal_line(g) -> str:
    return (f"post-weld scan goal: run {g.run_dir or '(the latest on the server)'}, measurement speed "
            + (f"{g.measure_speed_mps * 1e3:g} mm/s (override)" if g.measure_speed_mps > 0 else "the run's measurement speed")
            + f", {f'after pass {g.after_pass}' if g.after_pass > 0 else 'final (after every pass)'}"
            + f", direction {g.direction or 'the nearer end'}, tool orientation {g.tool_orientation or 'pass'}"
            + f", confirm {g.confirm_mode}"
            + (", DRY RUN (nothing sent)" if g.dry_run else "") + " (Ctrl-C = cancel = stop the scan)")


def result_lines(res) -> list:
    """The action result → the terminal lines."""
    out = [f"RESULT: {res.status} - {res.message}"]
    if str(res.status) == "ABORTED":
        out.append("  The scan was stopped and the robot was commanded to stop; nothing more was sent. Check the cell, then send the "
                   "next goal - no reset is needed.")
    if res.run_dir:
        out.append(f"  input run: {res.run_dir}")
    if res.out_dir:
        out.append(f"  output: {res.out_dir} (raw.npz, run_log.json)")
    return out


def main(argv=None, *, say=None) -> int:
    say = say or (lambda s_: print(s_, flush=True))
    a = build_parser().parse_args(argv)
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import String
    from hongik_scan_msgs.action import PostWeldScan

    try:
        goal = goal_from_args(a, cwd=Path.cwd())
    except ValueError as e:
        say(f"REFUSED: {e}")
        return 2
    if not rclpy.ok():
        # Ctrl-C = KeyboardInterrupt with the context alive (the cancel must still reach the server; auto_teach_client 2026-10-08)
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("hongik_post_weld_scan")
    client = ActionClient(node, PostWeldScan, a.action)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    rc = 1
    try:
        if not client.wait_for_server(timeout_sec=a.server_timeout):
            say(f"REFUSED: action server {a.action} not available (is the launch with this tree's overlay running?)")
            return 1
        topic = None
        if str(a.log_topic).strip().lower() != "none":
            cands = [a.log_topic] if a.log_topic else log_topic_candidates(a.action)
            t_end = time.monotonic() + LOG_DISCOVERY_S
            while topic is None:
                topic = next((t for t in cands if node.count_publishers(t) > 0), None)
                if topic is not None or time.monotonic() >= t_end:
                    break
                time.sleep(0.05)
        pr = ProgressPrinter(say, use_log=topic is not None)
        if topic is not None:
            qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1000, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE)
            node.create_subscription(String, topic, lambda m: pr.on_log(m.data), qos)
        say(goal_line(goal) + "; progress " + (f"from {topic}" if topic is not None else "from the action feedback"))
        fut = client.send_goal_async(goal, feedback_callback=lambda fb: pr.on_feedback(fb.feedback.progress_text,
                                                                                        bool(fb.feedback.awaiting_confirmation)))
        try:
            while not fut.done():
                time.sleep(0.05)
        except KeyboardInterrupt:
            if _wait(fut, CANCEL_ANSWER_S) and fut.result() is not None and fut.result().accepted:
                gh = fut.result()
                _cancel_goal(gh, gh.get_result_async())
            else:
                say("\nCtrl-C: the goal was not accepted (nothing to cancel)")
            return 1
        gh = fut.result()
        if gh is None or not gh.accepted:
            say("REFUSED: the goal was not accepted by the server (see the server log)")
            return 1
        res_fut = gh.get_result_async()
        try:
            while not res_fut.done():
                time.sleep(0.1)
        except KeyboardInterrupt:
            if not _cancel_goal(gh, res_fut):
                return 1
        if pr.use_log:
            # the log topic and the result arrive separately: wait briefly for the server's result line
            t_end = time.monotonic() + LOG_DRAIN_S
            while not str(pr.last_log or "").startswith("post_weld_scan result: ") and time.monotonic() < t_end:
                time.sleep(0.05)
        res = res_fut.result().result
        for line in result_lines(res):
            say(line)
        rc = 0 if str(res.status) in OK_STATUSES else 1
    finally:
        executor.shutdown()
        spin.join(timeout=5.0)
        client.destroy()
        node.destroy_node()
        rclpy.try_shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
