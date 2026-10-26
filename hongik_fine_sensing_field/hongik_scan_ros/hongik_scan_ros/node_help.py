"""--help for the ROS 2 node entry points (they take ROS arguments, not argparse): `<node> -h | --help` prints this text and exits
before rclpy starts - no node, no sensor connection, no robot connection.
"""
# (한국어 설명, 코드 주석) 노드 실행 파일 (ros2 run … / console_scripts) 의 --help 를 영어로 (사용자 2026-10-10: 시스템이 출력하는 문구는
# 영어만). 노드는 rclpy 인자를 받아 argparse 가 없다 — '--ros-args' 앞의 -h/--help 만 보고, rclpy.init 전에 이 글을 찍고 끝낸다
# (도움말을 보려다 센서·로봇에 접속하는 일이 없게).
from __future__ import annotations

import sys
from typing import Optional, Sequence

__all__ = ["NODE_HELP", "print_node_help"]

NODE_HELP = {
    "hongik_profiler_node": """usage: ros2 run hongik_scan_ros hongik_profiler_node [--ros-args -p name:=value ...]

VC nano 3D-Z profiler -> ~/profile (hongik_scan_msgs/ProfileRaw, frame hongik_profiler_optical_frame). The node holds the sensor
only while it streams (one TCP client at a time).

services (std_srvs/Trigger):
  ~/start            connect, back up the sensor parameters (ID 44), apply the standard init, stream; succeeds on the first frame
  ~/stop             stop streaming (RESET), restore the original parameters, disconnect (release the sensor); the auto-teach
                     server calls it before a goal
  ~/backup_params    read ID 44 and write a backup file (backup_dir) - not while streaming
  ~/restore_params   restore the original parameters of this node's session
topics: ~/profile (ProfileRaw), ~/sensor_owned (std_msgs/Bool, transient local)

parameters:
  platform_config (str)          platform YAML (sensor ip / port / init parameters)
  sensor_ip (str, '')            '' = platform config sensor.ip
  sensor_port (int, 0)           0 = platform config sensor.port
  frame_id (str)                 hongik_profiler_optical_frame
  autostart (bool, false)        true = start streaming at once (like ~/start)
  backup_dir (str)               ~/scan_data/sensor_backup
  first_frame_timeout_s (2.0)    wait for the first frame on ~/start
  apply_init (bool, true)        apply the standard sensor init on ~/start
""",
    "hongik_auto_teach_server": """usage: ros2 run hongik_scan_ros hongik_auto_teach_server --ros-args -p platform_config:=<yaml> -p calib_file:=<yaml> ...
       (normally started by hongik_moveit.launch.py)

Action servers (one goal at a time): /hongik_fine_sensing/precision_sensing (welding_interfaces/action/PrecisionSensing, KIRO) and
/hongik_fine_sensing/auto_teach (hongik_scan_msgs/action/AutoTeach, console client). Both run the 1G core through the MoveIt adapter
(the platform's move_group + trajectory controller + /joint_states + TF); the start pose is the arm pose at goal acceptance and the
specimen design comes from job_file. The server owns the sensor during a run. Progress lines: the action feedback and the server log
topic (~/log). Confirmation prompts (confirm_mode / kiro_confirm_mode other than none) are answered with ~/confirm_move
(std_srvs/SetBool: true = run, false = abort); ~/status (std_srvs/Trigger) reports the state and the readiness (move_group, trajectory
controller, joint states, TF). Cancelling a goal stops our trajectory and ends that run; there is nothing to reset. The first command
comes from KIRO's main system (the goal); the server has no robot-controller settings or modes and moves the arm only through the
platform's move_group and trajectory controller. Outputs per run under out_root: sections.json/.npz, weld_plan.json/.npz,
kiro/ (KIRO-format pass_<k>.yaml in KIRO's World frame + pass_<k>_info.yaml information, not commands; platform config
kiro_output), precision_sensing.npz and precision_sensing_result.yaml (the full result message; KIRO goals), verification/.
Operator client of the KIRO action: hongik_precision_sensing (the two points by coordinates: --a X,Y,Z --b X,Y,Z or --points
FILE; ~/status names the base frame of the goal points).
Post-weld data acquisition: action /hongik_fine_sensing/post_weld_scan (hongik_scan_msgs/action/PostWeldScan; client
hongik_post_weld_scan) - after the welding or after any pass of it, one straight scan of the whole weld along a finished 1G run's
measured seam (20 mm beyond each measured end, the run's pass standoff and orientation, the measurement speed) ->
<run>/post_weld_scan_p<k>|final_<stamp>/ raw.npz + run_log.json. One goal at a time over all actions.

parameters (main):
  platform_config (str, required), calib_file (str, required), job_file (str), out_root (str)
  confirm_mode (each|large|scan|auto|none), kiro_confirm_mode (default none)
  sensor_ip, sensor_port, sensor_init_overrides, local_speed_mps, transit_speed_mps, descend_speed_mps, dwell_frames,
  probe_frames, motion_timeout_s, profiler_release_service, profile_publish_hz
  job_overrides (str, '')        site-level run-time job overrides for every goal of both actions (the way KIRO's site adjusts:
                                 its PrecisionSensing goal has no field): a YAML / JSON mapping of dotted paths into the job
                                 document, e.g. '{precision.end_position_mm: 0.2, weld_plan.stick_out_mm: 17.5,
                                 weld_design.end_inset_mm.goal: 79.7}'; checked at start (a bad key stops the server); applied right
                                 after the job file is loaded, the AutoTeach goal's job_overrides (client --set) on top; recorded in
                                 verification/run_log.json job_overrides and verification/job_effective.yaml. Blocks: job, specimen,
                                 qc_targets, precision, weld_plan, weld_design, auto_teach (scan_l0.job_overrides). A measurement
                                 speed chosen at run time (job.speed_mps, goal measure_speed_mps) must lie inside the calibration's
                                 noise-prior speed range (calib g1 noise_priors speed_range_mps)
""",
}


def print_node_help(name: str, argv: Optional[Sequence[str]] = None) -> bool:
    """argv (기본 sys.argv[1:]) 의 '--ros-args' 앞에 -h / --help 가 있으면 그 노드의 도움말을 찍고 True (호출자는 바로 끝낸다)."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--ros-args" in args:
        args = args[:args.index("--ros-args")]
    if not any(a in ("-h", "--help") for a in args):
        return False
    print(NODE_HELP[name], end="", flush=True)
    return True
