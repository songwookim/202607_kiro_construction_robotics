"""Explicitly authorized +World-Z 10mm test; uses GUI's own motion APIs.

Requires an attended, cleared workspace and a left-only real hardware launch.
Never starts unless --execute-attended is passed. Does not activate robot power.
"""
import argparse
import json
import math
import threading
import time
from pathlib import Path

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rbpodo_msgs.msg import SystemState
from control_msgs.msg import JointTrajectoryControllerState
from construct_robot.weld_action_gui import WeldGuiNode, WeldActionGui, tip_link_for_group


class UI:
    _pose_values = staticmethod(WeldActionGui._pose_values)

    def __getattr__(self, name):
        return lambda *args: None

    def post(self, callback, *args):
        callback(*args)

    def log(self, *args):
        print(*args, flush=True)

    error = log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-attended', action='store_true')
    args = parser.parse_args()
    if not args.execute_attended:
        parser.error('Requires attended workspace confirmation and --execute-attended')
    rclpy.init()
    node = WeldGuiNode(UI())
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    path = Path('/home/irs/ros2_ws/weld_feedback') / time.strftime('left_jog_transition_%Y%m%d_%H%M%S.jsonl')
    path.parent.mkdir(exist_ok=True)
    stream = path.open('x')
    lock = threading.Lock()
    latest = {}
    phase = 'baseline'
    origin = None
    abort = threading.Event()
    monitor_done = threading.Event()
    enabled = False

    def record(kind, **values):
        with lock:
            stream.write(json.dumps(dict(time=time.time(), monotonic=time.monotonic(), phase=phase, kind=kind, **values), default=float) + '\n')

    def system(msg):
        latest['message'], latest['time'] = msg, time.monotonic()
        record('rb', q_actual=list(msg.jnt_ang), q_ref=list(msg.jnt_ref), joint_current=list(msg.jnt_cur),
               tcp=list(msg.tcp_pos), robot_state=msg.robot_state, task_state=msg.task_state,
               active=msg.init_state_info, speed=msg.default_speed)

    def jtc(msg):
        record('jtc', q_actual=list(msg.actual.positions), q_cmd=list(msg.desired.positions),
               v_cmd=list(msg.desired.velocities))

    node.create_subscription(SystemState, '/left_rbpodo_hardware/system_state', system, 50)
    node.create_subscription(JointTrajectoryControllerState, '/left_manipulator_controller/controller_state', jtc, 50)

    def check():
        msg = latest.get('message')
        if msg is None or time.monotonic() - latest['time'] > .3:
            raise RuntimeError('Robot feedback unavailable/stale')
        if (msg.init_state_info != 6 or msg.real_vs_simulation_mode or msg.is_freedrive_mode
                or msg.op_stat_collision_occur or msg.op_stat_self_collision or msg.op_stat_soft_estop_occur
                or msg.op_stat_sos_flag or msg.op_stat_ems_flag):
            raise RuntimeError('Robot not ready or safety fault; no automatic reset')
        pose = node._current_tcp_pose('left_manipulator')
        if origin is not None:
            delta = [getattr(pose.position, a) - getattr(origin.position, a) for a in 'xyz']
            if math.hypot(*delta[:2]) > .002 or not -.002 <= delta[2] <= .018:
                raise RuntimeError(f'Outside test corridor: {delta}')
            record('tcp', delta_m=delta)
        return pose

    def monitor():
        while not monitor_done.wait(.02):
            try:
                check()
            except Exception as error:
                record('abort', error=str(error))
                abort.set()
                node.clear_keyboard_velocity()
                node.cancel_active_motion()
                break

    try:
        deadline = time.monotonic() + 12
        while (not latest or not node.tf_buffer.can_transform(
                'World', tip_link_for_group('left_manipulator'), rclpy.time.Time())) and time.monotonic() < deadline:
            time.sleep(.05)
        check()
        names, joints, origin, _ = node.capture_measured_teaching_snapshot('left_manipulator', 'diagnostic_origin')
        record('origin', q=list(joints), tcp=UI._pose_values(origin))
        threading.Thread(target=monitor, daemon=True).start()
        phase = 'enable_keyboard'
        ok, detail = node.set_keyboard_velocity_controller_enabled('left', True)
        enabled = ok
        if not ok:
            raise RuntimeError(detail)
        if abort.is_set():
            raise RuntimeError('Aborted before jog')
        phase = 'jog_positive_z'
        velocity = node.resolve_keyboard_velocity('left_manipulator', 'Z', 'Up', .005, .01, 'world')
        record('velocity_command', values=velocity)
        node.set_keyboard_velocity('left', velocity)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not abort.is_set():
            pose = check()
            if pose.position.z - origin.position.z >= .010:
                break
            node.refresh_keyboard_velocity('left')
            time.sleep(.02)
        phase = 'release'
        node.clear_keyboard_velocity()
        record('velocity_command', values=[0] * 6)
        if not node.wait_until_arm_stopped('left') or abort.is_set():
            raise RuntimeError('Jog stop not confirmed or safety monitor aborted')
        if check().position.z - origin.position.z < .008:
            raise RuntimeError('Jog moved less than 8 mm; teaching return would be too short to time reliably')
        phase = 'disable_keyboard'
        ok, detail = node.set_keyboard_velocity_controller_enabled('left', False)
        if not ok:
            raise RuntimeError(detail)
        enabled = False
        phase = 'teaching_return'
        step = dict(pose_name='weld_start', pose_label='Diagnostic original TCP',
                    planning_group='left_manipulator', joint_names=names, positions=joints,
                    tcp_pose=origin, velocity_scale=.05, tcp_speed_m_s=.005, touch_guard=False)
        if abort.is_set():
            raise RuntimeError('Aborted before return')
        ok, detail = node.run_sequence_named_pose(step, True)
        record('return_result', success=ok, detail=detail)
        if not ok or abort.is_set():
            raise RuntimeError(detail)
        phase = 'final_hold'
        if not node.wait_until_arm_stopped('left'):
            raise RuntimeError('Final standstill unconfirmed')
        final = check()
        error_mm = math.dist([getattr(final.position, a) for a in 'xyz'],
                             [getattr(origin.position, a) for a in 'xyz']) * 1000
        record('complete', final_error_mm=error_mm)
        print(f'COMPLETE error={error_mm:.3f}mm log={path}', flush=True)
    finally:
        node.clear_keyboard_velocity()
        if node.active_motion_goal is not None:
            node.cancel_active_motion()
        if enabled and node.wait_until_arm_stopped('left'):
            node.set_keyboard_velocity_controller_enabled('left', False)
        monitor_done.set()
        executor.shutdown(timeout_sec=3)
        spin.join(timeout=3)
        stream.close()
        node.destroy_node()
        rclpy.shutdown()
        print(f'DIAGNOSTIC LOG {path}', flush=True)


if __name__ == '__main__':
    main()
