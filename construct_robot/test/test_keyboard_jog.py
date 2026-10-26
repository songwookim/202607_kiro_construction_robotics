"""Keyboard jog streaming: kinematics and the per-arm stream cycle."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from construct_robot.core.keyboard_jog_kinematics import (
    JOG_HALTED_AT_JOINT_LIMIT,
    JOG_HALTED_AT_SINGULARITY,
    JOG_OK,
    ArmChain,
    TwistRamp,
    jog_step,
)
from construct_robot.nodes.keyboard_jog_node import ArmJog

URDF_PATH = (
    Path(__file__).resolve().parents[2]
    / "construct_description" / "urdf_0528" / "construct_robot_0528_hongik_v4.urdf"
)
# A measured right-arm teaching pose (away from singularities).
Q_RIGHT = np.array([5.1067, 1.0121, 2.0954, 3.5405, 1.1626, 1.4709])


@pytest.fixture(scope="module")
def chain():
    return ArmChain(
        URDF_PATH.read_text(encoding="utf-8"),
        "right_manipulator_base_link",
        "right_manipulator_ee_point",
    )


def test_chain_has_the_six_arm_joints_in_order(chain):
    assert chain.joint_names == [f"right_manipulator_joint{i}" for i in range(1, 7)]
    assert np.all(chain.velocity_limits > 0)


def test_jacobian_matches_numerical_fk_at_the_tcp(chain):
    position, _ = chain.fk(Q_RIGHT)
    jacobian = chain.jacobian(Q_RIGHT)
    for index in range(6):
        dq = np.zeros(6)
        dq[index] = 1e-6
        numeric = (chain.fk(Q_RIGHT + dq)[0] - position) / 1e-6
        assert np.allclose(jacobian[:3, index], numeric, atol=1e-5)


@pytest.mark.parametrize("twist", [
    (0.015, 0, 0, 0, 0, 0), (0, -0.09, 0, 0, 0, 0), (0, 0, 0.045, 0, 0, 0),
])
def test_linear_jog_step_moves_the_tcp_exactly(chain, twist):
    dq, status = jog_step(chain, Q_RIGHT, twist, 0.032)
    assert status == JOG_OK
    moved = chain.fk(Q_RIGHT + dq)[0] - chain.fk(Q_RIGHT)[0]
    # First-order step: ~4 um curvature error at 90 mm/s, re-linearized each cycle.
    assert np.allclose(moved, np.array(twist[:3]) * 0.032, atol=1e-5)


def test_rotation_jog_keeps_the_tcp_in_place(chain):
    dq, status = jog_step(chain, Q_RIGHT, (0, 0, 0, 0, 0.1, 0), 0.032)
    assert status == JOG_OK and np.any(dq)
    moved = chain.fk(Q_RIGHT + dq)[0] - chain.fk(Q_RIGHT)[0]
    assert np.linalg.norm(moved) < 5e-6


def test_joint_velocity_limit_scales_the_whole_step(chain):
    dq, _ = jog_step(chain, Q_RIGHT, (0, 0, 5.0, 0, 0, 0), 0.032, velocity_scale=0.5)
    assert np.max(np.abs(dq) / (chain.velocity_limits * 0.5 * 0.032)) == pytest.approx(1.0)
    moved = chain.fk(Q_RIGHT + dq)[0] - chain.fk(Q_RIGHT)[0]
    assert moved[2] > 0 and abs(moved[0]) < 0.05 * moved[2]  # direction kept


def test_joint_limit_blocks_only_motion_further_into_the_limit(chain):
    q = Q_RIGHT.copy()
    q[0] = chain.upper[0] - 0.04  # inside the 0.05 rad margin
    dq, _ = jog_step(chain, q, (0.02, 0, 0, 0, 0, 0), 0.032)
    status_plus = jog_step(chain, q, (0.02, 0, 0, 0, 0, 0), 0.032)[1]
    status_minus = jog_step(chain, q, (-0.02, 0, 0, 0, 0, 0), 0.032)[1]
    assert JOG_HALTED_AT_JOINT_LIMIT in (status_plus, status_minus)
    assert status_plus != status_minus


def test_singular_pose_halts(chain):
    dq, status = jog_step(chain, Q_RIGHT, (0, 0, 0.01, 0, 0, 0), 0.032, stop_sigma=10.0)
    assert status == JOG_HALTED_AT_SINGULARITY and not np.any(dq)


def test_ramp_limits_linear_and_angular_acceleration():
    ramp = TwistRamp(linear_accel=0.5, angular_accel=3.0)
    first = ramp.step((0.09, 0, 0, 0, 0, 1.0), 0.032)
    assert first[0] == pytest.approx(0.016) and first[5] == pytest.approx(0.096)
    for _ in range(20):
        ramp.step((0.09, 0, 0, 0, 0, 1.0), 0.032)
    assert np.allclose(ramp.current, (0.09, 0, 0, 0, 0, 1.0))
    stop = ramp.step(np.zeros(6), 0.032)
    assert stop[0] == pytest.approx(0.09 - 0.016)


# ----------------------------------------------------------- stream cycle
class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        clock = self

        class Stamp:
            nanoseconds = int(clock.t * 1e9)

            def __sub__(self, other):
                return SimpleNamespace(nanoseconds=self.nanoseconds - other.nanoseconds)

            def to_msg(self):
                return None
        return Stamp()


def arm_jog(chain):
    params = {
        "max_linear_accel": 0.5, "max_angular_accel": 3.0, "command_timeout_s": 0.2,
        "joint_velocity_scale": 0.5, "joint_limit_margin_rad": 0.05,
        "singularity_slow_sigma": 0.05, "singularity_stop_sigma": 0.015,
        "reference_max_age_s": 0.5, "cycle_s": 0.032,
        "stall_stop_s": 0.1, "reference_deviation_rad": 0.003,
    }
    clock = Clock()
    node = SimpleNamespace(
        get_parameter=lambda name: SimpleNamespace(value=params[name]),
        get_clock=lambda: clock,
        get_logger=lambda: Mock(),
        create_publisher=Mock(side_effect=lambda *a, **k: Mock()),
        create_subscription=Mock(),
        create_service=Mock(),
    )
    jog = ArmJog(node, "right", chain)
    jog._controller_state(SimpleNamespace(
        joint_names=list(chain.joint_names),
        reference=SimpleNamespace(positions=list(Q_RIGHT)),
        desired=SimpleNamespace(positions=[]),
    ))
    return jog, clock


def twist_message(z, frame="right_manipulator_base_link"):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame),
        twist=SimpleNamespace(
            linear=SimpleNamespace(x=0.0, y=0.0, z=z),
            angular=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        ),
    )


def test_stream_reaches_the_commanded_speed_exactly(chain):
    jog, clock = arm_jog(chain)
    assert jog._start(None, SimpleNamespace()).success
    start = chain.fk(jog.q_cmd)[0]
    for _ in range(int(10.0 / 0.032)):  # 10 s held at 15 mm/s
        jog._twist(twist_message(0.015))
        clock.t += 0.032
        jog.tick(0.032)
    travelled = chain.fk(jog.q_cmd)[0][2] - start[2]
    # 0.5 m/s^2 reaches 15 mm/s within the first cycle: full commanded speed.
    # The press itself is integrated at once: one step more than the ticks.
    assert travelled == pytest.approx(0.015 * (int(10.0 / 0.032) + 1) * 0.032, abs=3e-4)


def test_stream_decelerates_then_holds_after_the_command_times_out(chain):
    jog, clock = arm_jog(chain)
    jog._start(None, SimpleNamespace())
    for _ in range(40):
        jog._twist(twist_message(0.045))
        clock.t += 0.032
        jog.tick(0.032)
    published = jog.trajectory_publisher.publish
    for _ in range(40):  # no more commands
        clock.t += 0.032
        jog.tick(0.032)
    last = published.call_args.args[0]
    assert len(last.points) == 2 and not any(last.points[-1].velocities)
    assert not jog.moving
    calls = published.call_count
    clock.t += 0.032
    jog.tick(0.032)
    assert published.call_count == calls  # idle: JTC holds, nothing streamed


def test_stamped_segment_starts_on_the_executing_spline(chain):
    jog, clock = arm_jog(chain)
    jog._start(None, SimpleNamespace())
    clock.t = 5.0
    jog._twist(twist_message(0.045))   # the press is streamed at once
    message = jog.trajectory_publisher.publish.call_args.args[0]
    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (5, 0)
    now, target, extended, stopped = message.points
    assert np.allclose(now.positions, Q_RIGHT) and not any(now.velocities)
    assert np.allclose(target.positions, jog.q_cmd)
    assert np.allclose(np.array(extended.positions) - target.positions,
                       np.array(target.velocities) * 0.032)
    assert np.allclose(extended.velocities, target.velocities) and not any(stopped.velocities)
    assert [p.time_from_start.nanosec for p in message.points] == [0, 32000000, 64000000, 164000000]


def test_twist_in_another_frame_is_ignored(chain):
    jog, clock = arm_jog(chain)
    jog._start(None, SimpleNamespace())
    jog._twist(twist_message(0.045, frame="World"))
    clock.t += 0.032
    jog.tick(0.032)
    jog.trajectory_publisher.publish.assert_not_called()


def test_start_requires_a_fresh_jtc_reference(chain):
    jog, clock = arm_jog(chain)
    clock.t += 1.0
    response = jog._start(None, SimpleNamespace())
    assert not response.success and not jog.active


def stamped_twist(z, stamp):
    from construct_robot.nodes.keyboard_jog_node import _stamp

    message = twist_message(z)
    message.header.stamp = _stamp(stamp)
    return message


def test_delayed_press_does_not_restart_a_released_jog(chain):
    jog, clock = arm_jog(chain)
    jog._start(None, SimpleNamespace())
    clock.t = 1.0
    jog._twist(stamped_twist(0.015, 0.99))
    clock.t = 1.02
    jog._twist(stamped_twist(0.0, 1.01))
    clock.t = 1.03
    jog._twist(stamped_twist(0.015, 1.0))
    assert not np.any(jog.twist)
    assert jog.last_twist_stamp == pytest.approx(1.01)


@pytest.mark.parametrize("stamp", [0.7, 1.1])
def test_expired_or_future_twist_cannot_start_motion(chain, stamp):
    jog, clock = arm_jog(chain)
    jog._start(None, SimpleNamespace())
    clock.t = 1.0
    jog._twist(stamped_twist(0.015, stamp))
    jog.trajectory_publisher.publish.assert_not_called()
    assert not np.any(jog.twist)


def test_delayed_heartbeat_keeps_original_expiry(chain):
    jog, clock = arm_jog(chain)
    jog._start(None, SimpleNamespace())
    clock.t = 1.0
    jog._twist(stamped_twist(0.015, 0.85))
    assert jog.moving
    clock.t = 1.08
    jog.tick(0.032)
    assert not jog.moving
    assert not any(jog.trajectory_publisher.publish.call_args.args[0].points[-1].velocities)


def test_reenable_discards_commands_sent_before_the_new_session(chain):
    jog, clock = arm_jog(chain)
    clock.t = 0.1
    jog._start(None, SimpleNamespace())
    jog._twist(stamped_twist(0.015, 0.09))
    jog.trajectory_publisher.publish.assert_not_called()


def test_jog_stream_queues_only_latest_command_and_reference(chain):
    jog, _clock = arm_jog(chain)
    publishers = jog.node.create_publisher.call_args_list
    assert next(call.args[2] for call in publishers if call.args[1].endswith("/joint_trajectory")) == 1
    assert all(call.args[3] == 1 for call in jog.node.create_subscription.call_args_list)


# ------------------------------------------- stream through a modelled JTC
class JtcTopicModel:
    """joint_trajectory_controller 2.x topic interface (open_loop_control, splines).

    Mirrors Trajectory::sample(): an unstamped message starts at its first
    sample from the previous command; a stamped one is sampled inside its
    points.  Segments with velocities are cubic Hermite; past the last point
    the last point is held.
    """

    PERIOD = 0.008  # 125 Hz controller_manager

    def __init__(self, q):
        self.position = np.array(q, dtype=float)
        self.velocity = np.zeros_like(self.position)
        self.message = None
        self.inbox = []
        self.history = []

    def deliver(self, arrival, message):
        self.inbox.append((arrival, message))

    def update(self, time):
        arrived = [m for t, m in self.inbox if t <= time]
        self.inbox = [(t, m) for t, m in self.inbox if t > time]
        if arrived:
            self.message = arrived[-1]
            stamp = self.message.header.stamp
            self.start = stamp.sec + stamp.nanosec * 1e-9 if (stamp.sec or stamp.nanosec) else time
            self.before = (time, self.position.copy(), self.velocity.copy())
        if self.message is not None:
            self.position, self.velocity = self._sample(time)
        self.history.append((time, self.position.copy()))

    def _sample(self, time):
        points = self.message.points
        times = [self.start + p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in points]
        if time < times[0]:
            t0, q0, v0 = self.before
            return self._cubic(t0, q0, v0, times[0], points[0], time)
        for t0, t1, a, b in zip(times, times[1:], points, points[1:]):
            if t0 <= time < t1:
                return self._cubic(t0, np.array(a.positions), np.array(a.velocities), t1, b, time)
        return np.array(points[-1].positions), np.array(points[-1].velocities)

    @staticmethod
    def _cubic(t0, q0, v0, t1, point, time):
        q1, v1, T, s = np.array(point.positions), np.array(point.velocities), t1 - t0, time - t0
        c2 = (-3 * q0 + 3 * q1 - 2 * v0 * T - v1 * T) / T ** 2
        c3 = (2 * q0 - 2 * q1 + v0 * T + v1 * T) / T ** 3
        return q0 + v0 * s + c2 * s ** 2 + c3 * s ** 3, v0 + 2 * c2 * s + 3 * c3 * s ** 2


def run_stream(chain, schedule, duration, jitter=0.0, latency=0.0015, stall=None, seed=3):
    """Drive ArmJog and the JTC model on one clock.

    schedule: [(time, z_speed)] twist commands from the GUI.  The jog timer
    ticks every cycle_s (+ uniform jitter), the GUI twist is re-sent at 50 Hz.
    """
    jog, clock = arm_jog(chain)
    jog.node.get_parameter = lambda name, p=dict(
        max_linear_accel=1.5, max_angular_accel=6.0, command_timeout_s=0.2,
        joint_velocity_scale=0.5, joint_limit_margin_rad=0.05,
        singularity_slow_sigma=0.05, singularity_stop_sigma=0.015,
        reference_max_age_s=0.5, cycle_s=0.016, stall_stop_s=0.1,
        reference_deviation_rad=0.003): SimpleNamespace(value=p[name])
    jtc = JtcTopicModel(Q_RIGHT)
    jog.trajectory_publisher.publish = lambda message: jtc.deliver(clock.t + latency, message)
    assert jog._start(None, SimpleNamespace()).success
    rng = np.random.default_rng(seed)
    events = [(t, "twist", z) for t, z in schedule]
    events += [(t, "gui", None) for t in np.arange(0.0, duration, 0.02)]
    tick = 0.0
    while tick < duration:
        if stall is None or not stall[0] <= tick < stall[1]:
            events.append((tick, "tick", None))
        tick += 0.016 + rng.uniform(0.0, jitter)
    events += [(t, "jtc", None) for t in np.arange(0.0, duration, JtcTopicModel.PERIOD)]
    events.sort(key=lambda e: (e[0], {"twist": 0, "gui": 1, "tick": 2, "jtc": 3}[e[1]]))
    current = 0.0
    for time, kind, value in events:
        clock.t = time
        if kind == "twist":
            current = value
            jog._twist(twist_message(current))
        elif kind == "gui":
            if current:
                jog._twist(twist_message(current))
        elif kind == "tick":
            jog.tick(0.016)
        else:
            jtc.update(time)
    times = np.array([t for t, _ in jtc.history])
    tcp = np.array([chain.fk(q)[0] for _, q in jtc.history])
    return jog, jtc, times, tcp


def tcp_speed(times, tcp):
    return np.linalg.norm(np.diff(tcp, axis=0), axis=1) / np.diff(times)


def test_held_key_commands_every_cycle_without_message_dips(chain):
    _jog, _jtc, times, tcp = run_stream(chain, [(0.1, 0.03)], 2.0, jitter=0.004)
    steady = tcp_speed(times, tcp)[(times[1:] > 0.4) & (times[1:] < 1.9)]
    # Unstamped streaming held one 8 ms cycle per message (speed 0, then 1.33x).
    assert steady.min() == pytest.approx(0.03, rel=0.01)
    assert steady.max() == pytest.approx(0.03, rel=0.01)


def test_held_key_travels_at_the_exact_speed(chain):
    _jog, _jtc, times, tcp = run_stream(chain, [(0.1, 0.05), (2.1, 0.0)], 2.6, jitter=0.004)
    travelled = tcp[-1][2] - tcp[0][2]
    # 2.0 s at 50 mm/s; the 33 ms ramps up and down cancel.
    assert travelled == pytest.approx(0.05 * 2.0, abs=0.5e-3)
    assert np.all(tcp_speed(times, tcp)[times[1:] > 2.3] < 1e-6)  # stopped, holding


def test_press_reaches_the_jtc_within_one_controller_cycle(chain):
    _jog, _jtc, times, tcp = run_stream(chain, [(0.1003, 0.05)], 0.3)
    moving = times[1:][tcp_speed(times, tcp) > 1e-5]
    # press 100.3 ms + 1.5 ms transport -> JTC cycle at 104 ms starts moving.
    assert moving[0] <= 0.1003 + 0.0015 + JtcTopicModel.PERIOD + 1e-9


def test_release_stops_within_the_ramp(chain):
    _jog, _jtc, times, tcp = run_stream(chain, [(0.1, 0.05), (1.0, 0.0)], 1.5)
    moving = times[1:][tcp_speed(times, tcp) > 1e-5]
    # 50 mm/s at 1.5 m/s^2: 33 ms ramp after the release reaches the JTC.
    assert moving[-1] < 1.0 + 0.0015 + 0.034 + 2 * 0.016 + JtcTopicModel.PERIOD


def test_stalled_stream_slows_down_smoothly_then_resumes(chain):
    _jog, _jtc, times, tcp = run_stream(chain, [(0.1, 0.05)], 1.5, stall=(0.6, 0.75))
    velocity = np.diff(tcp, axis=0) / np.diff(times)[:, None]
    acceleration = np.linalg.norm(np.diff(velocity, axis=0), axis=1) / JtcTopicModel.PERIOD
    # No jump back to speed after the 150 ms gap: the ramp restarts from the
    # twist the JTC actually commands (1.5 m/s^2 ramp, Hermite blends ~3x).
    assert acceleration.max() < 6.0
    assert tcp_speed(times, tcp)[times[1:] > 1.0].min() == pytest.approx(0.05, rel=0.01)


def test_stream_restarts_from_a_jtc_hold_instead_of_jumping_back(chain):
    jog, jtc, times, tcp = run_stream(chain, [(0.1, 0.05)], 0.5)
    held = jtc.position.copy() + 0.01  # e.g. a tolerance hold somewhere else
    jtc.message, jtc.position, jtc.velocity = None, held, np.zeros(6)
    jog._controller_state(SimpleNamespace(
        joint_names=list(chain.joint_names),
        reference=SimpleNamespace(positions=list(held)),
        desired=SimpleNamespace(positions=[]),
        header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=510_000_000)),
    ))
    jog.node.get_clock().t = 0.52
    jog.tick(0.016)
    message = jtc.inbox[-1][1]
    assert np.allclose(message.points[0].positions, held) and not any(message.points[0].velocities)


def test_stream_timer_runs_only_while_an_arm_is_in_keyboard_teaching(chain):
    jog, clock = arm_jog(chain)
    jog.node._sync_timer = Mock()
    assert jog._start(None, SimpleNamespace()).success
    jog._stop(None, SimpleNamespace())
    assert jog.node._sync_timer.call_count == 2


def test_node_timer_follows_arm_activity():
    from construct_robot.nodes.keyboard_jog_node import KeyboardJogNode

    class Timer:
        canceled = False

        def is_canceled(self):
            return self.canceled

        def cancel(self):
            self.canceled = True

        def reset(self):
            self.canceled = False

    node = object.__new__(KeyboardJogNode)
    node.timer = Timer()
    node.arms = {"right": SimpleNamespace(active=False), "left": SimpleNamespace(active=False)}
    KeyboardJogNode._sync_timer(node)
    assert node.timer.canceled
    node.arms["right"].active = True
    KeyboardJogNode._sync_timer(node)
    assert not node.timer.canceled
