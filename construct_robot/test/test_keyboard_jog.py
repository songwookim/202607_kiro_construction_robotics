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
    / "construct_description" / "urdf_0528" / "construct_robot_0528.urdf"
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
    assert travelled == pytest.approx(0.015 * int(10.0 / 0.032) * 0.032, abs=3e-4)


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
    assert len(last.points) == 1 and not any(last.points[0].velocities)
    assert not jog.moving
    calls = published.call_count
    clock.t += 0.032
    jog.tick(0.032)
    assert published.call_count == calls  # idle: JTC holds, nothing streamed


def test_segments_start_at_the_integrated_command_with_one_extrapolated_point(chain):
    jog, clock = arm_jog(chain)
    jog._start(None, SimpleNamespace())
    jog._twist(twist_message(0.045))
    clock.t += 0.032
    jog.tick(0.032)
    message = jog.trajectory_publisher.publish.call_args.args[0]
    first, second = message.points
    assert np.allclose(first.positions, jog.q_cmd)
    assert np.allclose(np.array(second.positions) - first.positions,
                       np.array(first.velocities) * 0.032)
    assert (first.time_from_start.nanosec, second.time_from_start.nanosec) == (32000000, 64000000)


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
