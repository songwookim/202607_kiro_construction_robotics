"""Cartesian keyboard-jog kinematics: URDF arm chain, TCP Jacobian, jog step.

Twists are expressed in the arm base frame, with rotation about the TCP (the
chain tip), which is what the GUI's keyboard mapping produces.
"""

import math

import numpy as np
import PyKDL as kdl
from urdf_parser_py.urdf import URDF


JOG_OK = 0
JOG_SLOWED_NEAR_SINGULARITY = 1
JOG_HALTED_AT_SINGULARITY = 2
JOG_HALTED_AT_JOINT_LIMIT = 5


def _origin_frame(origin):
    xyz = origin.xyz if origin is not None and origin.xyz is not None else (0.0, 0.0, 0.0)
    rpy = origin.rpy if origin is not None and origin.rpy is not None else (0.0, 0.0, 0.0)
    return kdl.Frame(kdl.Rotation.RPY(*rpy), kdl.Vector(*xyz))


class ArmChain:
    """Serial chain from base_link to tip_link built from a URDF string."""

    def __init__(self, urdf_xml, base_link, tip_link):
        # xacro output starts with an encoding declaration, which lxml only
        # accepts from bytes.
        if isinstance(urdf_xml, str):
            urdf_xml = urdf_xml.encode("utf-8")
        robot = URDF.from_xml_string(urdf_xml)
        by_child = {joint.child: joint for joint in robot.joints}
        joints = []
        link = tip_link
        while link != base_link:
            if link not in by_child:
                raise ValueError(f"{tip_link} is not below {base_link} in the URDF")
            joints.append(by_child[link])
            link = by_child[link].parent
        joints.reverse()

        self.chain = kdl.Chain()
        self.joint_names = []
        lower, upper, velocity = [], [], []
        for joint in joints:
            frame = _origin_frame(joint.origin)
            if joint.type in ("revolute", "continuous"):
                axis = frame.M * kdl.Vector(*(joint.axis or (0.0, 0.0, 1.0)))
                kdl_joint = kdl.Joint(joint.name, frame.p, axis, kdl.Joint.RotAxis)
                self.joint_names.append(joint.name)
                limit = joint.limit
                continuous = joint.type == "continuous" or limit is None
                lower.append(-math.inf if continuous else float(limit.lower))
                upper.append(math.inf if continuous else float(limit.upper))
                velocity.append(float(limit.velocity) if limit is not None and limit.velocity else math.inf)
            elif joint.type == "fixed":
                kdl_joint = kdl.Joint(joint.name, kdl.Joint.Fixed)
            else:
                raise ValueError(f"unsupported joint type {joint.type} ({joint.name})")
            self.chain.addSegment(kdl.Segment(joint.child, kdl_joint, frame))
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.velocity_limits = np.array(velocity)
        self._fk = kdl.ChainFkSolverPos_recursive(self.chain)
        self._jacobian = kdl.ChainJntToJacSolver(self.chain)

    def _array(self, q):
        array = kdl.JntArray(len(self.joint_names))
        for index, value in enumerate(q):
            array[index] = float(value)
        return array

    def fk(self, q):
        """TCP position (m) and rotation matrix in the base frame."""
        frame = kdl.Frame()
        self._fk.JntToCart(self._array(q), frame)
        position = np.array([frame.p[i] for i in range(3)])
        rotation = np.array([[frame.M[i, j] for j in range(3)] for i in range(3)])
        return position, rotation

    def jacobian(self, q):
        """6xN Jacobian in the base frame, reference point at the TCP."""
        jac = kdl.Jacobian(len(self.joint_names))
        self._jacobian.JntToJac(self._array(q), jac)
        return np.array([[jac[i, j] for j in range(jac.columns())] for i in range(6)])


class TwistRamp:
    """Limit the linear and angular acceleration of a commanded twist."""

    def __init__(self, linear_accel, angular_accel):
        self.linear_accel = float(linear_accel)
        self.angular_accel = float(angular_accel)
        self.current = np.zeros(6)

    def reset(self):
        self.current = np.zeros(6)

    def step(self, target, dt):
        target = np.asarray(target, dtype=float)
        change = target - self.current
        for part, limit in ((slice(0, 3), self.linear_accel), (slice(3, 6), self.angular_accel)):
            norm = np.linalg.norm(change[part])
            allowed = limit * dt
            if norm > allowed:
                change[part] *= allowed / norm
        self.current = self.current + change
        return self.current.copy()


def jog_step(
    chain,
    q,
    twist,
    dt,
    *,
    velocity_scale=0.5,
    limit_margin=0.05,
    slow_sigma=0.03,
    stop_sigma=0.008,
):
    """Joint increment for a base-frame TCP twist over dt.

    Damped least squares near singularities, joint-velocity scaling that keeps
    the Cartesian direction, and a hard stop for joints driven further into
    their limit margin.  Returns (dq, status).
    """
    q = np.asarray(q, dtype=float)
    twist = np.asarray(twist, dtype=float)
    if not np.any(twist):
        return np.zeros_like(q), JOG_OK
    jacobian = chain.jacobian(q)
    u, sigma, vt = np.linalg.svd(jacobian, full_matrices=False)
    sigma_min = float(sigma[-1])
    if sigma_min < stop_sigma:
        return np.zeros_like(q), JOG_HALTED_AT_SINGULARITY
    status = JOG_OK
    damping = 0.0
    if sigma_min < slow_sigma:
        status = JOG_SLOWED_NEAR_SINGULARITY
        damping = (1.0 - (sigma_min / slow_sigma) ** 2) * slow_sigma ** 2
    gains = sigma / (sigma ** 2 + damping)
    dq = vt.T @ (gains * (u.T @ twist)) * dt

    limits = chain.velocity_limits * velocity_scale * dt
    ratio = float(np.max(np.abs(dq) / limits))
    if ratio > 1.0:
        dq /= ratio

    target = q + dq
    deeper = ((target > chain.upper - limit_margin) & (dq > 0)) | (
        (target < chain.lower + limit_margin) & (dq < 0)
    )
    if np.any(deeper):
        return np.zeros_like(q), JOG_HALTED_AT_JOINT_LIMIT
    return dq, status
