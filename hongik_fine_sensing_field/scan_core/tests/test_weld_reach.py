"""weld_reach: the elbow shell of a spherical-wrist arm (RB20) - exact: a pose is inside the shell iff its elbow value is within the
URDF limits (every IK branch of this arm has the same |elbow|)."""
import math
import os

import numpy as np
import pytest

from scan_core.weld_reach import arm_reach_from_chain

URDF = os.path.join(os.path.dirname(__file__), "..", "..", "scan_l0", "tests", "data", "rb20_1900es_u_eoat.urdf")


def test_rb20_elbow_shell_matches_the_elbow_limit():
    fk_urdf = pytest.importorskip("scan_core.fk_urdf")
    chain = fk_urdf.chain_from_urdf(URDF, "link0", "tcp")
    T_ft = np.eye(4)
    T_ft[:3, 3] = [0.0, 0.0465, 0.297]
    m = arm_reach_from_chain(chain, T_ft)
    assert m is not None and m.joint == "elbow"
    assert math.sqrt(m.rho2_lo) == pytest.approx(2 * 0.885 * math.cos(2.618 / 2.0), abs=1e-6)   # the URDF elbow limit 2.618 rad
    assert math.sqrt(m.rho2_hi) == pytest.approx(2 * 0.885, abs=1e-6)
    rng = np.random.default_rng(3)
    lo, hi = chain.movable_joints[2].lower, chain.movable_joints[2].upper
    n_in = n_out = 0
    for _ in range(400):
        q = rng.uniform(-math.pi, math.pi, 6)
        T_tip = chain.fk(q)
        T_tcp = T_tip @ T_ft
        ok = bool(m.ok(T_tcp[:3, :3][None], T_tcp[:3, 3][None])[0])
        inside = lo <= q[2] <= hi
        assert ok == inside, (q, ok)
        n_in += inside
        n_out += not inside
    assert n_in > 100 and n_out > 50
