"""실행 뒤 실험 (scan_l0.experiments) 의 순수 부분: 블록 해석 (기본값 유도·거부), 센서 거리 창 모형, 끝 뷰 자세 (판 거리·기울기 유지,
선 가운데 = 목표), 가짜 센서가 회전한 레이저 선을 직각 판 끝 위에 그리는지 (프로파일 한 장 안의 단차: 판 상면 → 끝 너머).
가상 팔 종단은 test_sim_experiments.py."""
from __future__ import annotations

import math

import numpy as np
import pytest

from groove_world import make_world
from scan_l0 import config as C
from scan_l0.experiments import end_view_pose, parse_experiments, plate_window, stricter_confirm, view_target
from sim_harness import CALIB

PASS_M, TRANSIT_M = 0.332, 0.443


def parse(block):
    return parse_experiments(block, pass_plate_m=PASS_M, transit_plate_m=TRANSIT_M, dwell_frames=21, retreat_default_m=0.1)


def test_parse_relative_distances_directions_and_derived_defaults():
    p = parse({"standoff": {"plate_distances_m": ["pass", "pass+0.06", "pass + 0.119", 0.40], "directions": [1, -1]},
               "end_views": {"angles_deg": [45, 90], "repeats": 5, "frames": 50, "retreat_m": 0.03, "ends": ["S", "E"]}})
    assert p.order == ["standoff", "end_views"]
    assert p.standoff.plate_distances_m == pytest.approx([0.332, 0.392, 0.451, 0.40])
    assert p.standoff.directions == [1, -1] and not p.standoff.derived
    v = p.end_views
    assert (v.angles_deg, v.repeats, v.frames, v.retreat_m, v.ends, v.plates) == ([45.0, 90.0], 5, 50, 0.03, ["S", "E"], ["left", "right"])
    # derived: plate distances = pass, mid, transit; frames = the run's dwell grab; retreat = the acquisition's (job safe_clearance_m)
    q = parse({"end_views": {"angles_deg": [90], "repeats": 1}, "standoff": {}})
    assert q.order == ["end_views", "standoff"]
    assert q.standoff.derived and q.standoff.plate_distances_m == pytest.approx([PASS_M, 0.5 * (PASS_M + TRANSIT_M), TRANSIT_M])
    assert q.end_views.frames == 21 and q.end_views.retreat_m == 0.1 and set(q.end_views.derived) == {"frames", "retreat_m"}
    assert parse(None) is None and parse({}) is None


@pytest.mark.parametrize("block", [
    {"standof": {}},                                                                    # unknown experiment
    {"standoff": {"plate_distances_m": ["pas+0.06"]}},                                  # bad relative distance
    {"standoff": {"plate_distances_m": [-0.3]}},
    {"standoff": {"directions": [1, 1]}},
    {"standoff": {"speed": 0.02}},                                                      # unknown key (the speed is the job's)
    {"end_views": {"repeats": 3}},                                                      # the angles are the experiment's design
    {"end_views": {"angles_deg": [120], "repeats": 1}},
    {"end_views": {"angles_deg": [45], "repeats": 0}},
    {"end_views": {"angles_deg": [45], "repeats": 1, "ends": ["S", "X"]}},
    {"end_views": {"angles_deg": [45], "repeats": 1, "confirm": "sometimes"}},
])
def test_parse_refuses_bad_blocks(block):
    with pytest.raises(ValueError):
        parse(block)


def test_stricter_confirm_and_sensor_window():
    assert stricter_confirm("auto", "each") == "each" and stricter_confirm("each", "auto") == "each"
    assert stricter_confirm("scan", None) == "scan"
    kw = dict(near_m=0.245, far_m=0.544, pose_tol_m=0.002)
    assert plate_window(0.332, 0.010, -5.0, **kw) is None and plate_window(0.451, 0.010, -5.0, **kw) is None
    assert "near limit" in plate_window(0.240, 0.010, -5.0, **kw)
    assert "far limit" in plate_window(0.535, 0.010, -5.0, **kw)            # the groove bottom 545 mm is beyond 544 - 2


def test_end_view_pose_keeps_the_plate_distance_and_the_tilt_and_centres_the_line():
    rng = np.random.default_rng(3)
    n = np.array([0.02, -0.01, 1.0])
    n /= np.linalg.norm(n)
    # a pass pose: optical Z down along -n tilted by the pitch about the line axis X
    X0 = np.cross(n, [0.0, 1.0, 0.0])
    X0 /= np.linalg.norm(X0)
    Y0 = np.cross(-n, X0)
    p = math.radians(-5.0)
    Z = -n * math.cos(p) + Y0 * math.sin(p)
    Y = np.cross(Z, X0)
    T0 = np.eye(4)
    T0[:3, :3] = np.column_stack([X0, Y, Z])
    T0[:3, 3] = np.array([0.6, 0.0, 0.40])
    plane_pt = np.array([0.6, 0.0, 0.05])
    def hit(T):
        o, z = T[:3, 3], T[:3, 2]
        t = float((plane_pt - o) @ n) / float(z @ n)
        return o + t * z, t
    _, t0 = hit(T0)
    for psi in (math.radians(45), math.radians(-90), math.radians(30)):
        P = plane_pt + rng.normal(0, 0.05, 3)
        P = P - n * float((P - plane_pt) @ n)                                 # on the plate top
        T = end_view_pose(T0, n, P, psi)
        Q, t = hit(T)
        assert np.allclose(Q, P, atol=1e-12) and t == pytest.approx(t0, abs=1e-12)          # line centre on the target, same distance
        assert float(T[:3, 2] @ n) == pytest.approx(float(T0[:3, 2] @ n), abs=1e-12)       # same tilt to the plate
        # the laser line (optical X) turned by psi about the plate normal
        x0p = X0 - n * float(X0 @ n)
        xp = T[:3, 0] - n * float(T[:3, 0] @ n)
        ang = math.atan2(float(np.cross(x0p, xp) @ n), float(x0p @ xp))
        assert ang == pytest.approx(psi, abs=1e-9)


@pytest.mark.parametrize("psi_deg", [45.0, -45.0, 90.0])
def test_fake_sensor_draws_the_rotated_line_with_an_in_profile_step_at_a_square_plate_end(psi_deg):
    """The synthetic world is a full ray cast (scan_core.synth over the plates' height map with square ends at s = 0 and s = L): a laser
    line turned by psi and centred on a plate's top at its end shows, in ONE profile, the plate top on the inside part of the line and
    the support (table_drop) beyond the end; the last plate-top point along the line is at the true end within the column spacing."""
    calib = C.load_calib(CALIB)
    w = make_world(calib, n_points=1408, table_drop_m=0.030, sigma_z_mm=0.0, dropout=0.0)
    # a pass-like optical pose (pitch -5 deg) at the plate distance 332 mm over the S end, then the end view on the left (+lateral) plate
    T_pass = w.opt_pose(0.0, standoff_m=0.342, pitch_deg=-5.0)
    lat = 0.040
    top = w.S + w.lateral * lat                                                # plate top at the S end (s = 0), 40 mm off the seam
    T_view = end_view_pose(T_pass, w.n, top, math.radians(psi_deg))
    x, z = w.render(T_view)
    P = w.point_base(x, z, T_view)
    ok = np.all(np.isfinite(P), axis=1)
    s = (P[ok] - w.S) @ w.u
    h = (P[ok] - w.S) @ w.n
    on_top = np.abs(h) < 1e-4
    beyond = h < -0.025                                                        # the support 30 mm below the top
    assert on_top.sum() > 100 and beyond.sum() > 100, (on_top.sum(), beyond.sum())
    assert np.all(s[on_top] >= -1e-6)                                          # the plate top only inside the specimen (s >= 0)
    # along the line: the plate-top points end at the true end (s = 0) within the column spacing projected on the seam
    xo = np.asarray(x, dtype=float)[ok]
    xs = xo[on_top]
    k_edge = np.argmin(s[on_top])
    dx = float(np.median(np.diff(np.sort(xo)))) * 1e-3
    assert abs(float(s[on_top][k_edge])) <= dx * abs(math.sin(math.radians(psi_deg))) + 2e-4, (s[on_top][k_edge], dx)
    # the step lies inside the profile (not at its border): plate top on one side of the edge column, the support on the other
    xb = xo[beyond]
    assert np.all(xb > xs[k_edge] - 1e-6) or np.all(xb < xs[k_edge] + 1e-6), "the support is on one side of the edge"


def test_view_target_is_mid_band_on_the_plate_top():
    P, wl = view_target(np.zeros(3), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([-1.0, 0.0, 0.0]), 0.00827, 0.0729,
                        0.010)
    assert wl == pytest.approx(0.5 * (0.00827 + 0.0729)) and np.allclose(P, [-wl, 0.0, 0.010])
